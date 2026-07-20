"""BinaryMatchboxNet: MatchboxNet-BxRxC with CLAUDE.md 2.2's precision overlay.

Assembly is 100% config-driven -- every width, kernel, stride, dilation and
precision comes from `ModelConfig.stages`. The code knows *how* to build a
stage kind, never *which* sizes to use.

Binary TCS sub-block (CLAUDE.md 2.3), as implemented:

    x in {-1,+1}
      -> depthwise binary conv        (XNOR + popcount)
      -> BN -> sign                   (*)
      -> pointwise binary conv        (XNOR + popcount)
      [last sub-block: + residual, added to the integer accumulator]
      -> BN -> sign -> {-1,+1}

(*) The inner BN->sign between depthwise and pointwise is implied by the
charter's requirement that BOTH convs be XNOR/popcount: XNOR needs {-1,+1}
inputs, and a depthwise accumulator is an integer, not ±1. At export each
BN->sign pair fuses into one integer threshold (Cerutti eq. 3), so the
hardware block is: popcount -> threshold -> popcount -> add residual ->
threshold. No multipliers anywhere.

Residual (CLAUDE.md 2.3): joins the pointwise *integer accumulator* before
the final BN, so the whole block ends in exactly one threshold. Because the
hardware adds two integers there, the final pointwise and the skip projection
run with scale=False even when XNOR-Net alpha scaling is enabled elsewhere --
alpha would make the software accumulator non-integer and silently break
BN-to-threshold fusion. Interior convs may keep alpha: their own BN absorbs
a positive per-channel scale exactly.

Activation choice between plain stages: sign if the next stage is binary
(it needs ±1), ReLU otherwise, nothing after the last conv. This keeps the
epilogue (conv3 -> conv4 -> pool) in a continuous feature space, which is the
entire reason those layers are not binary.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.binary_ops import (
    BinaryConv1d,
    DepthwiseBinaryConv1d,
    PointwiseBinaryConv1d,
    same_padding,
    sign_ste,
)
from models.quant_ops import QuantConv1d
from train.config import ModelConfig, StageConfig

# STE clip for *activation* binarization. Pre-sign values are BN outputs
# (roughly unit variance), so the standard clip of 1 is appropriate.
_ACT_CLIP = 1.0


def _make_conv(precision: str, in_ch: int, out_ch: int, kernel: int,
               *, stride: int = 1, dilation: int = 1, groups: int = 1,
               bias: bool = False, scale: bool = True,
               ste_clip: float = 1.0) -> nn.Conv1d:
    """Instantiate a conv of the requested precision. Sizes come from config."""
    pad = same_padding(kernel, dilation)
    if precision == "binary":
        return BinaryConv1d(in_ch, out_ch, kernel, stride=stride, padding=pad,
                            dilation=dilation, groups=groups, bias=bias,
                            scale=scale, ste_clip=ste_clip)
    if precision == "int8":
        return QuantConv1d(in_ch, out_ch, kernel, stride=stride, padding=pad,
                           dilation=dilation, groups=groups, bias=bias, bits=8)
    if precision in ("fixed", "fp32"):
        # "fixed" trains in full precision; fixed-point conversion is export's
        # job (models/quant_ops.py docstring explains why).
        return nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad,
                         dilation=dilation, groups=groups, bias=bias)
    raise ValueError(f"unknown precision {precision!r}")


class _TCSSub(nn.Module):
    """dw -> BN -> sign -> pw. Returns the pointwise integer accumulator."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int,
                 mcfg: ModelConfig, *, final: bool) -> None:
        super().__init__()
        self.dw = DepthwiseBinaryConv1d(
            in_ch, kernel, padding=same_padding(kernel, dilation),
            dilation=dilation, scale=mcfg.scale_binary_weights,
            ste_clip=mcfg.weight_ste_clip)
        self.bn = nn.BatchNorm1d(in_ch, eps=mcfg.bn_eps, momentum=mcfg.bn_momentum)
        # final sub-block's pw feeds the integer residual add -> never scaled
        self.pw = PointwiseBinaryConv1d(
            in_ch, out_ch,
            scale=mcfg.scale_binary_weights and not final,
            ste_clip=mcfg.weight_ste_clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(sign_ste(self.bn(self.dw(x)), _ACT_CLIP))


class BinaryTCSBlock(nn.Module):
    """R binary TCS sub-blocks + integer-domain residual around the block."""

    def __init__(self, in_ch: int, out_ch: int, stage: StageConfig,
                 mcfg: ModelConfig) -> None:
        super().__init__()
        if stage.precision != "binary":
            raise ValueError(
                f"stage {stage.name}: TCS residual blocks are binary-only "
                f"(CLAUDE.md 2.2), got {stage.precision!r}")
        if stage.stride != 1:
            raise ValueError(
                f"stage {stage.name}: residual blocks require stride 1, "
                f"got {stage.stride}")

        R = stage.n_sub_blocks
        self.subs = nn.ModuleList()
        self.post_bns = nn.ModuleList()
        ch = in_ch
        for r in range(R):
            self.subs.append(_TCSSub(ch, out_ch, stage.kernel, stage.dilation,
                                     mcfg, final=(r == R - 1)))
            self.post_bns.append(nn.BatchNorm1d(out_ch, eps=mcfg.bn_eps,
                                                momentum=mcfg.bn_momentum))
            ch = out_ch

        # skip: identity when channels match (adds raw ±1), else an unscaled
        # binary 1x1 projection (integer popcount output)
        self.skip: nn.Module
        if in_ch == out_ch:
            self.skip = nn.Identity()
        else:
            self.skip = PointwiseBinaryConv1d(in_ch, out_ch, scale=False,
                                              ste_clip=mcfg.weight_ste_clip)
        self.drop = nn.Dropout(stage.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)                       # integer domain
        y = x
        last = len(self.subs) - 1
        for i, (sub, bn) in enumerate(zip(self.subs, self.post_bns)):
            acc = sub(y)                         # integer accumulator
            if i == last:
                acc = acc + res                  # integer + integer
            y = self.drop(sign_ste(bn(acc), _ACT_CLIP))
        return y


class PlainStage(nn.Module):
    """Prologue/epilogue conv stage: conv [-> BN -> activation].

    activation: "sign" (next stage is binary), "relu", or "none" (final
    stage: no BN, bias on, straight to the pooling head).
    """

    def __init__(self, in_ch: int, out_ch: int, stage: StageConfig,
                 mcfg: ModelConfig, activation: str) -> None:
        super().__init__()
        self.activation = activation
        final = activation == "none"
        conv_kw = dict(stride=stage.stride, dilation=stage.dilation,
                       bias=final, scale=mcfg.scale_binary_weights,
                       ste_clip=mcfg.weight_ste_clip)

        if stage.separable:
            # time-channel separable, non-residual (e.g. conv2 ablation).
            # Binary needs the inner BN->sign for the pointwise XNOR; int8+
            # can chain dw->pw directly.
            self.dw = _make_conv(stage.precision, in_ch, in_ch, stage.kernel,
                                 groups=in_ch, **conv_kw)
            if stage.precision == "binary":
                self.dw_bn: nn.Module = nn.BatchNorm1d(
                    in_ch, eps=mcfg.bn_eps, momentum=mcfg.bn_momentum)
            else:
                self.dw_bn = nn.Identity()
            self.pw = _make_conv(stage.precision, in_ch, out_ch, 1,
                                 bias=final, scale=mcfg.scale_binary_weights,
                                 ste_clip=mcfg.weight_ste_clip)
            self.conv = self.pw          # `.conv` = the stage's defining conv
        else:
            self.dw = None
            self.conv = _make_conv(stage.precision, in_ch, out_ch,
                                   stage.kernel, **conv_kw)

        self.bn = (nn.Identity() if final else
                   nn.BatchNorm1d(out_ch, eps=mcfg.bn_eps,
                                  momentum=mcfg.bn_momentum))
        self.drop = nn.Dropout(stage.dropout)
        self._binary_sep = stage.separable and stage.precision == "binary"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dw is not None:
            x = self.dw(x)
            if self._binary_sep:
                x = sign_ste(self.dw_bn(x), _ACT_CLIP)
            x = self.pw(x)
        else:
            x = self.conv(x)
        x = self.bn(x)
        if self.activation == "sign":
            x = sign_ste(x, _ACT_CLIP)
        elif self.activation == "relu":
            x = F.relu(x)
        return self.drop(x)


class BinaryMatchboxNet(nn.Module):
    """Config-assembled BinaryMatchboxNet-BxRxC. Input [B, 16, T] in {-1,+1},
    output [B, n_classes] logits (softmax lives in the loss)."""

    def __init__(self, mcfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = mcfg
        self.stages = nn.ModuleDict()

        ch = mcfg.in_channels
        n = len(mcfg.stages)
        for i, st in enumerate(mcfg.stages):
            out_ch = st.out_channels(mcfg.C, mcfg.n_classes)
            if st.residual:
                if not st.separable:
                    raise NotImplementedError(
                        f"stage {st.name}: residual is only supported on "
                        f"separable TCS blocks")
                self.stages[st.name] = BinaryTCSBlock(ch, out_ch, st, mcfg)
            else:
                if i == n - 1:
                    act = "none"
                elif mcfg.stages[i + 1].precision == "binary":
                    act = "sign"
                else:
                    act = "relu"
                self.stages[st.name] = PlainStage(ch, out_ch, st, mcfg, act)
            ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages.values():
            x = stage(x)
        # avg pool over time -> [B, n_classes]; CrossEntropyLoss applies softmax
        return F.adaptive_avg_pool1d(x, 1).squeeze(-1)

    # ------------------------------------------------------------------ #
    def summary(self) -> List[Tuple[str, int, int]]:
        """[(stage name, out_channels, n_params)], covering every parameter."""
        rows = []
        for st in self.cfg.stages:
            module = self.stages[st.name]
            rows.append((st.name,
                         st.out_channels(self.cfg.C, self.cfg.n_classes),
                         sum(p.numel() for p in module.parameters())))
        return rows

    def describe(self) -> str:
        rows = self.summary()
        prec = {s.name: s.precision for s in self.cfg.stages}
        total = sum(r[2] for r in rows)
        binary = sum(r[2] for r in rows if prec[r[0]] == "binary")
        lines = [f"{'stage':<8}{'out_ch':>8}{'params':>10}  precision",
                 "-" * 38]
        lines += [f"{n:<8}{c:>8}{p:>10,}  {prec[n]}" for n, c, p in rows]
        lines.append("-" * 38)
        lines.append(f"total {total:,} params "
                     f"({binary / total:.0%} in binary stages)")
        return "\n".join(lines)
