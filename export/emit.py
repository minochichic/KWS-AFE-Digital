"""Turn a trained checkpoint into the files RTL reads: manifest + ROMs + header.

The rule this exists to enforce (docs/ICD.md, CLAUDE.md 5): RTL source holds no
numbers. Shapes and widths arrive through a generated `parameters.vh`, weights
and thresholds arrive through `$readmemh` ROMs. Retraining then costs a re-run
of this script and a bitstream rebuild -- never an RTL edit.

WIDTHS COME FROM THE ANALYTIC BOUND, NOT FROM MEASUREMENT. A binary accumulator
sums n_terms values of +-1, so it lives in [-n_terms, +n_terms] and cannot leave
it. P5-1 measured the real ranges on xl_g12 and found the bound is not loose
enough to be worth exploiting: depthwise reaches it exactly, pointwise saves one
bit, and a guard bit for out-of-distribution inputs gives that back. The bound
needs no guard and no saturation logic. See docs/ROADMAP.md P5-1.

WHAT FUSES AND WHAT DOES NOT. `sign(BN(acc))` collapses to one integer compare
(export/fuse.py), so every stage whose activation is `sign` needs only a
threshold and a polarity bit -- no BN, no alpha, no multiplier. But the network
does not end that way. Stage activations are chosen by the NEXT stage's
precision, so with the stage list in configs/base.yaml:

    conv1      -> sign    (b1 is binary)      fuses
    b1/b2/b3   -> sign    (residual blocks)   fuses
    conv2.dw   -> sign    (inner re-binarize) fuses
    conv2.pw   -> relu    (conv3 is int8)     DOES NOT FUSE
    conv3      -> relu    (conv4 is fixed)    DOES NOT FUSE
    conv4      -> none    logits              DOES NOT FUSE

So the head is XNOR-popcount plus compares, and the tail is a small fixed-point
section. Pretending otherwise would produce RTL that silently computes the wrong
thing, so the manifest labels every layer with which one it is.

Run (needs the checkpoint, so on the training box):
    python -m export.emit --tag xl_g12
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from export.fuse import fuse_bn_to_threshold, conv_alpha
from export.pack import pack_pm1, to_hex_words, WORD_BITS
from export.tailbuild import check_site, fixed_weights, tail_plan
from export.tailfmt import FRAC_BITS
from models.binary_ops import BinaryConv1d
from models.binary_matchboxnet import BinaryMatchboxNet, BinaryTCSBlock, PlainStage
from models.quant_ops import QuantConv1d


def signed_bits(lo: int, hi: int) -> int:
    """Two's-complement width holding [lo, hi]. Mirrors export/ranges.py."""
    return 1 + max(1, math.ceil(math.log2(max(int(hi) + 1, -int(lo), 1))))


@dataclass
class Layer:
    """One RTL-visible operation."""

    name: str
    op: str                       # binary_dw | binary_pw | int8 | fixed
    in_ch: int
    out_ch: int
    kernel: int = 1
    stride: int = 1
    dilation: int = 1
    padding: int = 0
    groups: int = 1
    n_terms: int = 0              # |accumulator| bound
    acc_bits: int = 0
    epilogue: str = "none"        # threshold | bn_relu | logits
    weights: Optional[str] = None       # ROM basename
    weight_words: int = 0
    thresholds: Optional[str] = None    # ROM basename
    residual_from: Optional[str] = None
    input_binary: bool = True     # is this layer fed {-1,+1}?
    notes: str = ""


class Emitter:
    def __init__(self, model: BinaryMatchboxNet, out: Path) -> None:
        self.model = model
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.layers: List[Layer] = []
        # the AFE hands over {-1,+1}; each stage updates it from its
        # own output activation, which is the actual dataflow
        self._stage_in_binary = True
        self._T = 0
        self.roms: Dict[str, Dict[str, Any]] = {}

    # -- ROM writers ---------------------------------------------------- #
    def _write_hex(self, name: str, words: List[str], meta: Dict[str, Any]) -> None:
        p = self.out / f"{name}.hex"
        p.write_text("\n".join(words) + "\n")
        self.roms[name] = {"file": p.name, "n_words": len(words), **meta}

    def _binary_weights(self, name: str, conv: nn.Module) -> int:
        """sign(W) bit-packed along the axis RTL streams."""
        w = torch.sign(conv.weight.detach())
        w = torch.where(w == 0, torch.ones_like(w), w)       # sign(0) -> +1
        if conv.groups == conv.weight.shape[0]:              # depthwise [C,1,K]
            flat = w.reshape(w.shape[0], w.shape[2])         # pack along K
            axis = "kernel"
        else:                                                # pointwise [O,I,1]
            flat = w.reshape(w.shape[0], w.shape[1])         # pack along in_ch
            axis = "in_ch"
        packed = pack_pm1(flat)
        self._write_hex(name, to_hex_words(packed),
                        {"kind": "binary", "shape": list(flat.shape),
                         "n_valid": packed.n_valid, "word_bits": WORD_BITS,
                         "pack_axis": axis,
                         "note": "-1->0, +1->1, LSB first; tail bits are real "
                                 "-1 and must be masked before popcount"})
        return packed.words.numel()

    def _thresholds(self, name: str, bn: nn.BatchNorm1d,
                    alpha: Optional[torch.Tensor]) -> None:
        """sign(BN(alpha*n)) -> one integer compare per output channel."""
        f = fuse_bn_to_threshold(bn, alpha)
        words = [f"{int(t) & 0xFFFFFFFF:08x}" for t in f.t.tolist()]
        words += [f"{int(g):08x}" for g in f.take_ge.to(torch.int64).tolist()]
        self._write_hex(name, words,
                        {"kind": "threshold", "n_channels": int(f.t.numel()),
                         "layout": "n int32 thresholds (two's complement), then "
                                   "n polarity words (1 = fire on acc >= t)",
                         "rule": "out = +1 if (acc >= t) == take_ge else -1"})

    # -- accumulator bounds --------------------------------------------- #
    @staticmethod
    def _binary_terms(conv: nn.Module) -> int:
        w = conv.weight
        return int(w.shape[1] * w.shape[2])          # (in/groups) * kernel

    @staticmethod
    def _int8(conv: QuantConv1d) -> tuple:
        """(q, scale) exactly as the module itself quantized during training.

        Deliberately NOT export/pack.quantize_int8: that clamps amax at 1e-12
        before dividing while QuantConv1d clamps the scale at 1e-8 after. The
        two agree on every real weight and disagree on a degenerate channel,
        which is precisely the kind of difference that surfaces once and is
        untraceable. Ask the module.
        """
        s = conv.weight_scale()                             # [O,1,1], detached
        q = torch.round(conv.weight.detach() / s).clamp(-conv.qmax, conv.qmax)
        return q.to(torch.int8), s.reshape(-1)

    @classmethod
    def _int8_terms(cls, conv: QuantConv1d) -> int:
        """Bound for int8 weights against +-1 activations: max_o sum|q|.

        ONLY valid when the input really is +-1. sum|q| is what you get when
        every activation is at magnitude 1 and agrees in sign with its weight;
        against a real-valued input it bounds nothing. Callers must check --
        `_mark_real_inputs` clears n_terms for any layer whose predecessor does
        not end in a threshold.

        Tighter than 127*K*C_in and just as safe where it applies: the
        quantized weights are known at export time, so this is still a bound.
        """
        q, _ = cls._int8(conv)
        return int(q.abs().to(torch.int64).flatten(1).sum(dim=1).max())

    def _mark_real_inputs(self) -> None:
        """Drop the bound on any layer that is not fed +-1.

        `input_binary` is recorded during emission, from the dataflow, NOT by
        walking the layer list. The list order is not the dataflow: a block's
        skip projection is emitted first (it branches off the block input) and
        ends in `none`, so a positional rule would decide the depthwise right
        after it is fed real numbers, when it is fed the same +-1 the skip is.
        Harmless while only int8/fixed layers are cleared, wrong the moment
        anything else depends on it.
        """
        for l in self.layers:
            if not l.input_binary and l.op in ("int8", "fixed"):
                l.n_terms = 0
                l.notes = (l.notes + "  " if l.notes else "") + (
                    "input is real (previous stage ends in relu), so sum|q| is "
                    "not a bound; acc_bits comes from tailfmt instead -- "
                    "integer weights times a clamped fixed-point input, which "
                    "is a bound of the same kind, just from a different pair "
                    "of extremes.")
                # acc_bits is NOT cleared. It used to be, because the tail's
                # format was undecided and n_terms was the only source of a
                # width; now tailfmt gives these layers a real bound and
                # clearing it here would erase the number the RTL needs while
                # leaving the manifest looking complete.

    def _add(self, layer: Layer) -> Layer:
        layer.acc_bits = signed_bits(-layer.n_terms, layer.n_terms)
        self.layers.append(layer)
        return layer

    # -- walkers --------------------------------------------------------- #
    def _plain(self, name: str, st: Any, mod: PlainStage) -> None:
        act = mod.activation
        stage_in = self._stage_in_binary
        if mod.dw is not None:                        # separable
            dw, pw = mod.dw, mod.pw
            if isinstance(dw, BinaryConv1d):
                n = f"{name}_dw"
                self._add(Layer(n, "binary_dw", dw.in_channels, dw.out_channels,
                                dw.kernel_size[0], dw.stride[0], dw.dilation[0],
                                dw.padding[0], dw.groups, self._binary_terms(dw),
                                epilogue="threshold", weights=f"{n}_w",
                                weight_words=self._binary_weights(f"{n}_w", dw),
                                input_binary=stage_in))
                self._thresholds(f"{n}_t", mod.dw_bn, conv_alpha(dw))
                self.layers[-1].thresholds = f"{n}_t"
            else:
                raise NotImplementedError(
                    f"{name}: separable non-binary depthwise is not exported "
                    f"(config has precision={st.precision!r})")
            conv, cname = pw, f"{name}_pw"
        else:
            conv, cname = mod.conv, name

        if isinstance(conv, BinaryConv1d):
            op, terms = "binary_pw" if conv.kernel_size[0] == 1 else "binary_dw", \
                self._binary_terms(conv)
            wname = f"{cname}_w"
            words = self._binary_weights(wname, conv)
        elif isinstance(conv, QuantConv1d):
            op, terms = "int8", self._int8_terms(conv)
            wname = f"{cname}_w"
            q, scale = self._int8(conv)
            self._write_hex(wname, [f"{int(v) & 0xFF:02x}"
                                    for v in q.reshape(-1).tolist()],
                            {"kind": "int8", "shape": list(q.shape),
                             "scale": [float(s) for s in scale],
                             "note": "two's complement int8, row-major"})
            words = q.numel()
        else:                                          # fixed-point head (conv4)
            op, terms, wname = "fixed", 0, f"{cname}_w"
            q = fixed_weights(conv)
            amax = int(q.abs().max())
            self._write_hex(wname, [f"{int(v) & 0xFFFFFFFF:08x}"
                                    for v in q.reshape(-1).tolist()],
                            {"kind": "fixed", "shape": list(q.shape),
                             "frac_bits": FRAC_BITS,
                             "weight_bits": signed_bits(-amax, amax),
                             "abs_max": amax,
                             "bias": ([float(b) for b in conv.bias.detach()]
                                      if conv.bias is not None else None),
                             "note": f"two's complement, row-major, one shared "
                                     f"scale 2^-{FRAC_BITS} -- the grid "
                                     f"experiments/tail_fixedpoint.py measured"})
            words = q.numel()

        epi = {"sign": "threshold", "relu": "bn_relu", "none": "logits"}[act]
        lay = self._add(Layer(cname, op, conv.in_channels, conv.out_channels,
                              conv.kernel_size[0], conv.stride[0],
                              conv.dilation[0], conv.padding[0], conv.groups,
                              terms, epilogue=epi, weights=wname,
                              weight_words=words,
                              # a separable stage re-binarizes between dw and pw,
                              # so the pointwise is fed +-1 regardless of what
                              # the stage itself was handed
                              input_binary=(True if mod.dw is not None
                                            else stage_in)))
        if epi == "threshold":
            # alpha for an int8 conv is its per-channel scale: the real
            # accumulator is scale_o * (integer accumulator), and fuse() folds
            # that positive factor into the threshold exactly as it does XNOR's.
            self._thresholds(f"{cname}_t", mod.bn,
                             conv_alpha(conv) if isinstance(conv, BinaryConv1d)
                             else self._int8(conv)[1])
            lay.thresholds = f"{cname}_t"
        elif epi == "bn_relu":
            lay.notes = ("BN + relu survive: the next stage is not binary, so "
                         "there is no sign() to fold into. Emitted as an "
                         "integer gain + offset + shift; see the `tail` section "
                         "of this manifest and export/tailfmt.py.")
        else:
            lay.notes = ("logits -> avg-pool over time -> argmax. The pool's "
                         "divide by T is not built: it is the same positive "
                         "factor on every class, and argmax ignores it.")
        self._stage_in_binary = act == "sign"

    # -- the tail: BN survives as arithmetic, so it needs ROMs of its own -- #
    def _emit_tail(self) -> List[Dict[str, Any]]:
        """One `<layer>_bn` ROM per tail site, from export/tailbuild.py.

        Run after the stage walk rather than inside it, because the folds thread
        a format forward through the tail and that is easier to get right in one
        pass over the tail alone than interleaved with the binary layers. The
        arithmetic itself is in tailbuild so golden.py derives it from the same
        place -- a ROM and an expected vector computed from two copies of this
        would disagree, and the disagreement would look like an RTL bug.
        """
        by_name = {l.name: l for l in self.layers}
        rows: List[Dict[str, Any]] = []
        for s in tail_plan(self.model):
            check_site(s)
            lay = by_name.get(s.name)
            if lay is None:
                raise ValueError(
                    f"tail site {s.name!r} has no manifest layer -- tailbuild's "
                    f"naming has drifted from Emitter._plain's")
            words = [f"{int(a) & 0xFFFFFFFF:08x}" for a in s.fold.gain]
            words += [f"{int(b) & 0xFFFFFFFF:08x}" for b in s.fold.bias]
            err = s.fold.max_output_error_lsb((1 << (s.acc_bits - 1)) - 1)
            self._write_hex(f"{s.name}_bn", words, {
                "kind": "affine",
                "n_channels": s.fold.n_channels,
                "shift": s.fold.shift,
                "relu": s.fold.relu,
                "in_format": None if s.in_fmt is None else str(s.in_fmt),
                "out_format": str(s.out_fmt),
                "out_bits": s.out_fmt.bits,
                "acc_bits": s.acc_bits,
                "gain_bits": s.fold.gain_bits_used(),
                "quietest_gain_bits": s.fold.quietest_gain_bits(),
                "max_output_error_lsb": round(err, 5),
                "weight_absmax": s.weight_absmax,
                "layout": "n int32 gains (two's complement), then n int32 offsets",
                "rule": "y = (A[c]*acc + B[c] + (1<<(shift-1))) >>> shift; then "
                        "relu if set, then saturate to out_bits",
                **s.meta})
            lay.thresholds = f"{s.name}_bn"
            lay.acc_bits = s.acc_bits
            row = {"name": s.name, "epilogue": s.kind,
                   "out_format": str(s.out_fmt), "shift": s.fold.shift,
                   "acc_bits": s.acc_bits, "n_out": s.fold.n_channels,
                   "max_output_error_lsb": round(err, 5)}
            if s.kind == "logits":
                # the pool sums T frames and never divides, so it is T times
                # wider than one logit
                row["pool_acc_bits"] = s.out_fmt.bits + (self._T - 1).bit_length()
            rows.append(row)
        return rows

    def _tcs(self, name: str, mod: BinaryTCSBlock) -> None:
        # every conv in a TCS block is fed +-1: the block input is a
        # sign() output, and dw->BN->sign re-binarizes before the pw
        assert self._stage_in_binary, f"{name}: TCS block fed non-binary"
        last = len(mod.subs) - 1
        skip_terms = 0
        if isinstance(mod.skip, BinaryConv1d):
            n = f"{name}_skip"
            skip_terms = self._binary_terms(mod.skip)
            self._add(Layer(n, "binary_pw", mod.skip.in_channels,
                            mod.skip.out_channels, 1, 1, 1, 0, 1, skip_terms,
                            epilogue="none", weights=f"{n}_w",
                            weight_words=self._binary_weights(f"{n}_w", mod.skip),
                            notes="projection for the residual; unscaled, its "
                                  "integer accumulator is added directly"))
        else:
            skip_terms = 1          # identity: adds the +-1 activation itself

        for i, (sub, bn) in enumerate(zip(mod.subs, mod.post_bns)):
            for tag, conv, op, inner_bn in (
                    ("dw", sub.dw, "binary_dw", sub.bn),
                    ("pw", sub.pw, "binary_pw", None)):
                n = f"{name}_s{i}_{tag}"
                terms = self._binary_terms(conv)
                lay = self._add(Layer(
                    n, op, conv.in_channels, conv.out_channels,
                    conv.kernel_size[0], conv.stride[0], conv.dilation[0],
                    conv.padding[0], conv.groups, terms,
                    epilogue="threshold", weights=f"{n}_w",
                    weight_words=self._binary_weights(f"{n}_w", conv)))
                if inner_bn is not None:               # dw -> BN -> sign
                    self._thresholds(f"{n}_t", inner_bn, conv_alpha(conv))
                    lay.thresholds = f"{n}_t"
                elif i != last:                        # pw -> post_bn -> sign
                    self._thresholds(f"{n}_t", bn, conv_alpha(conv))
                    lay.thresholds = f"{n}_t"
                else:                                  # pw + skip -> BN -> sign
                    lay.epilogue = "none"
                    lay.notes = "feeds the residual add; unscaled by construction"

            if i == last:
                n = f"{name}_add"
                add = self._add(Layer(
                    n, "residual_add", mod.subs[i].pw.out_channels,
                    mod.subs[i].pw.out_channels,
                    n_terms=self._binary_terms(mod.subs[i].pw) + skip_terms,
                    epilogue="threshold",
                    residual_from=(f"{name}_skip"
                                   if isinstance(mod.skip, BinaryConv1d)
                                   else "block input (identity)")))
                self._thresholds(f"{n}_t", bn, None)   # both addends unscaled
                add.thresholds = f"{n}_t"
        self._stage_in_binary = True          # a TCS block ends in sign()

    def run(self, cfg: Any, tag: str) -> Dict[str, Any]:
        self._T = int(cfg.model.T)
        for st in self.model.cfg.stages:
            mod = self.model.stages[st.name]
            if isinstance(mod, BinaryTCSBlock):
                self._tcs(st.name, mod)
            else:
                self._plain(st.name, st, mod)

        # the tail after the walk: its folds thread a format forward, and the
        # ROMs it writes give conv3/conv4 the acc_bits _mark_real_inputs would
        # otherwise have nothing to put there
        tail_rows = self._emit_tail()
        self._mark_real_inputs()
        # only sized layers count -- the binary bound does not apply to the
        # tail, whose widths come from tailfmt instead and are listed separately
        widest = max((l.acc_bits for l in self.layers if l.n_terms), default=0)
        man = {
            "tag": tag,
            "n_channels": int(cfg.afe.n_channels),
            "T": int(cfg.model.T),
            "native_T": int(cfg.afe.native_T),
            "pad_left": (int(cfg.model.T) - int(cfg.afe.native_T)) // 2,
            "pad_value": -1,
            "n_classes": int(cfg.model.n_classes),
            "C": int(cfg.model.C),
            "frame_ms": float(cfg.afe.envelope_win_ms),
            "datapath": "folded",
            "acc_bits_widest": widest,
            "acc_bits_source": "analytic bound +-n_terms, over the layers fed "
                               "+-1 (see _mark_real_inputs)",
            "unsized_layers": [l.name for l in self.layers
                               if not l.n_terms and not l.acc_bits],
            "tail": {
                "frac_bits": FRAC_BITS,
                "sites": tail_rows,
                "note": "the tail's widths come from the measured ranges and "
                        "the confirmed frac (rtl/README.md 3-07), not from the "
                        "binary +-n_terms bound. acc_bits here is a real bound "
                        "too: integer weights known at export time times a "
                        "clamped fixed-point input.",
            },
            "word_bits": WORD_BITS,
            "layers": [asdict(l) for l in self.layers],
            "roms": self.roms,
        }
        (self.out / "manifest.json").write_text(json.dumps(man, indent=2))
        return man


VH_HEADER = """// GENERATED by export/emit.py -- do not edit.
// Source: runs/{tag}/  |  regenerate after every retrain.
// RTL must take every dimension from here; no literals in the design files.
`ifndef KWS_PARAMETERS_VH
`define KWS_PARAMETERS_VH
"""


def write_parameters_vh(man: Dict[str, Any], path: Path) -> None:
    L = [VH_HEADER.format(tag=man["tag"])]
    a = L.append
    a(f"`define KWS_N_CH        {man['n_channels']}")
    a(f"`define KWS_T           {man['T']}")
    a(f"`define KWS_NATIVE_T    {man['native_T']}")
    a(f"`define KWS_PAD_LEFT    {man['pad_left']}   // pads are -1, not 0")
    a(f"`define KWS_N_CLASSES   {man['n_classes']}")
    a(f"`define KWS_FRAME_MS    {man['frame_ms']:.0f}")
    a(f"`define KWS_WORD_BITS   {man['word_bits']}")
    a(f"`define KWS_ACC_BITS    {man['acc_bits_widest']}"
      f"   // widest accumulator, analytic bound")
    a(f"`define KWS_N_LAYERS    {len(man['layers'])}")
    a("")
    for i, l in enumerate(man["layers"]):
        p = f"KWS_L{i}_{l['name'].upper()}"
        a(f"// {l['name']}: {l['op']}, epilogue={l['epilogue']}")
        for k in ("in_ch", "out_ch", "kernel", "stride", "dilation", "padding",
                  "groups", "n_terms", "acc_bits"):
            a(f"`define {p}_{k.upper():<10} {l[k]}")
        a("")

    # The tail. Every one of these is a number the RTL would otherwise have to
    # hardcode, and a retrain can move all of them: the shift depends on the
    # trained BN gains, the accumulator width on the quantized weights.
    tail = man.get("tail")
    if tail:
        a(f"// ---- tail fixed-point (frac={tail['frac_bits']}) ----")
        a(f"`define KWS_TAIL_FRAC   {tail['frac_bits']}")
        for s in tail["sites"]:
            p = f"KWS_{s['name'].upper()}"
            a(f"// {s['name']}: {s['epilogue']}, out {s['out_format']}")
            a(f"`define {p}_ACC_BITS  {s['acc_bits']}")
            a(f"`define {p}_OUT_BITS  {_fmt_bits(s['out_format'])}")
            a(f"`define {p}_SHIFT     {s['shift']}")
            a(f"`define {p}_N_OUT     {s['n_out']}")
            if "pool_acc_bits" in s:
                a(f"`define {p}_POOL_BITS {s['pool_acc_bits']}"
                  f"   // sum over T, no divide")
        a("")
    a("`endif")
    path.write_text("\n".join(L) + "\n")


def _fmt_bits(fmt: str) -> int:
    """'8.6' -> 14. The manifest carries the format as a string because that is
    what a human reads; the header needs the width."""
    i, f = fmt.split(".")
    return int(i) + int(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=None, help="default runs/<tag>/rtl")
    args = ap.parse_args()

    from train.config import load_config
    run = Path(args.runs) / args.tag
    cfg = load_config(str(run / "config.yaml"))
    model = BinaryMatchboxNet(cfg.model).eval()
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])

    out = Path(args.out) if args.out else run / "rtl"
    man = Emitter(model, out).run(cfg, args.tag)
    write_parameters_vh(man, out / "parameters.vh")

    print(f"{len(man['layers'])} layers, {len(man['roms'])} ROMs -> {out}")
    print(f"{'layer':<18}{'op':<14}{'in':>5}{'out':>5}{'k':>4}"
          f"{'terms':>7}{'acc':>5}  epilogue")
    for l in man["layers"]:
        print(f"{l['name']:<18}{l['op']:<14}{l['in_ch']:>5}{l['out_ch']:>5}"
              f"{l['kernel']:>4}{l['n_terms'] or '-':>7}"
              f"{l['acc_bits'] or '-':>5}  "
              f"{l['epilogue']}")
    print(f"\nwidest SIZED accumulator: {man['acc_bits_widest']} bits "
          f"(folded -> this is the one register that matters)")
    if man["unsized_layers"]:
        print(f"NOT sized: {', '.join(man['unsized_layers'])}")
    n_fuse = sum(1 for l in man["layers"] if l["epilogue"] == "threshold")
    print(f"{n_fuse}/{len(man['layers'])} layers fuse to an integer compare; "
          f"the rest need fixed-point arithmetic")

    t = man.get("tail")
    if t:
        print(f"\ntail, frac={t['frac_bits']}:")
        print(f"{'site':<10}{'out':>6}{'acc':>5}{'shift':>7}{'A bits':>8}"
              f"{'quiet':>7}{'err LSB':>9}")
        for s in t["sites"]:
            rom = man["roms"].get(f"{s['name']}_bn", {})
            print(f"{s['name']:<10}{s['out_format']:>6}{s['acc_bits']:>5}"
                  f"{s['shift']:>7}{rom.get('gain_bits', '-'):>8}"
                  f"{rom.get('quietest_gain_bits', '-'):>7}"
                  f"{rom.get('max_output_error_lsb', '-'):>9}")
        print("err LSB = how much the integer fold moves the result, in LSBs "
              "of that site's own output grid. One whole LSB is the step the "
              "frac sweep measured as costing 0.0pp, so these are noise.")


if __name__ == "__main__":
    main()
