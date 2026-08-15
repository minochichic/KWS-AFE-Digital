"""Layer-by-layer reference activations, so RTL can be debugged at all.

Without these, a wrong bit anywhere shows up only as "the twelve logits look
odd" and there is nothing to bisect. With them the testbench compares one layer
at a time and the first mismatch names the module.

WHAT IS DUMPED IS WHAT RTL COMPUTES, NOT WHAT THE MODULE RETURNS

Two places where the obvious choice is the wrong one:

* The accumulator is the *pre-alpha integer* -- `2*popcount(XNOR) - N`. A
  binary conv module returns `alpha * n`, which is not what any register in the
  design holds.
* The +-1 output is `FusedThreshold.apply(acc)`, i.e. the integer compare, not
  a hooked activation tensor. tests/test_export.py already pins that this
  equals `sign(BN(alpha*n))`, so dumping the compare keeps the files aligned
  with the ROMs rather than with PyTorch's intermediate shapes.

The residual add is assembled here too: `last_pw_acc + skip_acc`, both unscaled
by construction, thresholded once. That matches the `*_add` layer in the
manifest.

Real-valued layers (conv2_pw's bn_relu, conv3, conv4) are written as text and
labelled reference-only -- the tail's fixed-point format is still open.

NAMES MUST MATCH THE MANIFEST, or the testbench looks for files that do not
exist, compares nothing, and passes. tests/test_golden.py asserts the two name
sets are identical.

Run:  python -m export.golden --tag xl_g12 --clips 8
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from export.fuse import binary_accumulator, fuse_bn_to_threshold, conv_alpha
from export.pack import pack_pm1, to_hex_words
from models.binary_ops import BinaryConv1d
from models.binary_matchboxnet import BinaryMatchboxNet, BinaryTCSBlock, PlainStage
from models.quant_ops import QuantConv1d


@dataclass
class Site:
    """One manifest layer, with everything needed to reproduce its RTL output."""

    name: str
    conv: Optional[nn.Module]           # None for a residual add
    bn: Optional[nn.BatchNorm1d] = None  # set when the layer fuses to a compare
    scaled: bool = True                 # is alpha applied before this BN?
    # (final pw, skip layer or None, block-input layer). When the skip is an
    # Identity the residual adds the block INPUT itself, so the add needs the
    # tensor fed to the first depthwise, not any accumulator.
    add_of: Optional[tuple] = None


def layer_sites(model: BinaryMatchboxNet) -> List[Site]:
    """Mirrors Emitter._plain/_tcs, including the names."""
    sites: List[Site] = []
    for st in model.cfg.stages:
        mod = model.stages[st.name]
        if isinstance(mod, BinaryTCSBlock):
            skip = (f"{st.name}_skip" if isinstance(mod.skip, BinaryConv1d)
                    else None)
            if skip:
                sites.append(Site(skip, mod.skip))          # unscaled, no BN
            last = len(mod.subs) - 1
            for i, sub in enumerate(mod.subs):
                sites.append(Site(f"{st.name}_s{i}_dw", sub.dw, sub.bn))
                # the final pw feeds the add, so its BN belongs to the add
                sites.append(Site(f"{st.name}_s{i}_pw", sub.pw,
                                  None if i == last else mod.post_bns[i]))
            sites.append(Site(f"{st.name}_add", None, mod.post_bns[last],
                              add_of=(f"{st.name}_s{last}_pw", skip,
                                      f"{st.name}_s0_dw")))
        elif isinstance(mod, PlainStage):
            fuses = mod.activation == "sign"
            if mod.dw is not None:
                sites.append(Site(f"{st.name}_dw", mod.dw, mod.dw_bn))
                sites.append(Site(f"{st.name}_pw", mod.pw,
                                  mod.bn if fuses else None))
            else:
                sites.append(Site(st.name, mod.conv, mod.bn if fuses else None))
    return sites


def _alpha(conv: nn.Module) -> Optional[torch.Tensor]:
    if isinstance(conv, QuantConv1d):
        return conv.weight_scale().reshape(-1)      # real acc = scale * int acc
    return conv_alpha(conv)


def _int_words(t: torch.Tensor) -> List[str]:
    return [f"{int(v) & 0xFFFFFFFF:08x}"
            for v in t.detach().reshape(-1).round().to(torch.int64).tolist()]


def _pack_words(o: torch.Tensor) -> List[str]:
    """[N, C, T] in +-1 -> one word per (clip, frame), channels LSB first."""
    return to_hex_words(pack_pm1(o.transpose(1, 2).reshape(-1, o.shape[1])))


@torch.no_grad()
def dump_golden(model: BinaryMatchboxNet, x: torch.Tensor, out: Path,
                tag: str) -> Dict[str, Any]:
    """Run `x` [N, C, T] in {-1,+1} and write every layer's reference values."""
    out.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    sites = layer_sites(model)
    by_conv = {id(s.conv): s.name for s in sites if s.conv is not None}

    acc: Dict[str, torch.Tensor] = {}

    def pre_hook(mod, inp):
        name = by_conv.get(id(mod))
        if name is None:
            return
        if isinstance(mod, BinaryConv1d):
            acc[name] = binary_accumulator(mod, inp[0]).detach()
            acc[name + "@in"] = inp[0].detach()      # identity skips need this
        elif isinstance(mod, QuantConv1d):
            # an integer accumulator exists only where the input really is +-1
            u = torch.unique(inp[0])
            if bool(((u == 1) | (u == -1)).all()):
                s = mod.weight_scale()
                q = torch.round(mod.weight.detach() / s).clamp(-mod.qmax,
                                                              mod.qmax)
                acc[name] = torch.nn.functional.conv1d(
                    inp[0], q, None, mod.stride, mod.padding, mod.dilation,
                    mod.groups).detach()
        else:                                    # float head: keep its input
            acc[name + "@in"] = inp[0].detach()

    real_out: Dict[str, torch.Tensor] = {}

    def post_hook(mod, inp, o):
        name = by_conv.get(id(mod))
        if name is not None:
            real_out[name] = o.detach()

    hooks = []
    for m in model.modules():
        if id(m) in by_conv:
            hooks.append(m.register_forward_pre_hook(pre_hook))
            hooks.append(m.register_forward_hook(post_hook))
    try:
        logits = model(x)
    finally:
        for h in hooks:
            h.remove()

    # the residual add is not a module -- assemble it the way the block does
    for s in sites:
        if s.add_of:
            pw, skip, block_in = s.add_of
            a = acc[pw].clone()
            # a projected skip contributes its own integer accumulator; an
            # identity skip contributes the block input, which is +-1
            a = a + (acc[skip] if skip else acc[block_in + "@in"])
            acc[s.name] = a

    files: Dict[str, Dict[str, Any]] = {}
    p = pack_pm1(x.transpose(1, 2).reshape(-1, x.shape[1]))
    (out / "input.hex").write_text("\n".join(to_hex_words(p)) + "\n")
    files["input"] = {"file": "input.hex", "shape": list(x.shape),
                      "layout": "one word per (clip, frame); bit c = channel c, "
                                "LSB first, -1 -> 0"}

    for s in sites:
        a = acc.get(s.name)
        if a is not None and a.dtype.is_floating_point:
            f = f"{s.name}_acc.hex"
            (out / f).write_text("\n".join(_int_words(a)) + "\n")
            files[f"{s.name}_acc"] = {
                "file": f, "shape": list(a.shape), "dtype": "int32",
                "order": "clip-major, then out_ch, then frame",
                "note": "pre-alpha integer accumulator"}
        if s.bn is not None and a is not None:
            # exactly what RTL does: one integer compare per output channel
            fused = fuse_bn_to_threshold(
                s.bn, _alpha(s.conv) if (s.conv is not None and s.scaled)
                else None)
            o = fused.apply(a)
            f = f"{s.name}_out.hex"
            (out / f).write_text("\n".join(_pack_words(o)) + "\n")
            files[f"{s.name}_out"] = {
                "file": f, "shape": list(o.shape),
                "layout": "one word per (clip, frame)",
                "note": "FusedThreshold.apply(acc) -- the compare RTL performs"}
        elif s.name in real_out and s.bn is None and a is None:
            o = real_out[s.name]
            f = f"{s.name}_out.txt"
            (out / f).write_text("\n".join(f"{v:.8e}" for v in
                                           o.reshape(-1).tolist()) + "\n")
            files[f"{s.name}_out"] = {
                "file": f, "shape": list(o.shape), "dtype": "float32",
                "note": "REFERENCE ONLY -- real-valued; the tail's fixed-point "
                        "format is not chosen yet"}

    (out / "logits.txt").write_text(
        "\n".join(" ".join(f"{v:.8e}" for v in row) for row in logits.tolist())
        + "\n")
    (out / "predictions.txt").write_text(
        "\n".join(str(int(i)) for i in logits.argmax(1).tolist()) + "\n")

    man = {"tag": tag, "n_clips": int(x.shape[0]),
           "n_channels": int(x.shape[1]), "T": int(x.shape[2]),
           "files": files,
           "note": "compare layer by layer; the first mismatch names the module"}
    (out / "golden.json").write_text(json.dumps(man, indent=2))
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--out", default=None, help="default runs/<tag>/rtl/golden")
    args = ap.parse_args()

    from train.config import load_config
    from data.afe import AFEFrontend, load_afe_state
    from data.speech_commands import build_dataloaders

    run = Path(args.runs) / args.tag
    cfg = load_config(str(run / "config.yaml"))
    afe = AFEFrontend(cfg.afe).eval()
    model = BinaryMatchboxNet(cfg.model).eval()
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    load_afe_state(afe, ck["afe"])

    te = build_dataloaders(cfg.data, cfg.train.batch_size, cfg.afe.sample_rate,
                           seed=cfg.train.seed)[2]
    wav, labels = next(iter(te))
    wav, labels = wav[:args.clips], labels[:args.clips]
    with torch.no_grad():
        x = afe(wav, target_T=cfg.model.T)

    out = Path(args.out) if args.out else run / "rtl" / "golden"
    man = dump_golden(model, x, out, args.tag)
    (out / "labels.txt").write_text(
        "\n".join(str(int(v)) for v in labels.tolist()) + "\n")

    pred = [int(v) for v in (out / "predictions.txt").read_text().split()]
    ok = sum(int(p == int(l)) for p, l in zip(pred, labels.tolist()))
    print(f"{man['n_clips']} clips -> {len(man['files'])} files in {out}")
    print(f"reference predictions match labels on {ok}/{len(pred)} "
          f"(a sanity check on the dump, not an accuracy measurement)")


if __name__ == "__main__":
    main()
