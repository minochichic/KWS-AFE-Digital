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

THE TAIL IS DUMPED AS INTEGERS TOO. conv2_pw, conv3 and conv4 keep their BN --
nothing downstream is a sign() to fold it into -- so they are not compares but
fixed-point arithmetic. export/tailbuild.py turns each into an integer gain,
offset and shift, and the same function that builds the ROM is used here to
produce the expected values. Deriving them twice is how a ROM and its own
expected output come to disagree, and every symptom of that points at the RTL.

golden.json's `tail` section reports, per site, how far the integer path moved
from the float one in LSBs of that site's own grid, how wide the accumulator
actually got, and how many values hit the clamp. Those three numbers are what
say whether the chosen format is right, and none of them is visible from an
accuracy figure.

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
from export.tailbuild import (TailSite, apply_site, int8_weights, tail_plan)
from export.tailfmt import pooled_argmax
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


def _conv_int(x: torch.Tensor, w: torch.Tensor, conv: nn.Module) -> torch.Tensor:
    """Integer 1x1 conv, done in float64 because F.conv1d has no int64 kernel.

    Exact, not approximately exact: conv3's accumulator reaches 1.3e8 and
    conv4's 1.7e7, both far inside float64's 53 bits of integer range. Written
    down because "it uses floats" is otherwise a reasonable thing to distrust.
    """
    y = torch.nn.functional.conv1d(x.double(), w.double(), None, conv.stride,
                                   conv.padding, conv.dilation, conv.groups)
    return torch.round(y).to(torch.int64)


@torch.no_grad()
def _dump_tail(model: BinaryMatchboxNet, acc: Dict[str, torch.Tensor],
               out: Path) -> tuple:
    """Chain the tail in integers and write what every register holds.

    The chaining is the point. conv3's input is not the float model's conv2_pw
    output -- it is the QUANTIZED one, because that is what the previous layer's
    register contains. Feeding conv3 the float activations would produce vectors
    the RTL can never match, and they would look right to within a percent,
    which is the worst possible amount of wrong.

    conv2_pw's own accumulator needs no such treatment: its input comes through
    conv2_dw's threshold, which is exact, so the value the pre-hook recorded is
    already the integer the hardware sums.
    """
    sites: List[TailSite] = tail_plan(model)
    files: Dict[str, Dict[str, Any]] = {}
    report: List[Dict[str, Any]] = []

    cur: Optional[torch.Tensor] = None          # previous site's integer output
    for s in sites:
        if s.in_fmt is None:                    # conv2_pw: binary accumulator
            a = acc[s.name].round().to(torch.int64)
        else:
            if cur is None:
                raise ValueError(f"{s.name}: no quantized input to chain from")
            w = (int8_weights(s.conv)[0] if isinstance(s.conv, QuantConv1d)
                 else s.weights)
            a = _conv_int(cur, w, s.conv)
        y = apply_site(s, a)

        fa, fy = f"{s.name}_acc.hex", f"{s.name}_out.hex"
        (out / fa).write_text("\n".join(_int_words(a)) + "\n")
        (out / fy).write_text("\n".join(_int_words(y)) + "\n")
        files[f"{s.name}_acc"] = {
            "file": fa, "shape": list(a.shape), "dtype": "int32",
            "order": "clip-major, then out_ch, then frame",
            "note": f"integer accumulator, {s.acc_bits} bits signed"}
        files[f"{s.name}_out"] = {
            "file": fy, "shape": list(y.shape), "dtype": "int32",
            # multi-bit, so there is nothing to pack. A testbench that assumed
            # `_out.hex` meant packed +-1 would misread these as 32 channels of
            # nonsense, so the kind of file is declared rather than inferred
            # from the name.
            "packed": False,
            "order": "clip-major, then out_ch, then frame",
            "note": f"(A*acc + B + half) >> {s.fold.shift}"
                    f"{', relu' if s.fold.relu else ''}, saturated to "
                    f"{s.out_fmt} -- one word per value"}

        # ---- two different questions, and the first version asked the wrong
        # one. Measuring the integer output against the CONTINUOUS float value
        # necessarily includes rounding onto the 1/2^frac grid, which is up to
        # half an LSB by definition -- so it reported ~0.5 for every site and
        # made a fold that is correct look 2000x worse than emit's estimate.
        #
        # What isolates the fold is comparing against float-then-rounded: do
        # the integer arithmetic and the float arithmetic land on the SAME grid
        # point? That is the question the RTL cares about.
        gain = torch.tensor(s.fold.gain_real, dtype=torch.float64).view(1, -1, 1)
        bias = torch.tensor(s.fold.bias_real, dtype=torch.float64).view(1, -1, 1)
        ref = gain * a.double() + bias
        if s.fold.relu:
            ref = torch.clamp_min(ref, 0.0)
        scale = float(1 << s.out_fmt.frac_bits)
        ref_q = torch.clamp(torch.round(ref * scale),
                            s.out_fmt.lo, s.out_fmt.hi)
        fold_dev = float((y.double() - ref_q).abs().max())
        grid_dev = float((y.double() / scale - ref).abs().max()) * scale
        clamped = int(((y == s.out_fmt.hi) | (y == s.out_fmt.lo)).sum())
        report.append({"name": s.name, "out_format": str(s.out_fmt),
                       "shift": s.fold.shift, "acc_bits": s.acc_bits,
                       "acc_absmax": int(a.abs().max()),
                       # 0 means the integer path and the float path chose the
                       # same grid point everywhere. 1 is allowed and expected
                       # only on exact ties, where torch.round goes to even and
                       # `(x + half) >> s` goes up.
                       "fold_dev_lsb": round(fold_dev, 4),
                       # <= 0.5 by construction: this is the grid itself, not
                       # an error the fold introduced.
                       "grid_dev_lsb": round(grid_dev, 4),
                       "clamped_values": clamped,
                       "n_values": int(y.numel())})
        cur = y

    # the head: sum over time, argmax, no divide
    if cur is not None:
        pooled = cur.sum(dim=2)                             # [N, classes]
        (out / "pooled.txt").write_text(
            "\n".join(" ".join(str(int(v)) for v in row)
                      for row in pooled.tolist()) + "\n")
        pred = [pooled_argmax([[int(v) for v in cur[n, :, t].tolist()]
                               for t in range(cur.shape[2])])
                for n in range(cur.shape[0])]
        (out / "predictions_fixed.txt").write_text(
            "\n".join(str(p) for p in pred) + "\n")
        files["pooled"] = {
            "file": "pooled.txt", "shape": list(pooled.shape), "dtype": "int",
            "note": "sum of the per-frame logits over T. The pool's divide by T "
                    "is not built: same positive factor on every class, and "
                    "argmax ignores it"}
        files["predictions_fixed"] = {
            "file": "predictions_fixed.txt", "shape": [int(cur.shape[0])],
            "note": "argmax of the pooled integer logits -- the answer the "
                    "hardware gives. Compare against predictions.txt, which is "
                    "the float model's"}
        # torch.argmax and pooled_argmax must agree on the same numbers, or one
        # of the two tie rules is not what was assumed
        if pred != pooled.argmax(1).tolist():
            raise ValueError("pooled_argmax disagrees with torch.argmax on the "
                             "integer logits -- check the tie rule")
    return files, {"sites": report}


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

    tail_names = {t.name for t in tail_plan(model)}
    for s in sites:
        if s.name in tail_names:
            # written by _dump_tail, which knows the width and the format.
            # conv2_pw would otherwise be dumped twice with the same contents
            # and two different notes.
            continue
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
                "file": f, "shape": list(o.shape), "packed": True,
                "layout": "one word per (clip, frame)",
                "note": "FusedThreshold.apply(acc) -- the compare RTL performs"}
    # ---- the tail, as the integers RTL holds ------------------------------ #
    tail_files, tail_report = _dump_tail(model, acc, out)
    files.update(tail_files)

    (out / "logits.txt").write_text(
        "\n".join(" ".join(f"{v:.8e}" for v in row) for row in logits.tolist())
        + "\n")
    (out / "predictions.txt").write_text(
        "\n".join(str(int(i)) for i in logits.argmax(1).tolist()) + "\n")

    man = {"tag": tag, "n_clips": int(x.shape[0]),
           "n_channels": int(x.shape[1]), "T": int(x.shape[2]),
           "files": files, "tail": tail_report,
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

    t = man.get("tail")
    if t:
        print("\ntail, integer path:")
        print(f"{'site':<10}{'out':>6}{'shift':>7}{'acc':>5}{'|acc| max':>11}"
              f"{'bound':>12}{'used':>6}{'fold':>7}{'grid':>7}{'clamped':>14}")
        for st in t["sites"]:
            bound = (1 << (st["acc_bits"] - 1)) - 1
            used = max(1, int(st["acc_absmax"])).bit_length() + 1
            clamp = f"{st['clamped_values']}/{st['n_values']}"
            print(f"{st['name']:<10}{st['out_format']:>6}{st['shift']:>7}"
                  f"{st['acc_bits']:>5}{st['acc_absmax']:>11}{bound:>12}"
                  f"{used:>6}{st['fold_dev_lsb']:>7}{st['grid_dev_lsb']:>7}"
                  f"{clamp:>14}")
        fx = [int(v) for v in (out / "predictions_fixed.txt").read_text().split()]
        agree = sum(int(a == b) for a, b in zip(fx, pred))
        print(f"\nfixed-point argmax agrees with the float model on "
              f"{agree}/{len(fx)} clips")
        print("fold = does the integer path land on the same grid point as "
              "float-then-round? 0 is what it must be; 1 is allowed only on an "
              "exact tie. This is the fold's own error.")
        print("grid = distance to the CONTINUOUS float value. <= 0.5 by "
              "construction -- it is the 1/64 grid itself, not something the "
              "fold introduced. Reporting this one alone was the earlier "
              "mistake: it reads ~0.5 for a perfectly correct fold.")
        print("acc vs used = allocated width against what these clips actually "
              "reached. Large slack is expected (the bound assumes every weight "
              "and activation is extreme and agrees in sign) and is NOT a "
              "reason to narrow on two clips -- run export.ranges for that.")
        if agree != len(fx):
            print("NOTE: a disagreement is not automatically a bug -- a clip "
                  "whose top two logits are within one LSB can legitimately "
                  "flip. Check the margin before touching the format.")


if __name__ == "__main__":
    main()
