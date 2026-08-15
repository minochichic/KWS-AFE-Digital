"""Measure the integer accumulator range at every site the RTL must size.

Why this runs before any Verilog gets written.

A binary conv accumulator is `n = 2*popcount(XNOR) - N`, so it is bounded by
+-N where N is the number of terms (K*C_in/groups). Sizing the datapath from
that bound is correct but wasteful: the bound is only reached when every single
input agrees with every single weight, which trained networks never do. The
residual add is worse -- `acc + res` bounds add, so the worst case doubles,
while in practice the two are far from co-extreme.

So we measure. Feed the real test set through the trained model, record the
true min/max at each site, and size from that plus a guard bit.

WHAT COUNTS AS A SITE. Every place RTL holds an integer:

  binary    a BinaryConv1d-family module. The accumulator is the pre-alpha
            integer, which is what `fuse.py` defines its thresholds against --
            NOT the module's own output, which has alpha folded in when
            `scale=True`. `binary_accumulator()` recovers it.
  residual  the input to a block's LAST post_bn, which is `pw_acc + skip_acc`.
            Both addends are unscaled by construction (`final=True` and
            `skip(scale=False)`), so their sum is exactly integer, and hooking
            that BN's input measures the joint range rather than the looser
            sum-of-two-ranges bound.
  real      int8/fixed-point stages (conv1, conv3, conv4). Reported as floats;
            their widths come from the quantization scheme, not from here.

SAFETY. The measured range is only as good as the data it saw. `guard_bits`
defaults to 1 so a slightly out-of-distribution input cannot wrap the
accumulator; `saturate` in RTL is the belt to this suspenders.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from export.fuse import binary_accumulator
from models.binary_ops import BinaryConv1d
from models.binary_matchboxnet import BinaryTCSBlock


def signed_bits(lo: int, hi: int) -> int:
    """Two's-complement width holding every value in [lo, hi].

    Needs 2^(w-1) >= hi+1 and 2^(w-1) >= -lo. The max(..., 1) keeps log2 defined
    when the range sits entirely on one side of zero.
    """
    need = max(int(hi) + 1, -int(lo), 1)
    return 1 + max(1, math.ceil(math.log2(need)))


@dataclass
class Site:
    """One integer holding place in the datapath."""

    name: str
    kind: str                    # binary | residual | real
    lo: float = float("inf")
    hi: float = float("-inf")
    n_terms: int = 0             # worst-case |acc| bound; 0 when not applicable
    n_seen: int = 0

    def update(self, x: torch.Tensor) -> None:
        self.lo = min(self.lo, float(x.min()))
        self.hi = max(self.hi, float(x.max()))
        self.n_seen += x.numel()

    # -- derived ----------------------------------------------------------
    @property
    def measured_bits(self) -> Optional[int]:
        if self.kind == "real" or self.n_seen == 0:
            return None
        return signed_bits(math.floor(self.lo), math.ceil(self.hi))

    @property
    def worst_bits(self) -> Optional[int]:
        if self.n_terms == 0:
            return None
        return signed_bits(-self.n_terms, self.n_terms)

    @property
    def saved(self) -> Optional[int]:
        w, m = self.worst_bits, self.measured_bits
        return None if (w is None or m is None) else w - m


def _n_terms(conv: nn.Module) -> int:
    """Worst-case |accumulator| for a binary conv: one term per MAC."""
    w = conv.weight
    return int(w.shape[1] * w.shape[2])       # (in_ch/groups) * kernel


def _collect_sites(model: nn.Module) -> Dict[str, Site]:
    sites: Dict[str, Site] = {}
    residual_bns = set()

    for name, mod in model.named_modules():
        if isinstance(mod, BinaryTCSBlock) and len(mod.post_bns) > 0:
            # the last post_bn's input is `pw_acc + skip_acc`
            residual_bns.add(f"{name}.post_bns.{len(mod.post_bns) - 1}")

    for name, mod in model.named_modules():
        if isinstance(mod, BinaryConv1d):
            sites[name] = Site(name, "binary", n_terms=_n_terms(mod))
        elif name in residual_bns:
            sites[name] = Site(name, "residual")
        elif isinstance(mod, nn.Conv1d):          # int8 / fixed-point stages
            sites[name] = Site(name, "real")
    return sites


@torch.no_grad()
def measure_ranges(model: nn.Module,
                   loader: Iterable,
                   device: torch.device | str = "cpu",
                   max_batches: Optional[int] = None) -> List[Site]:
    """Run data through `model` and record the integer range at every site.

    `model` is put in eval mode -- dropout and BN must be frozen or the
    accumulators are not the ones the exported network will produce.
    """
    model = model.to(device).eval()
    sites = _collect_sites(model)
    handles = []

    def bin_hook(name):
        def fn(mod, inp):
            sites[name].update(binary_accumulator(mod, inp[0]))
        return fn

    def pre_hook(name):
        def fn(mod, inp):
            sites[name].update(inp[0])
        return fn

    def out_hook(name):
        def fn(mod, inp, out):
            sites[name].update(out)
        return fn

    for name, mod in model.named_modules():
        s = sites.get(name)
        if s is None:
            continue
        if s.kind == "binary":
            handles.append(mod.register_forward_pre_hook(bin_hook(name)))
        elif s.kind == "residual":
            handles.append(mod.register_forward_pre_hook(pre_hook(name)))
        else:
            handles.append(mod.register_forward_hook(out_hook(name)))

    try:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            model(x.to(device))
    finally:
        for h in handles:
            h.remove()

    return [s for s in sites.values() if s.n_seen > 0]


def print_range_report(sites: List[Site], guard_bits: int = 1) -> None:
    print(f"{'site':<34} {'kind':<9} {'measured range':>20} "
          f"{'meas':>5} {'worst':>6} {'save':>5}")
    print("-" * 84)
    # Only sites that have BOTH numbers may be summed. `residual` has no
    # analytic bound here (n_terms=0), so counting it on the measured side and
    # not on the worst side would compare 18 sites against 15 and invent a
    # saving that is not there.
    tot_m = tot_w = 0
    for s in sites:
        rng = (f"[{s.lo:.3g}, {s.hi:.3g}]" if s.kind == "real"
               else f"[{int(math.floor(s.lo))}, {int(math.ceil(s.hi))}]")
        m = s.measured_bits
        w = s.worst_bits
        if m is not None and w is not None:
            tot_m += m + guard_bits
            tot_w += w
        print(f"{s.name:<34} {s.kind:<9} {rng:>20} "
              f"{'' if m is None else m + guard_bits:>5} "
              f"{'' if w is None else w:>6} "
              f"{'' if s.saved is None else s.saved:>5}")
    print("-" * 84)
    both = [s for s in sites
            if s.measured_bits is not None and s.worst_bits is not None]
    print(f"{'sum over the ' + str(len(both)) + ' comparable sites':<44} "
          f"{tot_m:>5} {tot_w:>6} {tot_w - tot_m:>5}")

    # What a FOLDED datapath actually pays for: one accumulator register, sized
    # by the widest site. The sum is a fully-unrolled quantity -- there, every
    # layer owns its own hardware. Reporting only the sum invites sizing a
    # design we are not building (CLAUDE.md 0: folded).
    max_m = max((s.measured_bits + guard_bits for s in sites
                 if s.measured_bits is not None), default=0)
    max_w = max((s.worst_bits for s in sites if s.worst_bits is not None),
                default=0)
    print(f"{'FOLDED: widest single accumulator':<44} {max_m:>5} {max_w:>6} "
          f"{max_w - max_m:>5}")

    print("\nmeas = measured + guard bits; worst = +-(K*C_in) bound.")
    print("`save` is per-site bits removed by measuring instead of bounding.")
    if max_w <= max_m:
        print(f"\n-> measuring did NOT shrink the folded accumulator "
              f"({max_m} vs {max_w} bits). Size from the bound: it is exact, "
              f"needs no guard bit, and cannot overflow.")
    print("`real` sites are fake-quantized FLOAT outputs, not integer "
          "accumulators -- use them for fixed-point placement, not for width.")


def to_json(sites: List[Site], path: str, guard_bits: int = 1) -> None:
    """Emit the widths the manifest (and parameters.vh) will be built from."""
    out = {
        "guard_bits": guard_bits,
        "sites": [
            {**asdict(s),
             "measured_bits": s.measured_bits,
             "worst_bits": s.worst_bits,
             "rtl_bits": (None if s.measured_bits is None
                          else s.measured_bits + guard_bits)}
            for s in sites
        ],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}  ({len(sites)} sites)")
