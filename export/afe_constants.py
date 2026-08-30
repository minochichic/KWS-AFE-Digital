"""Dump the AFE-side constants a trained run pinned down, as plain JSON.

These are the numbers that live LEFT of the comparator, so they are what the
analog side needs and what any circuit-level simulation has to be told. They
never reach the FPGA (docs/ICD.md) -- do not confuse this with the RTL
manifest, which carries shapes, bit widths and weights.

Why JSON and not "import torch and read the checkpoint": the SPICE scripts in
analog/AFE/ are numpy + ngspice and deliberately have no torch dependency, and
they run on a machine that may not have the checkpoints at all.

The one subtlety worth stating. `lse_temp` is stored as BOTH the absolute value
and the fraction it came from:

    lse_temp = lse_temp_frac * median(per-frame cross-channel max)

The absolute value is in the software envelope domain (sqrt-mel), which is not
volts. The FRACTION is dimensionless and is re-derivable from any envelope set,
including SPICE volts -- which is exactly why `lse_temp_frac` was defined as a
ratio in the first place (data/afe.py `_xlse`: it "survives not knowing the mic
sensitivity"). So a circuit simulation should use `lse_temp_frac` and recompute
the median from its own envelopes, NOT copy `lse_temp` across domains.

`fixed_lo`/`fixed_hi` have no such escape: they are absolute levels in the
software domain, so mapping them into volts needs a stated correspondence
between the two dynamic ranges (the assumption analog/AFE/scripts/
learned_r7r8.py already documents).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


def afe_constants(afe: torch.nn.Module, cfg: Any) -> Dict[str, Any]:
    """Everything left of the comparator, as JSON-safe values."""
    a = cfg.afe
    out: Dict[str, Any] = {
        "normalize": a.normalize,
        "compression": getattr(a, "compression", "log"),
        "n_channels": int(a.n_channels),
        "comparators_per_channel": int(getattr(a, "comparators_per_channel", 1)),
        "envelope_win_ms": float(a.envelope_win_ms),
        "envelope_reduce": a.envelope_reduce,
        "envelope_tau_ms": float(getattr(a, "envelope_tau_ms", 0.0)),
        "filterbank_source": getattr(a, "filterbank_source", "mel"),
        # per-channel comparator thresholds. For divider forms these ARE the
        # divider ratios Rb/(Ra+Rb) and are clamped to [0,1] in the forward pass,
        # so clamp here too -- an unclamped latent value is not what the circuit
        # would be built to.
        "threshold": [float(v) for v in afe.threshold.detach().flatten()],
    }
    # threshold_min 도 forward 에서 clamp 된다. straight-through 라 잠재값은
    # 자유롭게 흘러가고 실제로 -6.269 까지 간 런이 있다 -- 그걸 그대로 내보내면
    # 회로가 만들어진 값과 다른 숫자가 저항이 된다.
    tmin = float(getattr(a, "threshold_min", 0.0) or 0.0)
    if tmin > 0.0:
        out["threshold_min"] = tmin
        out["threshold"] = [max(tmin, v) for v in out["threshold"]]
    if a.normalize in ("xmix", "xlse"):
        out["threshold_is_divider_ratio"] = True
        out["threshold"] = [min(1.0, max(0.0, v)) for v in out["threshold"]]
    for name in ("fixed_lo", "fixed_hi", "xmax_floor", "lse_temp"):
        buf = getattr(afe, name, None)
        if buf is not None:
            out[name] = float(buf)
    if a.normalize == "xlse":
        # the dimensionless form -- the one that crosses domains safely
        out["lse_temp_frac"] = float(a.lse_temp_frac)
    out["xmax_floor_frac"] = float(getattr(a, "xmax_floor_frac", 0.0))
    return out


def dump_afe(afe: torch.nn.Module, cfg: Any, path: str | Path) -> Dict[str, Any]:
    d = afe_constants(afe, cfg)
    Path(path).write_text(json.dumps(d, indent=2))
    print(f"wrote {path}  (normalize={d['normalize']}, "
          f"{len(d['threshold'])} thresholds)")
    return d
