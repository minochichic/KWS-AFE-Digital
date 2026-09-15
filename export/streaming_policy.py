"""Convert validation-selected streaming thresholds to RTL integer units."""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
from typing import Any, Dict


def add_streaming_policy(man: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    """Quantize a validation-selected streaming policy onto the pool grid.

    The model reports average logits in units of 2^-frac. RTL deliberately
    keeps the undivided sum over ``pool_frames``, so a float margin ``m`` maps
    to ``ceil(m * 2^frac * pool_frames)``. Ceil preserves the meaning of a
    lower bound: quantization must not make the hardware gate looser.
    """
    required = ("checkpoint_tag", "hop_frames", "required_consecutive",
                "cooldown_windows", "margin_float")
    missing = [key for key in required if key not in policy]
    if missing:
        raise ValueError(f"streaming policy missing: {', '.join(missing)}")
    if str(policy["checkpoint_tag"]) != str(man["tag"]):
        raise ValueError(
            f"streaming policy is for {policy['checkpoint_tag']!r}, "
            f"but export tag is {man['tag']!r}")

    tail = man.get("tail") or {}
    frac_bits = int(tail.get("frac_bits", -1))
    pool_frames = int(tail.get("pool_frames", 0))
    if frac_bits < 0 or pool_frames <= 0:
        raise ValueError("manifest has no valid tail frac_bits/pool_frames")

    hop = int(policy["hop_frames"])
    consecutive = int(policy["required_consecutive"])
    cooldown = int(policy["cooldown_windows"])
    margin = Decimal(str(policy["margin_float"]))
    if hop <= 0 or consecutive <= 0 or cooldown < 0 or margin < 0:
        raise ValueError("streaming hop/N must be positive; cooldown/margin nonnegative")

    scaled = margin * Decimal(1 << frac_bits) * Decimal(pool_frames)
    margin_int = int(scaled.to_integral_value(rounding=ROUND_CEILING))
    pool_site = next((site for site in tail.get("sites", [])
                      if "pool_acc_bits" in site), None)
    if pool_site is None:
        raise ValueError("manifest has no pooled-logit accumulator width")
    pool_acc_bits = int(pool_site["pool_acc_bits"])
    margin_bits = pool_acc_bits + 1
    if margin_int >= (1 << pool_acc_bits):
        raise ValueError(
            f"streaming margin {margin_int} does not fit signed {margin_bits} bits")
    stream = {
        "checkpoint_tag": str(policy["checkpoint_tag"]),
        "hop_frames": hop,
        "required_consecutive": consecutive,
        "cooldown_windows": cooldown,
        "margin_float": float(margin),
        "margin_int": margin_int,
        "pool_frames": pool_frames,
        "frac_bits": frac_bits,
        "margin_bits": margin_bits,
        "margin_rule": "ceil(margin_float * pool_frames * 2^frac_bits)",
    }
    for key in ("selection_split", "status", "source"):
        if key in policy:
            stream[key] = policy[key]
    man["streaming"] = stream
    return stream
