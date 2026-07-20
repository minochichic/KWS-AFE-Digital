"""Print a resolved config as a stage table.

Sanity-check what a YAML + overrides actually builds before burning Colab GPU
hours on it.

    python experiments/inspect_config.py configs/base.yaml
    python experiments/inspect_config.py configs/base.yaml model.C=16 model.T=96
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.config import load_config  # noqa: E402


def _parse_overrides(args) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for a in args:
        if "=" not in a:
            raise SystemExit(f"bad override {a!r}, expected key.path=value")
        k, v = a.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = {"true": True, "false": False}.get(v.lower(), v)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cfg = load_config(sys.argv[1], _parse_overrides(sys.argv[2:]))
    m, a = cfg.model, cfg.afe

    print(f"tag={cfg.tag}  C={m.C}  T={m.T}  classes={m.n_classes}")
    print(
        f"AFE: {a.n_channels} ch, {a.f_min:.0f}-{a.f_max:.0f} Hz, "
        f"STFT {a.stft_win_ms:.0f}/{a.stft_hop_ms:.0f} ms, "
        f"envelope {a.envelope_reduce} over {a.envelope_win_ms:.0f} ms "
        f"-> native T={a.native_T}"
    )
    print(f"input: [batch, {m.in_channels}, {m.T}]\n")

    hdr = f"{'stage':<8}{'R':>3}{'k':>5}{'str':>5}{'dil':>5}{'out_ch':>8}  {'precision':<10}{'kind'}"
    print(hdr)
    print("-" * len(hdr))
    for s in m.stages:
        kind = "TCS sep" if s.separable else "conv"
        if s.residual:
            kind += " +res"
        print(
            f"{s.name:<8}{s.n_sub_blocks:>3}{s.kernel:>5}{s.stride:>5}"
            f"{s.dilation:>5}{s.out_channels(m.C, m.n_classes):>8}  "
            f"{s.precision:<10}{kind}"
        )

    binary = [s.name for s in m.stages if s.precision == "binary"]
    print(f"\nbinary stages: {', '.join(binary)}")
    print(f"non-binary   : first={m.stages[0].name} ({m.stages[0].precision}), "
          f"last={m.stages[-1].name} ({m.stages[-1].precision})")

    for w in cfg.warnings():
        print(f"\n[warn] {w}")


if __name__ == "__main__":
    main()
