"""What is in runs/, and which of them can be exported to RTL.

Written because the obvious question -- "which track is which tag?" -- had no
one-line answer, and guessing a tag costs a FileNotFoundError several commands
into a pipeline.

Reports the two fields that actually separate the tracks: `norm_mode` (whether
the channels share a diode-OR node) and `threshold_mode`. The accuracy comes
from the checkpoint rather than history.json, because that is the number
export.emit will be exporting.

Run:  python -m experiments.list_runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cfg_bits(path: Path) -> dict:
    """The few AFE fields that tell the tracks apart, without importing torch."""
    out, want = {}, ("norm_mode", "threshold_mode", "compression",
                     "lse_temp_frac", "fixed_scale_quantile", "aug_gain_db")
    try:
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            k, v = (p.strip() for p in line.split(":", 1))
            if k in want and v:
                out[k] = v
    except OSError:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    root = Path(args.runs)
    if not root.is_dir():
        print(f"no {root}/ here")
        return

    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        cfg, ck = d / "config.yaml", d / "best.pt"
        acc = ""
        if ck.is_file():
            try:
                import torch
                acc = f"{torch.load(ck, map_location='cpu', weights_only=True).get('best_acc', -1):.4f}"
            except Exception as e:                      # noqa: BLE001
                acc = f"?({type(e).__name__})"
        rows.append((d.name, cfg.is_file(), ck.is_file(), acc, _cfg_bits(cfg)))

    if not rows:
        print(f"{root}/ is empty")
        return

    w = max(len(r[0]) for r in rows)
    print(f"{'tag':<{w}}  {'export?':<8}{'best_acc':<10}config")
    for name, has_cfg, has_ck, acc, bits in rows:
        ok = "yes" if (has_cfg and has_ck) else (
            "no cfg" if not has_cfg else "no ckpt")
        detail = "  ".join(f"{k}={v}" for k, v in bits.items())
        print(f"{name:<{w}}  {ok:<8}{acc:<10}{detail}")
    print()
    print("export? = has both config.yaml and best.pt, which is what "
          "export.emit needs")


if __name__ == "__main__":
    main()
