"""Score board captures (afe_web's out/afe/live/*.jsonl) with any trained run, offline.

The FPGA computes exactly what best.pt computes on the same image (2026-10-30:
234/236, 161/163), so a run's answer on a capture's bits IS what the board would
answer with that run's weights loaded -- no export, no bitstream. That lets one
bench session at a run's voltages (runs/<tag>/vthr_levels.csv) be scored with
that run, and lets later runs be scored on the same captures.

    python -m experiments.score_live out/afe/live/*.jsonl --run runs/bd_grid_sim runs/bd_base

Only captures with a settled ARB (ref + label) count. They are grouped by the
V_thr typed on the page; --levels-of keeps only captures taken at that run's
vthr_levels.csv.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from experiments.afe_clip import bits_of, classify, load


def load_scored(paths: List[Path]) -> List[dict]:
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("ref") and r.get("label") and r.get("clip"):
                    out.append(r)
    return out


def run_levels(run: Path) -> Tuple[float, ...]:
    rows = (run / "vthr_levels.csv").read_text().split()[1:]
    return tuple(float(r.split(",")[1]) for r in rows)


def score(recs: List[dict], model, names: List[str]) -> np.ndarray:
    """1/0 per record: the run's answer on the board's bits == the ARB's word."""
    if not recs:
        return np.zeros(0)
    pred, _ = classify(model, np.stack([bits_of(r["clip"]) for r in recs]))
    return np.array([names[int(p)] == r["label"] for p, r in zip(pred, recs)], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--run", nargs="+", required=True, type=Path)
    ap.add_argument("--levels-of", type=Path, default=None,
                    help="run dir: keep only captures taken at its vthr_levels.csv")
    args = ap.parse_args()

    from data.speech_commands import KEYWORDS
    names = KEYWORDS + ["silence", "unknown"]
    recs = load_scored(args.logs)
    if args.levels_of:
        want = run_levels(args.levels_of)
        recs = [r for r in recs if tuple(float(v) for v in (r.get("vthr_mv") or [])) == want]
    if not recs:
        raise SystemExit("no capture with a settled ARB (and matching V_thr)")
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(recs):
        v = r.get("vthr_mv")
        key = "(V_thr not set)" if not v else (f"{v[0]:g} all" if len(set(v)) == 1
                                                else ",".join(f"{x:g}" for x in v))
        groups[key].append(i)
    fpga = np.array([r.get("fpga") == r["label"] for r in recs], dtype=float)
    cols = [(run.name, score(recs, load(run)[0], names)) for run in args.run]
    print(f"{len(recs)} captures, {len({r['ref'] for r in recs})} ARBs")
    print(f"{'V_thr':<40}{'n':>5}{'FPGA':>8}" + "".join(f"{n[:14]:>16}" for n, _ in cols))
    for key, idx in sorted(groups.items()):
        print(f"{key[:39]:<40}{len(idx):>5}{fpga[idx].mean():>8.3f}"
              + "".join(f"{s[idx].mean():>16.3f}" for _, s in cols))
    print(f"{'all':<40}{len(recs):>5}{fpga.mean():>8.3f}" + "".join(f"{s.mean():>16.3f}" for _, s in cols))
    print("FPGA = the answer the board gave (the weights it was built with)")


if __name__ == "__main__":
    main()
