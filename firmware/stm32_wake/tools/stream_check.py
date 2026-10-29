"""Compare what the board captured from a stream with what the simulator says.

    python -I stream_check.py caps.json <stream number, 1-based>

caps.json  [{"end": int, "got": int, "clip": [100 ints]}, ...] read off the
           STM32 (board_caps) by read_stats.ps1 -Stream
Expected   firmware/stm32_wake/gen/stream_expect.json from make_streams.py

Three checks per capture, all must hold:
  where   the board's capture ends on the same stream frame as the simulator's
  what    the 100 frames it sent are the simulator's 100 frames, bit for bit
  answer  the FPGA answered (no timeout / CRC error), and where the cut window
          is a golden clip itself, answered that clip's predictions_fixed
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NAMES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
         "silence", "unknown"]
EXPECT = Path(__file__).resolve().parents[1] / "gen" / "stream_expect.json"


def name(c):
    if c is None:
        return "-"
    return {255: "timeout", 254: "CRC"}.get(c, NAMES[c] if c < len(NAMES) else f"#{c}")


def main() -> None:
    caps = json.loads(Path(sys.argv[1]).read_text())
    k = int(sys.argv[2])
    exp = json.loads(EXPECT.read_text())[k - 1]
    want = exp["captures"]
    print(f"stream {k} ({exp['name']}): expected {len(want)} captures, board made {len(caps)}")
    print(f"{'#':>2} {'end sim':>7} {'end board':>9} {'frames':>7} {'FPGA':>8} "
          f"{'model':>6} {'label':>6}  result")
    ok_all = len(caps) == len(want)
    known = known_ok = label_ok = 0
    for i in range(max(len(caps), len(want))):
        w = want[i] if i < len(want) else None
        c = caps[i] if i < len(caps) else None
        if w is None or c is None:
            print(f"{i + 1:>2} {'-' if w is None else w['end']:>7} {'-' if c is None else c['end']:>9}"
                  f"  {'missing' if c is None else 'extra'}")
            ok_all = False
            continue
        where = c["end"] == w["end"]
        what = c["clip"] == w["frames"]
        answered = c["got"] < len(NAMES)
        model = w["expect_class"]
        ans_ok = answered and (model is None or c["got"] == model)
        if model is not None:
            known += 1
            known_ok += c["got"] == model
        label_ok += answered and c["got"] == w["label"]
        good = where and what and ans_ok
        ok_all &= good
        why = "ok" if good else ", ".join(s for s, f in [("where", where), ("frames", what),
                                                        ("answer", ans_ok)] if not f) + " differ"
        print(f"{i + 1:>2} {w['end']:>7} {c['end']:>9} {'same' if what else 'DIFF':>7} "
              f"{name(c['got']):>8} {name(model):>6} {name(w['label']):>6}  {why}")
    print(f"model answer: {known_ok}/{known} known captures match predictions_fixed; "
          f"true label: {label_ok}/{len(want)} (information)")
    print("PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
