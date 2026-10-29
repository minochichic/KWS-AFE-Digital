"""The float model on the clips the board actually sent (wake plan step 3.5).

firmware/stm32_wake/tools/stream_check.py can only score a capture against
predictions_fixed when the trigger window IS a golden clip (9 of 21). For the
rest the FPGA answered, but nothing said what it should have answered. This
runs best.pt on the very 100 frames the board sent (out/stream/*.json) and
puts its answer next to the FPGA's and the clip's label.

The FPGA runs the exported integer network, not best.pt: the two agree on
most clips but not by construction, so a disagreement here is a fact to look
at, not a failure by itself.

    python -m experiments.stream_captures_model --run runs/bd_base \
        out/stream/stream1_x.json:1 out/stream/stream2_y.json:2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXPECT = Path("firmware/stm32_wake/gen/stream_expect.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/bd_base", help="dir with best.pt + config.yaml")
    ap.add_argument("caps", nargs="+", help="board capture JSON:stream number")
    args = ap.parse_args()

    import torch
    from data.speech_commands import KEYWORDS
    from experiments.eval_streaming import _infer
    from experiments.eval_trigger import _padded
    from models.binary_matchboxnet import BinaryMatchboxNet
    from train.config import load_config

    run = Path(args.run)
    cfg = load_config(str(run / "config.yaml"))
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model = BinaryMatchboxNet(cfg.model)
    model.load_state_dict(ck["model"])
    model.eval()
    names = KEYWORDS + ["silence", "unknown"]
    expect = json.loads(EXPECT.read_text())

    total = agree = right = fpga_right = 0
    for spec in args.caps:
        path, k = spec.rsplit(":", 1)
        caps = json.loads(Path(path).read_text())
        exp = expect[int(k) - 1]
        frames = np.array([[[(f >> c) & 1 for c in range(16)] for f in cap["clip"]]
                           for cap in caps], dtype=np.uint8)           # [N, 100, 16]
        _, pred = _infer(model, np.stack([_padded(x) for x in frames]),
                         torch.device("cpu"), 64)
        print(f"stream {k} ({exp.get('name', '')}): {len(caps)} captures")
        print(f"  {'#':>2} {'end':>5} {'label':>7} {'FPGA':>7} {'best.pt':>7}")
        for i, (cap, p) in enumerate(zip(caps, pred)):
            lab = next((c["label"] for c in exp["captures"] if c["end"] == cap["end"]), None)
            g = cap["got"]
            print(f"  {i + 1:2d} {cap['end']:5d} {names[lab] if lab is not None else '?':>7} "
                  f"{names[g] if g < 12 else hex(g):>7} {names[p]:>7}"
                  f"{'' if g == p else '   <- FPGA != best.pt'}")
            total += 1
            agree += int(g == p)
            right += int(p == lab)
            fpga_right += int(g == lab)
    print(f"\nFPGA == best.pt {agree}/{total}; vs label: best.pt {right}/{total}, "
          f"FPGA {fpga_right}/{total}")


if __name__ == "__main__":
    main()
