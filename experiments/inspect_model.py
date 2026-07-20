"""Build the model from a YAML and report stage table + parameter counts.

    python experiments/inspect_model.py configs/base.yaml
    python experiments/inspect_model.py configs/base.yaml model.C=32 model.T=64
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from experiments.inspect_config import _parse_overrides  # noqa: E402
from models.binary_matchboxnet import BinaryMatchboxNet  # noqa: E402
from train.config import load_config  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cfg = load_config(sys.argv[1], _parse_overrides(sys.argv[2:]))
    model = BinaryMatchboxNet(cfg.model)

    print(f"tag={cfg.tag}  C={cfg.model.C}  T={cfg.model.T}\n")
    print(model.describe())

    x = torch.randint(0, 2, (2, cfg.model.in_channels, cfg.model.T)).float() * 2 - 1
    out = model(x)
    print(f"\ndummy forward: {list(x.shape)} -> {list(out.shape)}")

    for w in cfg.warnings():
        print(f"[warn] {w}")


if __name__ == "__main__":
    main()
