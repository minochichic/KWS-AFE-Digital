"""Wake-up trigger policies, scored before any firmware exists (wake plan step 1).

docs/wake_stm32_plan.md: the STM32 sleeps, wakes on comparator activity,
captures ONE second and hands it to the FPGA. The open risk is not hardware but
data -- where the word lands inside that second. A trigger on the first edge
puts the word at the left edge of the window, while GSC clips (what the model
was trained on) put it roughly in the middle.

This tool runs the firmware's decision rule frame by frame on spliced streams

    silence (1 s) + target (1 s) + silence (1 s) + silence (1 s, run-out only)

and classifies exactly the windows the firmware would capture. A policy is

    k     channels that must be on in one frame for it to count as "sound"
    run   consecutive sound frames before triggering
    pre   frames kept from BEFORE the onset (0 = M1, pure event wake; >0 needs
          the M2 ring buffer, so the STM32 wakes every 10 ms to fill it)
    hot   channels firing in more than this fraction of silence frames are
          left out of the k count (they are still in the model input) -- the
          STM32 just does not arm EXTI on them. 1.0 = count every channel.

After a capture the scan resumes `busy` frames after its end (UART + FPGA).

The model sees each window exactly as in training: 100 native frames, 14 frames
of -1 on each side. Scores use the float model; pick a policy on val, then run
test once.

    python -m experiments.eval_trigger --tag bd_base --split val --clips-per-class 64
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

NATIVE = 100                 # frames in one capture (1 s at 10 ms)
TARGET_START, TARGET_END = 100, 200
SCAN_END = 300               # triggers only inside the first three seconds


@dataclass(frozen=True)
class TriggerPolicy:
    k: int = 1
    run: int = 1
    pre: int = 0
    hot: float = 1.0
    busy: int = 8            # ~17 ms UART + 57 ms FPGA, rounded up to frames

    def __post_init__(self) -> None:
        if self.k < 1 or self.run < 1:
            raise ValueError("k and run must be >= 1")
        if not 0 <= self.pre < NATIVE:
            raise ValueError("pre must be in [0, NATIVE)")
        if self.busy < 0:
            raise ValueError("busy must be >= 0")

    @property
    def mode(self) -> str:
        return "M1" if self.pre == 0 and self.run == 1 else "M2"

    @property
    def label(self) -> str:
        hot = "all" if self.hot >= 1.0 else f"hot<{self.hot:g}"
        return f"k{self.k}_run{self.run}_pre{self.pre}_{hot}"


@dataclass(frozen=True)
class Capture:
    onset: int               # first frame of the qualifying run
    start: int
    end: int


def hot_mask(silence_frames: np.ndarray, hot: float) -> np.ndarray:
    """Channels that count toward k. silence_frames: [frames, channels] 0/1."""
    rate = silence_frames.mean(axis=0)
    return rate <= hot


def simulate_triggers(stream: np.ndarray, policy: TriggerPolicy,
                      counted: np.ndarray,
                      scan_end: int = SCAN_END) -> List[Capture]:
    """The firmware's IDLE -> CAPTURE -> busy loop over one [T, C] stream."""
    sound = stream[:, counted].sum(axis=1) >= policy.k
    caps: List[Capture] = []
    t, run = 0, 0
    while t < min(scan_end, len(stream)):
        run = run + 1 if sound[t] else 0
        if run >= policy.run:
            onset = t - policy.run + 1
            start = max(0, onset - policy.pre)
            end = start + NATIVE
            if end > len(stream):
                break
            caps.append(Capture(onset, start, end))
            t, run = end + policy.busy, 0
            continue
        t += 1
    return caps


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def score_case(caps: Sequence[Capture], preds: Sequence[int], label: int,
               word: Tuple[int, int], n_keywords: int) -> Dict[str, float]:
    """One stream. `word` is the keyword's frame span (or the target clip)."""
    is_kw = label < n_keywords
    hit = wrong = fa = 0
    cover: Optional[float] = None
    for cap, pred in zip(caps, preds):
        on_word = overlap(cap.start, cap.end, *word) > 0
        if is_kw and on_word:
            if cover is None:
                cover = overlap(cap.start, cap.end, *word) / max(1, word[1] - word[0])
            if pred == label:
                hit = 1
            elif pred < n_keywords:
                wrong = 1
        elif pred < n_keywords:
            fa += 1
    return {"keyword": int(is_kw), "hit": hit, "wrong": wrong, "fa": fa,
            "wakes": len(caps), "cover": cover}


def summarize(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    kw = [r for r in rows if r["keyword"]]
    covers = [r["cover"] for r in kw if r["cover"] is not None]
    n_kw = max(1, len(kw))
    return {
        "recall": sum(r["hit"] for r in kw) / n_kw,
        "wrong": sum(r["wrong"] and not r["hit"] for r in kw) / n_kw,
        "missed_word": sum(r["cover"] is None for r in kw) / n_kw,
        "fa_per_stream": sum(r["fa"] for r in rows) / max(1, len(rows)),
        "wakes_per_stream": sum(r["wakes"] for r in rows) / max(1, len(rows)),
        "cover_mean": float(np.mean(covers)) if covers else 0.0,
        "cover_full": (sum(c >= 0.999 for c in covers) / len(covers)
                       if covers else 0.0),
    }


def _padded(window: np.ndarray, pad: int = 14) -> np.ndarray:
    """[100, C] 0/1 -> [C, 128] +-1, -1 padding, as in training."""
    out = np.full((window.shape[1], NATIVE + 2 * pad), -1, dtype=np.int8)
    out[:, pad:pad + NATIVE] = window.T.astype(np.int8) * 2 - 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="bd_base")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--clips-per-class", type=int, default=64)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--run", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--pre", type=int, nargs="+", default=[0, 10, 20, 30])
    ap.add_argument("--hot", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--busy", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--word-active-frac", type=float, default=0.02)
    ap.add_argument("--out", default="", help="CSV; default out/trigger/<tag>_<split>.csv")
    args = ap.parse_args()

    import torch
    from data.afe import AFEFrontend, load_afe_state
    from data.speech_commands import KEYWORDS, SILENCE_INDEX, build_dataloaders
    from experiments.eval_streaming import _infer, _resolve_device, collect_balanced_waveforms
    from experiments.window_offset import FRAME, word_span
    from models.binary_matchboxnet import BinaryMatchboxNet
    from train.config import load_config

    run_dir = Path(args.runs) / args.tag
    cfg = load_config(str(run_dir / "config.yaml"))
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    if "afe" not in ck:
        raise SystemExit("frame-trained run (no AFE state) -- this tool needs audio")
    device = _resolve_device(args.device)
    model = BinaryMatchboxNet(cfg.model)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    afe = AFEFrontend(cfg.afe)
    load_afe_state(afe, ck["afe"])
    afe.to(device).eval()

    split = {"train": 0, "val": 1, "test": 2}[args.split]
    loaders = build_dataloaders(cfg.data, args.batch_size, cfg.afe.sample_rate,
                                num_workers=0, seed=cfg.train.seed)
    _, waves, labels = collect_balanced_waveforms(
        loaders[split], args.clips_per_class, cfg.model.n_classes, args.seed)
    with torch.no_grad():
        bits = np.concatenate([
            (afe(waves[i:i + args.batch_size].to(device), target_T=NATIVE) > 0)
            .to(torch.uint8).cpu().numpy()
            for i in range(0, len(waves), args.batch_size)])          # [N, C, 100]
    wa, wb = word_span(waves, active_frac=args.word_active_frac)
    n_kw = len(KEYWORDS)

    silence = np.flatnonzero(labels == SILENCE_INDEX)
    if silence.size == 0:
        raise SystemExit(f"split {args.split!r} has no silence clips")
    sil_frames = bits[silence].transpose(0, 2, 1).reshape(-1, bits.shape[1])
    rate = sil_frames.mean(axis=0)
    print("silence firing rate per channel:",
          " ".join(f"{r:.2f}" for r in rate))

    rng = np.random.default_rng(args.seed + 1)
    streams, words = [], []
    for i, label in enumerate(labels):
        b, a, a2 = rng.choice(silence, size=3, replace=True)
        streams.append(np.concatenate(
            [bits[b], bits[i], bits[a], bits[a2]], axis=1).T)          # [400, C]
        if label < n_kw and int(wb[i]) > int(wa[i]):
            words.append((TARGET_START + int(wa[i]) // FRAME,
                          TARGET_START + int(wb[i]) // FRAME))
        else:
            words.append((TARGET_START, TARGET_END))

    # center window = the clip itself: the bridge back to clip accuracy
    _, center_pred = _infer(model, np.stack([_padded(s[TARGET_START:TARGET_END])
                                             for s in streams]),
                            device, args.batch_size)
    print(f"clip accuracy on these {len(labels)} clips (center window): "
          f"{(center_pred == labels).mean():.4f}")

    cache: Dict[Tuple[int, int], int] = {}
    policies = [TriggerPolicy(k, r, p, h, args.busy) for k, r, p, h in
                itertools.product(args.k, args.run, args.pre, args.hot)]
    plan = []
    for pol in policies:
        counted = hot_mask(sil_frames, pol.hot)
        caps = [simulate_triggers(s, pol, counted) for s in streams]
        plan.append((pol, caps))
        for i, cs in enumerate(caps):
            for c in cs:
                cache.setdefault((i, c.start), -1)
    keys = list(cache)
    if keys:
        _, pred = _infer(model, np.stack([_padded(streams[i][s:s + NATIVE])
                                          for i, s in keys]),
                         device, args.batch_size)
        cache = dict(zip(keys, pred.tolist()))

    results = []
    for pol, caps in plan:
        rows = [score_case(cs, [cache[(i, c.start)] for c in cs],
                           int(labels[i]), words[i], n_kw)
                for i, cs in enumerate(caps)]
        results.append({**asdict(pol), "mode": pol.mode, "label": pol.label,
                        "counted_channels": int(hot_mask(sil_frames, pol.hot).sum()),
                        **summarize(rows)})
    results.sort(key=lambda r: (-r["recall"], r["fa_per_stream"]))

    print(f"\n{'policy':28s} mode  ch  recall  wrong  missed  FA/str  wakes  cover  full")
    for r in results:
        print(f"{r['label']:28s} {r['mode']:4s} {r['counted_channels']:3d}  "
              f"{r['recall']:.3f}  {r['wrong']:.3f}  {r['missed_word']:.3f}  "
              f"{r['fa_per_stream']:.3f}  {r['wakes_per_stream']:.2f}  "
              f"{r['cover_mean']:.2f}  {r['cover_full']:.2f}")

    out = Path(args.out or f"out/trigger/{args.tag}_{args.split}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
