"""Wake-up trigger policies, scored before any firmware exists (wake plan step 1).

docs/wake_stm32_plan.md: the STM32 sleeps, wakes on comparator activity,
captures ONE second and hands it to the FPGA. Two things can go wrong, and
neither is hardware: the trigger may never be quiet (so the MCU never sleeps),
and the word may land in the wrong place inside the captured second.

This tool runs the firmware's decision rule frame by frame on spliced streams

    silence (LEAD frames, random 50..150) + target (1 s) + silence (2 s)

and classifies exactly the windows the firmware would capture. LEAD is random
per stream ON PURPOSE: the first version used a fixed 1 s, and when the trigger
fired on the first frame and re-fired right after each capture, the second
window always started at 108 - pre -- pre=10 lined it up with the target to
within 2 frames by arithmetic, not by detecting anything. A random lead breaks
that coincidence.

A policy is

    k      channels that must be on in one frame for it to count as "sound"
    delta  >0: ALSO need `on >= background + delta`, background being an EMA of
           the on-count over idle frames (1/32 per frame, ~0.3 s). Measured on
           bd_base, GSC silence keeps 15-89% of the comparators on (ch15 89%),
           so an absolute k is satisfied by the noise itself; delta asks for
           a change instead. 0 = absolute k only.
    run    consecutive sound frames before triggering
    pre    frames kept from BEFORE the onset (0 = M1, pure event wake; >0 needs
           the M2 ring buffer -- and delta>0 does too, since the background is
           tracked every 10 ms)
    hot    channels firing in more than this fraction of silence frames are
           left out of the counts (still in the model input). 1.0 = all.

After a capture the scan resumes `busy` frames after its end (UART + FPGA).

Reported per policy: keyword recall, wrong keyword, false alarms per stream,
and DUTY -- the fraction of silence-stream time spent capturing. Duty near 1
means the MCU never sleeps, whatever the recall says.

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
LEAD_MIN, LEAD_MAX = 50, 150
TAIL = 200                   # silence after the target
SCAN_AFTER = 100             # keep scanning 1 s past the target
BG_ALPHA = 1.0 / 32


@dataclass(frozen=True)
class TriggerPolicy:
    k: int = 1
    delta: float = 0.0
    run: int = 1
    pre: int = 0
    hot: float = 1.0
    busy: int = 8            # ~17 ms UART + 57 ms FPGA, rounded up to frames

    def __post_init__(self) -> None:
        if self.k < 1 or self.run < 1:
            raise ValueError("k and run must be >= 1")
        if self.delta < 0:
            raise ValueError("delta must be >= 0")
        if not 0 <= self.pre < NATIVE:
            raise ValueError("pre must be in [0, NATIVE)")
        if self.busy < 0:
            raise ValueError("busy must be >= 0")

    @property
    def mode(self) -> str:
        return ("M1" if self.pre == 0 and self.run == 1 and self.delta == 0
                else "M2")

    @property
    def label(self) -> str:
        hot = "all" if self.hot >= 1.0 else f"hot<{self.hot:g}"
        d = f"_d{self.delta:g}" if self.delta else ""
        return f"k{self.k}{d}_run{self.run}_pre{self.pre}_{hot}"


@dataclass(frozen=True)
class Capture:
    onset: int               # first frame of the qualifying run
    start: int
    end: int


def hot_mask(silence_frames: np.ndarray, hot: float) -> np.ndarray:
    """Channels that are counted. silence_frames: [frames, channels] 0/1."""
    return silence_frames.mean(axis=0) <= hot


def simulate_triggers(stream: np.ndarray, policy: TriggerPolicy,
                      counted: np.ndarray, scan_end: int,
                      bg0: Optional[float] = None) -> List[Capture]:
    """The firmware's IDLE -> CAPTURE -> busy loop over one [T, C] stream.

    bg0 seeds the background on-count (the firmware has been listening before
    the stream starts); None seeds it from the first frame."""
    on = stream[:, counted].sum(axis=1).astype(float)
    bg = float(on[0]) if bg0 is None else float(bg0)
    caps: List[Capture] = []
    t, run = 0, 0
    while t < min(scan_end, len(stream)):
        sound = on[t] >= policy.k and (policy.delta == 0
                                       or on[t] >= bg + policy.delta)
        run = run + 1 if sound else 0
        if not sound:
            bg += BG_ALPHA * (on[t] - bg)    # learn the floor only from quiet
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
               word: Tuple[int, int], n_keywords: int,
               scanned: int) -> Dict[str, float]:
    """One stream. `word` is the keyword's frame span (or the target clip);
    `scanned` is how many frames the stream was watched for (duty's base)."""
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
    awake = sum(min(c.end, scanned) - c.start for c in caps if c.start < scanned)
    return {"keyword": int(is_kw), "hit": hit, "wrong": wrong, "fa": fa,
            "wakes": len(caps), "cover": cover,
            "duty": awake / max(1, scanned)}


def summarize(rows: Sequence[Dict[str, float]],
              quiet: Sequence[bool]) -> Dict[str, float]:
    """`quiet` marks the silence-label streams: duty is measured on those."""
    kw = [r for r in rows if r["keyword"]]
    covers = [r["cover"] for r in kw if r["cover"] is not None]
    q = [r for r, is_q in zip(rows, quiet) if is_q]
    n_kw = max(1, len(kw))
    return {
        "recall": sum(r["hit"] for r in kw) / n_kw,
        "wrong": sum(r["wrong"] and not r["hit"] for r in kw) / n_kw,
        "missed_word": sum(r["cover"] is None for r in kw) / n_kw,
        "fa_per_stream": sum(r["fa"] for r in rows) / max(1, len(rows)),
        "quiet_duty": float(np.mean([r["duty"] for r in q])) if q else 0.0,
        "quiet_wakes": float(np.mean([r["wakes"] for r in q])) if q else 0.0,
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
    ap.add_argument("--k", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--delta", type=float, nargs="+", default=[0, 2, 3, 4, 6])
    ap.add_argument("--run", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--pre", type=int, nargs="+", default=[0, 10, 20, 30])
    ap.add_argument("--hot", type=float, nargs="+", default=[1.0])
    ap.add_argument("--busy", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--word-active-frac", type=float, default=0.02)
    ap.add_argument("--top", type=int, default=30, help="rows to print")
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
    print("silence firing rate per channel:",
          " ".join(f"{r:.2f}" for r in sil_frames.mean(axis=0)))
    on_sil = sil_frames.sum(axis=1)
    print(f"silence on-count per frame: mean {on_sil.mean():.2f}, "
          f"p50 {np.percentile(on_sil, 50):.0f}, p90 {np.percentile(on_sil, 90):.0f}, "
          f"p99 {np.percentile(on_sil, 99):.0f} (of {bits.shape[1]})")

    rng = np.random.default_rng(args.seed + 1)
    streams, words, leads, bg0s = [], [], [], []
    for i, label in enumerate(labels):
        s1, s2, a1, a2, p = rng.choice(silence, size=5, replace=True)
        lead = int(rng.integers(LEAD_MIN, LEAD_MAX + 1))
        before = np.concatenate([bits[s1], bits[s2]], axis=1)[:, -lead:]
        streams.append(np.concatenate(
            [before, bits[i], bits[a1], bits[a2]], axis=1).T)        # [lead+300, C]
        leads.append(lead)
        bg0s.append(float(bits[p].sum(axis=0).mean()))            # listened before
        if label < n_kw and int(wb[i]) > int(wa[i]):
            words.append((lead + int(wa[i]) // FRAME, lead + int(wb[i]) // FRAME))
        else:
            words.append((lead, lead + NATIVE))

    _, center_pred = _infer(model, np.stack([_padded(s[l:l + NATIVE])
                                             for s, l in zip(streams, leads)]),
                            device, args.batch_size)
    print(f"clip accuracy on these {len(labels)} clips (center window): "
          f"{(center_pred == labels).mean():.4f}")

    quiet = [int(l) == SILENCE_INDEX for l in labels]
    policies = [TriggerPolicy(k, d, r, p, h, args.busy) for k, d, r, p, h in
                itertools.product(args.k, args.delta, args.run, args.pre, args.hot)]
    cache: Dict[Tuple[int, int], int] = {}
    plan = []
    for pol in policies:
        counted = hot_mask(sil_frames, pol.hot)
        caps = [simulate_triggers(s, pol, counted, l + NATIVE + SCAN_AFTER, bg0)
                for s, l, bg0 in zip(streams, leads, bg0s)]
        plan.append((pol, counted, caps))
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
    for pol, counted, caps in plan:
        rows = [score_case(cs, [cache[(i, c.start)] for c in cs],
                           int(labels[i]), words[i], n_kw,
                           leads[i] + NATIVE + SCAN_AFTER)
                for i, cs in enumerate(caps)]
        results.append({**asdict(pol), "mode": pol.mode, "label": pol.label,
                        "counted_channels": int(counted.sum()),
                        **summarize(rows, quiet)})
    results.sort(key=lambda r: (-r["recall"], r["quiet_duty"]))

    print(f"\n{'policy':30s} mode  ch  recall  wrong  missed  FA/str  "
          f"q.duty  q.wakes  cover  full")
    for r in results[:args.top]:
        print(f"{r['label']:30s} {r['mode']:4s} {r['counted_channels']:3d}  "
              f"{r['recall']:.3f}  {r['wrong']:.3f}  {r['missed_word']:.3f}  "
              f"{r['fa_per_stream']:.3f}  {r['quiet_duty']:.3f}  "
              f"{r['quiet_wakes']:.2f}     {r['cover_mean']:.2f}  {r['cover_full']:.2f}")
    sleepy = [r for r in results if r["quiet_duty"] <= 0.2]
    if sleepy:
        print(f"\nbest with quiet duty <= 0.2 ({len(sleepy)} policies):")
        for r in sleepy[:10]:
            print(f"  {r['label']:30s} recall {r['recall']:.3f}  "
                  f"q.duty {r['quiet_duty']:.3f}  FA/str {r['fa_per_stream']:.3f}")
    else:
        print("\nno policy keeps quiet duty <= 0.2")

    out = Path(args.out or f"out/trigger/{args.tag}_{args.split}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"\nwritten {out} ({len(results)} policies)")


if __name__ == "__main__":
    main()
