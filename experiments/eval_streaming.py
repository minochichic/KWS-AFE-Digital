"""Evaluate a trained model on synthetic continuous AFE frame streams.

Each case is a three-second, already-binarized AFE stream made from clips in
one official split::

    silence (1 s) + target (1 s) + silence (1 s)

The stream is passed through :mod:`experiments.streaming_window`, producing
one 1-second snapshot every 100 ms.  The model predictions then pass through
the configurable consecutive-vote state machine.  The center snapshot
``[100, 200)`` is exactly the original target clip, including the model's
14-frame padding on both sides; its accuracy is therefore the bridge back to
ordinary clip evaluation.

This is a controlled spliced-recording stress test.  It checks window motion,
prediction stability, and event logic, but it is not a field false-alarm or
continuous-speech accuracy measurement.

Example::

    python -m experiments.eval_streaming --tag bd_base --split val \
        --clips-per-class 32
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from experiments.streaming_window import (
    VoteConfig,
    VoteStep,
    WindowSnapshot,
    WindowSpec,
    make_snapshots,
    vote_sequence,
)


TARGET_START_FRAME = 100
TARGET_END_FRAME = 200


def select_balanced_indices(
    labels: Sequence[int],
    clips_per_class: int,
    n_classes: int,
    seed: int,
) -> List[int]:
    """Choose up to ``clips_per_class`` examples from every class.

    Selection is deterministic for a given seed and does not duplicate clips.
    Classes are emitted in numerical order so output files remain easy to
    compare between runs.
    """

    if clips_per_class <= 0:
        raise ValueError("clips_per_class must be positive")
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")

    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError(f"labels must be one-dimensional, got {y.shape}")
    if np.any((y < 0) | (y >= n_classes)):
        raise ValueError("labels contain a class outside n_classes")

    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for label in range(n_classes):
        candidates = np.flatnonzero(y == label)
        count = min(clips_per_class, candidates.size)
        if count:
            pick = rng.choice(candidates, size=count, replace=False)
            selected.extend(sorted(int(i) for i in pick))
    return selected


def splice_clips(
    before: np.ndarray,
    target: np.ndarray,
    after: np.ndarray,
    spec: WindowSpec = WindowSpec(),
) -> np.ndarray:
    """Join three ``[channel, time]`` binary clips into a time-major stream."""

    expected = (spec.n_channels, spec.native_frames)
    clips = [np.asarray(x) for x in (before, target, after)]
    for i, clip in enumerate(clips):
        if clip.shape != expected:
            raise ValueError(f"clip {i} shape must be {expected}, got {clip.shape}")
        if not np.all((clip == 0) | (clip == 1)):
            raise ValueError(f"clip {i} must contain only 0 or 1")
    return np.concatenate(clips, axis=1).T.astype(np.uint8, copy=True)


def interval_frames_in_window(
    snapshot: WindowSnapshot,
    start_frame: int,
    end_frame: int,
) -> int:
    """Number of interval frames present in one half-open snapshot."""

    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("interval must be non-negative and non-empty")
    return max(
        0,
        min(snapshot.end_frame, end_frame)
        - max(snapshot.start_frame, start_frame),
    )


def target_frames_in_window(
    snapshot: WindowSnapshot,
    target_start: int = TARGET_START_FRAME,
    target_end: int = TARGET_END_FRAME,
) -> int:
    """Number of inserted-clip frames present in one snapshot interval."""

    return interval_frames_in_window(snapshot, target_start, target_end)


def _padded_clip(clip: np.ndarray, spec: WindowSpec) -> np.ndarray:
    out = np.full(
        (spec.n_channels, spec.model_frames), -1, dtype=np.int8
    )
    lo = spec.pad_left
    out[:, lo : lo + spec.native_frames] = clip.astype(np.int8) * 2 - 1
    return out


def _infer(
    model: torch.nn.Module,
    inputs: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    logits: List[np.ndarray] = []
    with torch.no_grad():
        for lo in range(0, len(inputs), batch_size):
            x = torch.as_tensor(
                inputs[lo : lo + batch_size],
                dtype=torch.float32,
                device=device,
            )
            logits.append(model(x).detach().cpu().numpy())
    scores = np.concatenate(logits, axis=0)
    return scores, scores.argmax(axis=1)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    return device


def collect_balanced_waveforms(
    loader,
    clips_per_class: int,
    n_classes: int,
    seed: int,
) -> Tuple[List[int], torch.Tensor, np.ndarray]:
    """Reservoir-sample balanced waveforms without retaining the full split."""

    if clips_per_class <= 0:
        raise ValueError("clips_per_class must be positive")
    rng = np.random.default_rng(seed)
    seen = np.zeros(n_classes, dtype=np.int64)
    reservoir: List[List[Tuple[int, torch.Tensor]]] = [
        [] for _ in range(n_classes)
    ]
    source_index = 0
    for waves, labels in loader:
        for wave, label_value in zip(waves, labels):
            label = int(label_value)
            if label < 0 or label >= n_classes:
                raise ValueError(f"label {label} is outside n_classes")
            seen[label] += 1
            item = (source_index, wave.detach().cpu().clone())
            bucket = reservoir[label]
            if len(bucket) < clips_per_class:
                bucket.append(item)
            else:
                replace = int(rng.integers(0, seen[label]))
                if replace < clips_per_class:
                    bucket[replace] = item
            source_index += 1

    chosen: List[Tuple[int, torch.Tensor, int]] = []
    for label, bucket in enumerate(reservoir):
        chosen.extend((i, wave, label) for i, wave in sorted(bucket))
    if not chosen:
        raise ValueError("data loader produced no waveforms")
    return (
        [i for i, _, _ in chosen],
        torch.stack([wave for _, wave, _ in chosen]),
        np.asarray([label for _, _, label in chosen], dtype=np.int64),
    )


def _write_csv(path: Path, rows: List[Dict[str, object]], n_classes: int) -> None:
    fields = [
        "case_id",
        "sample_index",
        "target",
        "target_name",
        "window_id",
        "start_frame",
        "end_frame",
        "target_frames",
        "word_start_frame",
        "word_end_frame",
        "word_frames",
        "word_fraction",
        "is_center",
        "prediction",
        "prediction_name",
        "candidate",
        "streak",
        "cooldown_remaining",
        "armed",
        "detection",
        "detection_name",
    ] + [f"logit_{i}" for i in range(n_classes)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="bd_base")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--csv-root", default="",
                    help="override data.analog_csv_root from config.yaml")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--clips-per-class", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--required-consecutive", type=int, default=3)
    ap.add_argument("--cooldown-windows", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--device", default="auto",
                    help="auto, cpu, cuda, or a device such as cuda:1")
    ap.add_argument("--hop-frames", type=int, default=10,
                    help="snapshot stride in 10 ms frames (default: 10 = 100 ms)")
    ap.add_argument("--word-active-frac", type=float, default=0.02,
                    help="peak-relative 10 ms RMS gate for keyword boundaries")
    ap.add_argument("--out", default="",
                    help="output prefix; default out/streaming/<tag>_<split>")
    args = ap.parse_args()
    if args.hop_frames <= 0 or TARGET_START_FRAME % args.hop_frames:
        ap.error("--hop-frames must be a positive divisor of 100")
    if not 0.0 < args.word_active_frac < 1.0:
        ap.error("--word-active-frac must be between 0 and 1")

    from data.speech_commands import KEYWORDS, SILENCE_INDEX, UNKNOWN_INDEX
    from models.binary_matchboxnet import BinaryMatchboxNet
    from train.config import load_config

    names = KEYWORDS + ["_silence_", "_unknown_"]
    run = Path(args.runs) / args.tag
    cfg = load_config(str(run / "config.yaml"))
    if args.csv_root:
        cfg.data.analog_csv_root = args.csv_root
    if cfg.model.n_classes != len(names):
        raise SystemExit(
            f"this evaluator expects {len(names)} classes, got "
            f"{cfg.model.n_classes}"
        )

    spec = WindowSpec(hop_frames=args.hop_frames)
    if cfg.model.in_channels != spec.n_channels or cfg.model.T != spec.model_frames:
        raise SystemExit(
            "checkpoint geometry does not match the streaming contract: "
            f"model is [{cfg.model.in_channels}, {cfg.model.T}], expected "
            f"[{spec.n_channels}, {spec.model_frames}]"
        )

    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model = BinaryMatchboxNet(cfg.model)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    split_index = {"train": 0, "val": 1, "test": 2}[args.split]
    if getattr(cfg.data, "analog_csv_root", ""):
        from data.analog_spectrogram import build_analog_dataloaders

        loaders = build_analog_dataloaders(
            cfg.data,
            batch_size=args.batch_size,
            target_T=cfg.model.T,
            num_workers=0,
            seed=cfg.train.seed,
        )
        dataset = loaders[split_index].dataset
        all_bits = np.asarray(dataset.bits)
        all_labels = np.asarray(dataset.labels)
        selected = select_balanced_indices(
            all_labels,
            clips_per_class=args.clips_per_class,
            n_classes=cfg.model.n_classes,
            seed=args.seed,
        )
        sample_indices = selected
        bits = all_bits[selected]
        labels = all_labels[selected]
        input_source = "analog_csv"
        word_intervals: List[Tuple[int, int] | None] = [None] * len(labels)
    else:
        if "afe" not in checkpoint:
            raise SystemExit(
                "checkpoint has no saved AFE state; this is a frame model, so "
                "set data.analog_csv_root or pass --csv-root"
            )
        from data.afe import AFEFrontend, load_afe_state
        from data.speech_commands import build_dataloaders

        loaders = build_dataloaders(
            cfg.data,
            args.batch_size,
            cfg.afe.sample_rate,
            num_workers=0,
            seed=cfg.train.seed,
        )
        sample_indices, waves, labels = collect_balanced_waveforms(
            loaders[split_index],
            clips_per_class=args.clips_per_class,
            n_classes=cfg.model.n_classes,
            seed=args.seed,
        )
        from experiments.window_offset import FRAME as WORD_FRAME, word_span

        if cfg.afe.sample_rate != 100 * WORD_FRAME:
            raise SystemExit(
                "word-boundary diagnostic expects 10 ms frames at the AFE "
                f"sample rate, got {cfg.afe.sample_rate} Hz"
            )
        word_a, word_b = word_span(waves, active_frac=args.word_active_frac)
        word_intervals = []
        for label, a, b in zip(labels, word_a.tolist(), word_b.tolist()):
            if int(label) < len(KEYWORDS):
                if b <= a:
                    raise RuntimeError(
                        f"keyword label {int(label)} has no detectable word span"
                    )
                word_intervals.append((
                    TARGET_START_FRAME + int(a) // WORD_FRAME,
                    TARGET_START_FRAME + int(b) // WORD_FRAME,
                ))
            else:
                word_intervals.append(None)
        afe = AFEFrontend(cfg.afe)
        load_afe_state(afe, checkpoint["afe"])
        afe.to(device).eval()
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for lo in range(0, len(waves), args.batch_size):
                native = afe(
                    waves[lo : lo + args.batch_size].to(device),
                    target_T=spec.native_frames,
                )
                chunks.append((native > 0).to(torch.uint8).cpu().numpy())
        bits = np.concatenate(chunks, axis=0)
        input_source = "software_afe_from_wav"

    if bits.ndim != 3 or bits.shape[1:] != (
        spec.n_channels,
        spec.native_frames,
    ):
        raise SystemExit(
            "selected clips must have shape [N, 16, 100], got "
            f"{bits.shape}"
        )
    silence_pool = np.flatnonzero(labels == SILENCE_INDEX)
    if silence_pool.size == 0:
        raise SystemExit(f"split {args.split!r} contains no silence clips")

    rng = np.random.default_rng(args.seed + 1)
    cases: List[
        Tuple[int, int, Tuple[int, int] | None, List[WindowSnapshot]]
    ] = []
    all_inputs: List[np.ndarray] = []
    for local_index, (sample_index, word_interval) in enumerate(
        zip(sample_indices, word_intervals)
    ):
        before_i, after_i = rng.choice(silence_pool, size=2, replace=True)
        stream = splice_clips(
            bits[int(before_i)], bits[local_index], bits[int(after_i)], spec
        )
        snapshots = make_snapshots(stream, spec)
        expected_snapshots = (
            (stream.shape[0] - spec.native_frames) // spec.hop_frames + 1
        )
        if len(snapshots) != expected_snapshots:
            raise RuntimeError(
                "three-second stream produced "
                f"{len(snapshots)} windows, expected {expected_snapshots}"
            )
        center = next((s for s in snapshots if s.start_frame == TARGET_START_FRAME), None)
        if center is None or not np.array_equal(
            center.model_input, _padded_clip(bits[local_index], spec)
        ):
            raise RuntimeError("center snapshot does not reproduce the source clip")
        cases.append((
            sample_index, int(labels[local_index]), word_interval, snapshots
        ))
        all_inputs.extend(s.model_input for s in snapshots)

    input_array = np.stack(all_inputs)
    logits, predictions = _infer(model, input_array, device, args.batch_size)
    vote_cfg = VoteConfig(
        required_consecutive=args.required_consecutive,
        cooldown_windows=args.cooldown_windows,
        n_classes=cfg.model.n_classes,
        quiet_classes=(SILENCE_INDEX, UNKNOWN_INDEX),
    )

    rows: List[Dict[str, object]] = []
    per_class = [
        {
            "class": i,
            "name": names[i],
            "clips": 0,
            "center_correct": 0,
            "event_hits": 0,
            "misses": 0,
            "false_alarm_streams": 0,
            "correct_detections": 0,
            "wrong_detections": 0,
            "outside_detections": 0,
            "duplicate_correct_detections": 0,
        }
        for i in range(cfg.model.n_classes)
    ]
    cursor = 0
    first_latencies_ms: List[int] = []
    for case_id, (
        sample_index, target, word_interval, snapshots
    ) in enumerate(cases):
        n = len(snapshots)
        case_logits = logits[cursor : cursor + n]
        case_predictions = predictions[cursor : cursor + n]
        cursor += n
        steps: List[VoteStep] = vote_sequence(
            (int(p) for p in case_predictions),
            (s.window_id for s in snapshots),
            vote_cfg,
        )

        stats = per_class[target]
        stats["clips"] += 1
        center_i = next(i for i, s in enumerate(snapshots)
                        if s.start_frame == TARGET_START_FRAME)
        stats["center_correct"] += int(case_predictions[center_i] == target)

        correct = []
        wrong = []
        outside = []
        for snapshot, step in zip(snapshots, steps):
            if step.detection is None:
                continue
            overlap = target_frames_in_window(snapshot)
            if overlap == 0:
                outside.append((snapshot, step))
            elif step.detection == target and target not in vote_cfg.quiet_classes:
                correct.append((snapshot, step))
            else:
                wrong.append((snapshot, step))

        stats["correct_detections"] += len(correct)
        stats["wrong_detections"] += len(wrong)
        stats["outside_detections"] += len(outside)
        stats["duplicate_correct_detections"] += max(0, len(correct) - 1)
        if target in vote_cfg.quiet_classes:
            stats["false_alarm_streams"] += int(bool(correct or wrong or outside))
        elif correct:
            stats["event_hits"] += 1
            first_latencies_ms.append(
                (correct[0][0].end_frame - TARGET_START_FRAME) * 10
            )
        else:
            stats["misses"] += 1

        for snapshot, score, prediction, step in zip(
            snapshots, case_logits, case_predictions, steps
        ):
            overlap = target_frames_in_window(snapshot)
            if word_interval is None:
                word_start = word_end = word_overlap = word_fraction = ""
            else:
                word_start, word_end = word_interval
                word_overlap = interval_frames_in_window(
                    snapshot, word_start, word_end
                )
                word_fraction = f"{word_overlap / (word_end - word_start):.6f}"
            row: Dict[str, object] = {
                "case_id": case_id,
                "sample_index": sample_index,
                "target": target,
                "target_name": names[target],
                "window_id": snapshot.window_id,
                "start_frame": snapshot.start_frame,
                "end_frame": snapshot.end_frame,
                "target_frames": overlap,
                "word_start_frame": word_start,
                "word_end_frame": word_end,
                "word_frames": word_overlap,
                "word_fraction": word_fraction,
                "is_center": int(snapshot.start_frame == TARGET_START_FRAME),
                "prediction": int(prediction),
                "prediction_name": names[int(prediction)],
                "candidate": "" if step.candidate is None else step.candidate,
                "streak": step.streak,
                "cooldown_remaining": step.cooldown_remaining,
                "armed": int(step.armed),
                "detection": "" if step.detection is None else step.detection,
                "detection_name": (
                    "" if step.detection is None else names[step.detection]
                ),
            }
            row.update({f"logit_{i}": f"{float(v):.8g}"
                        for i, v in enumerate(score)})
            rows.append(row)

    total = sum(int(x["clips"]) for x in per_class)
    center_correct = sum(int(x["center_correct"]) for x in per_class)
    keyword_clips = sum(int(per_class[i]["clips"]) for i in range(len(KEYWORDS)))
    event_hits = sum(int(per_class[i]["event_hits"]) for i in range(len(KEYWORDS)))
    quiet_clips = sum(int(per_class[i]["clips"])
                      for i in (SILENCE_INDEX, UNKNOWN_INDEX))
    quiet_false_streams = sum(int(per_class[i]["false_alarm_streams"])
                              for i in (SILENCE_INDEX, UNKNOWN_INDEX))
    wrong_detections = sum(int(x["wrong_detections"]) for x in per_class)
    outside_detections = sum(int(x["outside_detections"]) for x in per_class)
    duplicate_detections = sum(
        int(x["duplicate_correct_detections"]) for x in per_class
    )

    summary = {
        "schema_version": 2,
        "scope": "synthetic_spliced_three_second_stream",
        "warning": (
            "Controlled spliced-recording stress test; do not interpret as "
            "field continuous-speech accuracy or false alarms per hour."
        ),
        "tag": args.tag,
        "checkpoint": str(run / "best.pt"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "input_source": input_source,
        "analog_csv_root": str(cfg.data.analog_csv_root),
        "device": str(device),
        "seed": args.seed,
        "clips_per_class_cap": args.clips_per_class,
        "window": asdict(spec),
        "vote": asdict(vote_cfg),
        "target_interval_frames": [TARGET_START_FRAME, TARGET_END_FRAME],
        "word_boundary": {
            "available": input_source == "software_afe_from_wav",
            "method": "10_ms_rms_relative_to_clip_peak",
            "active_fraction": args.word_active_frac,
            "quiet_classes_recorded_as_blank": True,
        },
        "metrics": {
            "clips": total,
            "windows": len(rows),
            "center_correct": center_correct,
            "center_accuracy": center_correct / total if total else None,
            "keyword_clips": keyword_clips,
            "keyword_event_hits": event_hits,
            "keyword_event_recall": event_hits / keyword_clips
            if keyword_clips else None,
            "quiet_clips": quiet_clips,
            "quiet_false_alarm_streams": quiet_false_streams,
            "quiet_stream_false_alarm_fraction": quiet_false_streams / quiet_clips
            if quiet_clips else None,
            "wrong_detections_while_target_overlaps": wrong_detections,
            "detections_outside_target_overlap": outside_detections,
            "duplicate_correct_detections": duplicate_detections,
            "synthetic_quiet_exposure_seconds": quiet_clips * 3,
            "median_first_correct_detection_latency_ms": float(
                np.median(first_latencies_ms)
            ) if first_latencies_ms else None,
        },
        "per_class": per_class,
    }

    prefix = Path(args.out) if args.out else Path(
        "out/streaming"
    ) / f"{args.tag}_{args.split}"
    csv_path = prefix.parent / f"{prefix.name}_windows.csv"
    json_path = prefix.parent / f"{prefix.name}_summary.json"
    _write_csv(csv_path, rows, cfg.model.n_classes)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\ncheckpoint: {run / 'best.pt'} (epoch {checkpoint.get('epoch')})")
    print(f"device: {device}; split: {args.split}; cases: {total}; windows: {len(rows)}")
    print(f"center-window accuracy: {center_correct}/{total} "
          f"({center_correct / total:.4f})")
    print(f"keyword event recall: {event_hits}/{keyword_clips} "
          f"({event_hits / keyword_clips:.4f})")
    print(f"quiet streams with an event: {quiet_false_streams}/{quiet_clips} "
          f"({quiet_false_streams / quiet_clips:.4f})")
    if first_latencies_ms:
        print("median first correct event latency: "
              f"{np.median(first_latencies_ms):.0f} ms")
    print(f"window trace: {csv_path}")
    print(f"summary: {json_path}")
    print("Interpret these as spliced-stream diagnostics, not field metrics.")


if __name__ == "__main__":
    main()
