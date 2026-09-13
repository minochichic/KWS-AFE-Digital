"""Sweep a keyword-versus-quiet logit margin on a streaming window trace.

``eval_streaming`` stores every window's logits.  This tool reuses those
scores, so gate and vote policies can be explored without running the AFE or
neural network again.  A raw keyword prediction is accepted only when::

    keyword_logit - max(silence_logit, unknown_logit) >= margin

Rejected keyword predictions are treated as silence before the consecutive
vote.  Use validation traces to choose the policy; evaluate the selected
policy on test only after it is frozen.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from experiments.streaming_window import VoteConfig, vote_sequence


@dataclass(frozen=True)
class TraceWindow:
    window_id: int
    end_frame: int
    target_frames: int
    prediction: int
    logits: Tuple[float, ...]


@dataclass(frozen=True)
class TraceCase:
    case_id: int
    target: int
    target_name: str
    windows: Tuple[TraceWindow, ...]


def read_trace(path: Path) -> List[TraceCase]:
    """Read and validate an ``eval_streaming`` window CSV."""

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("trace has no CSV header")
        logit_fields = sorted(
            (name for name in reader.fieldnames if name.startswith("logit_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        if not logit_fields:
            raise ValueError("trace has no logit columns")
        grouped: Dict[int, List[Dict[str, str]]] = {}
        for row in reader:
            grouped.setdefault(int(row["case_id"]), []).append(row)

    cases: List[TraceCase] = []
    for case_id, rows in sorted(grouped.items()):
        targets = {int(row["target"]) for row in rows}
        names = {row["target_name"] for row in rows}
        if len(targets) != 1 or len(names) != 1:
            raise ValueError(f"case {case_id} changes target within the stream")
        windows = tuple(
            TraceWindow(
                window_id=int(row["window_id"]),
                end_frame=int(row["end_frame"]),
                target_frames=int(row["target_frames"]),
                prediction=int(row["prediction"]),
                logits=tuple(float(row[name]) for name in logit_fields),
            )
            for row in sorted(rows, key=lambda row: int(row["window_id"]))
        )
        if any(len(w.logits) != len(logit_fields) for w in windows):
            raise ValueError(f"case {case_id} has inconsistent logit widths")
        cases.append(TraceCase(case_id, targets.pop(), names.pop(), windows))
    if not cases:
        raise ValueError("trace contains no cases")
    return cases


def keyword_quiet_margin(
    window: TraceWindow,
    quiet_classes: Sequence[int],
) -> float:
    """Return predicted-keyword score minus the strongest quiet score."""

    if window.prediction in quiet_classes:
        raise ValueError("margin is defined only for a keyword prediction")
    return window.logits[window.prediction] - max(
        window.logits[i] for i in quiet_classes
    )


def gate_predictions(
    windows: Sequence[TraceWindow],
    margin: float,
    quiet_classes: Tuple[int, ...] = (10, 11),
    rejected_class: int = 10,
) -> List[int]:
    """Replay the gate, including its original quiet/re-arm semantics."""

    if rejected_class not in quiet_classes:
        raise ValueError("rejected_class must be a quiet class")
    return [
        window.prediction
        if (window.prediction in quiet_classes or margin <= 0.0
            or keyword_quiet_margin(window, quiet_classes) >= margin)
        else rejected_class
        for window in windows
    ]


def evaluate_gate(
    cases: Sequence[TraceCase],
    margin: float,
    required_consecutive: int,
    cooldown_windows: int,
    quiet_classes: Tuple[int, ...] = (10, 11),
    rejected_class: int = 10,
    target_start_frame: int = 100,
    frame_ms: int = 10,
) -> Dict[str, object]:
    """Apply one margin and vote policy to in-memory trace cases."""

    n_classes = len(cases[0].windows[0].logits)
    if rejected_class not in quiet_classes:
        raise ValueError("rejected_class must be a quiet class")
    vote_cfg = VoteConfig(
        required_consecutive=required_consecutive,
        cooldown_windows=cooldown_windows,
        n_classes=n_classes,
        quiet_classes=quiet_classes,
    )
    keyword_cases = 0
    keyword_hits = 0
    quiet_cases = 0
    quiet_false_streams = 0
    quiet_by_class: Dict[int, List[int]] = {
        label: [0, 0] for label in quiet_classes
    }  # [false streams, total streams]
    wrong_detections = 0
    outside_detections = 0
    duplicate_detections = 0
    raw_keyword_windows = 0
    accepted_keyword_windows = 0
    latencies: List[int] = []

    for case in cases:
        gated = gate_predictions(case.windows, margin, quiet_classes, rejected_class)
        raw_keyword_windows += sum(w.prediction not in quiet_classes for w in case.windows)
        accepted_keyword_windows += sum(p not in quiet_classes for p in gated)

        steps = vote_sequence(
            gated,
            (window.window_id for window in case.windows),
            vote_cfg,
        )
        correct = []
        wrong = []
        outside = []
        for window, step in zip(case.windows, steps):
            if step.detection is None:
                continue
            if window.target_frames == 0:
                outside.append((window, step))
            elif (case.target not in quiet_classes
                  and step.detection == case.target):
                correct.append((window, step))
            else:
                wrong.append((window, step))

        wrong_detections += len(wrong)
        outside_detections += len(outside)
        duplicate_detections += max(0, len(correct) - 1)
        if case.target in quiet_classes:
            quiet_cases += 1
            false_stream = int(bool(correct or wrong or outside))
            quiet_false_streams += false_stream
            quiet_by_class[case.target][0] += false_stream
            quiet_by_class[case.target][1] += 1
        else:
            keyword_cases += 1
            if correct:
                keyword_hits += 1
                latencies.append(
                    (correct[0][0].end_frame - target_start_frame) * frame_ms
                )

    return {
        "margin": margin,
        "required_consecutive": required_consecutive,
        "cooldown_windows": cooldown_windows,
        "keyword_cases": keyword_cases,
        "keyword_event_hits": keyword_hits,
        "keyword_event_recall": (
            keyword_hits / keyword_cases if keyword_cases else None
        ),
        "quiet_cases": quiet_cases,
        "quiet_false_alarm_streams": quiet_false_streams,
        "quiet_stream_false_alarm_fraction": (
            quiet_false_streams / quiet_cases if quiet_cases else None
        ),
        "quiet_false_alarm_by_class": {
            str(label): {"false_streams": counts[0], "streams": counts[1]}
            for label, counts in quiet_by_class.items()
        },
        "wrong_detections_while_target_overlaps": wrong_detections,
        "detections_outside_target_overlap": outside_detections,
        "duplicate_correct_detections": duplicate_detections,
        "raw_keyword_windows": raw_keyword_windows,
        "accepted_keyword_windows": accepted_keyword_windows,
        "accepted_keyword_window_fraction": (
            accepted_keyword_windows / raw_keyword_windows
            if raw_keyword_windows else 0.0
        ),
        "median_first_correct_detection_latency_ms": (
            float(np.median(latencies)) if latencies else None
        ),
    }


def _sidecar_summary(trace: Path) -> Dict[str, object]:
    suffix = "_windows.csv"
    if not trace.name.endswith(suffix):
        return {}
    path = trace.with_name(trace.name[: -len(suffix)] + "_summary.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--required-consecutive", type=int, default=0,
                    help="default: value recorded in the sidecar summary")
    ap.add_argument("--cooldown-windows", type=int, default=-1,
                    help="default: value recorded in the sidecar summary")
    ap.add_argument("--margins", type=float, nargs="*", default=None,
                    help="explicit margins; otherwise use validation quantiles")
    ap.add_argument("--quantiles", type=float, nargs="+",
                    default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cases = read_trace(args.trace)
    sidecar = _sidecar_summary(args.trace)
    recorded_vote = sidecar.get("vote", {}) if sidecar else {}
    required = args.required_consecutive or int(
        recorded_vote.get("required_consecutive", 5)
    )
    cooldown = (args.cooldown_windows if args.cooldown_windows >= 0 else
                int(recorded_vote.get("cooldown_windows", 10)))
    quiet_classes = tuple(int(x) for x in
                          recorded_vote.get("quiet_classes", [10, 11]))

    raw_margins = [
        keyword_quiet_margin(window, quiet_classes)
        for case in cases
        for window in case.windows
        if window.prediction not in quiet_classes
    ]
    if not raw_margins:
        raise SystemExit("trace has no keyword-predicted windows")
    if args.margins is None:
        thresholds = [0.0] + [
            max(0.0, float(np.percentile(raw_margins, q)))
            for q in args.quantiles
        ]
    else:
        thresholds = list(args.margins)
    thresholds = sorted({round(value, 12) for value in thresholds})

    results = [
        evaluate_gate(
            cases,
            margin=value,
            required_consecutive=required,
            cooldown_windows=cooldown,
            quiet_classes=quiet_classes,
            rejected_class=quiet_classes[0],
        )
        for value in thresholds
    ]

    names = {
        int(item["class"]): str(item["name"])
        for item in sidecar.get("per_class", [])
    }
    quiet_names = [names.get(i, str(i)) for i in quiet_classes]
    print(f"trace: {args.trace}; cases: {len(cases)}; N={required}; "
          f"cooldown={cooldown}")
    print(f"{'margin':>11} {'keep':>7} {'recall':>8} {'quietFA':>8} "
          f"{quiet_names[0] + 'FA':>12} {quiet_names[1] + 'FA':>12} "
          f"{'wrong':>7} {'latency':>8}")
    for result in results:
        by_class = result["quiet_false_alarm_by_class"]
        quiet_text = [
            f"{by_class[str(i)]['false_streams']}/{by_class[str(i)]['streams']}"
            for i in quiet_classes
        ]
        latency = result["median_first_correct_detection_latency_ms"]
        print(
            f"{result['margin']:>11.6g} "
            f"{result['accepted_keyword_window_fraction']:>7.3f} "
            f"{result['keyword_event_recall']:>8.4f} "
            f"{result['quiet_stream_false_alarm_fraction']:>8.4f} "
            f"{quiet_text[0]:>12} {quiet_text[1]:>12} "
            f"{result['wrong_detections_while_target_overlaps']:>7} "
            f"{str(latency):>8}"
        )

    out = args.out or args.trace.with_name(
        args.trace.stem.removesuffix("_windows") + "_margin_sweep.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for result in results:
        row = {key: value for key, value in result.items()
               if key != "quiet_false_alarm_by_class"}
        for label, counts in result["quiet_false_alarm_by_class"].items():
            row[f"quiet_{label}_false_streams"] = counts["false_streams"]
            row[f"quiet_{label}_streams"] = counts["streams"]
        flat_rows.append(row)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"sweep: {out}")


if __name__ == "__main__":
    main()
