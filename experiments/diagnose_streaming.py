"""Partition streaming misses using saved validation predictions and logits.

This is an offline diagnostic: labels identify where correct predictions were
available, but are never used by the replayed gate or vote. A correct window
does not imply a usable event detector; a hit can coexist with wrong events.
No AFE or neural-network inference is run here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from experiments.streaming_window import VoteConfig, vote_sequence
from experiments.sweep_streaming_gate import TraceCase, gate_predictions, read_trace


def correct_support(
    case: TraceCase, predictions: Sequence[int], required: int,
) -> Tuple[int, int, Optional[int]]:
    """Measure total, consecutive, and M-of-K support for the correct class.

    Only windows overlapping the inserted target clip count. ``min_span`` is
    the shortest adjacent-window interval containing ``required`` correct
    predictions. It is ``None`` when that many correct predictions do not
    occur inside one uninterrupted window-ID segment.
    """

    if len(predictions) != len(case.windows):
        raise ValueError("one prediction is required per window")
    if required <= 0:
        raise ValueError("required must be positive")
    total = longest = streak = 0
    min_span: Optional[int] = None
    hit_ids: List[int] = []
    previous = None
    for window, prediction in zip(case.windows, predictions):
        if previous is not None and window.window_id != previous + 1:
            streak = 0
            hit_ids = []
        if prediction == case.target and window.target_frames > 0:
            total += 1
            streak += 1
            hit_ids.append(window.window_id)
            if len(hit_ids) >= required:
                span = hit_ids[-1] - hit_ids[-required] + 1
                min_span = span if min_span is None else min(min_span, span)
        else:
            streak = 0
        longest = max(longest, streak)
        previous = window.window_id
    return total, longest, min_span


def longest_correct_streak(case: TraceCase, predictions: Sequence[int]) -> int:
    """Count adjacent correct top-1 windows overlapping the target clip."""

    return correct_support(case, predictions, required=1)[1]


def diagnose_case(
    case: TraceCase, margin: float, config: VoteConfig,
) -> Dict[str, object]:
    """Give each case one outcome, plus event counts and observed streaks.

    Miss categories are ordered diagnostics, not independent causal effects:
    a fragmented case might also have an early wrong event. The last category
    means a gated correct streak of sufficient length existed, yet the actual
    stateful policy emitted no correct event (cooldown and/or re-arm block).
    """

    if not case.windows or not config.quiet_classes:
        raise ValueError("case windows and quiet classes must be non-empty")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    gated = gate_predictions(case.windows, margin, config.quiet_classes,
                             config.quiet_classes[0])
    steps = vote_sequence(gated, [w.window_id for w in case.windows], config)
    quiet = case.target in config.quiet_classes
    correct = wrong = outside = 0
    for window, step in zip(case.windows, steps):
        if step.detection is None:
            continue
        if window.target_frames == 0:
            outside += 1
        elif not quiet and step.detection == case.target:
            correct += 1
        else:
            wrong += 1

    raw_count, raw_streak, raw_span = correct_support(
        case, [w.prediction for w in case.windows], config.required_consecutive,
    )
    gated_count, gated_streak, gated_span = correct_support(
        case, gated, config.required_consecutive,
    )
    if quiet:
        outcome = "quiet_false_event" if wrong or outside else "quiet_no_event"
    elif correct:
        outcome = "hit"
    elif raw_streak == 0:
        outcome = "no_correct_window"
    elif raw_streak < config.required_consecutive:
        outcome = "correct_but_not_consecutive"
    elif gated_streak < config.required_consecutive:
        outcome = "margin_broke_correct_streak"
    else:
        outcome = "blocked_after_detection"

    return {
        "case_id": case.case_id,
        "target": case.target,
        "target_name": case.target_name,
        "outcome": outcome,
        "raw_correct_windows": raw_count,
        "raw_max_correct_streak": raw_streak,
        "raw_min_span_for_required": raw_span,
        "gated_correct_windows": gated_count,
        "gated_max_correct_streak": gated_streak,
        "gated_min_span_for_required": gated_span,
        "correct_events": correct,
        "wrong_events": wrong,
        "outside_events": outside,
        "clean_single_hit": int(correct == 1 and wrong == 0 and outside == 0),
    }


def diagnose_cases(
    cases: Sequence[TraceCase], margin: float, config: VoteConfig,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if not cases:
        raise ValueError("trace must contain cases")
    rows = [diagnose_case(case, margin, config) for case in cases]
    keywords = [r for r in rows if r["target"] not in config.quiet_classes]
    quiet = [r for r in rows if r["target"] in config.quiet_classes]
    counts = Counter(r["outcome"] for r in keywords)
    nk, nq = len(keywords), len(quiet)
    hits = counts["hit"]
    clean_hits = sum(int(r["clean_single_hit"]) for r in keywords)
    quiet_fa = sum(r["outcome"] == "quiet_false_event" for r in quiet)
    required = config.required_consecutive
    raw_support = Counter(
        "no_correct_window" if r["raw_correct_windows"] == 0
        else "fewer_than_required" if r["raw_correct_windows"] < required
        else "enough_but_fragmented" if r["raw_max_correct_streak"] < required
        else "required_consecutive"
        for r in keywords
    )

    def histogram(field: str) -> Dict[str, int]:
        values = Counter(r[field] for r in keywords)
        ordered = sorted(values, key=lambda x: (-1 if x is None else x))
        return {
            ("unavailable" if value is None else str(value)): values[value]
            for value in ordered
        }

    summary = {
        "schema_version": 2,
        "margin": margin,
        "required_consecutive": config.required_consecutive,
        "cooldown_windows": config.cooldown_windows,
        "quiet_classes": list(config.quiet_classes),
        "keyword_cases": nk,
        "keyword_outcomes": {name: counts[name] for name in (
            "hit", "no_correct_window", "correct_but_not_consecutive",
            "margin_broke_correct_streak", "blocked_after_detection",
        )},
        "keyword_event_recall": hits / nk if nk else None,
        "clean_single_hit_cases": clean_hits,
        "clean_single_hit_fraction": clean_hits / nk if nk else None,
        "raw_any_correct_window_cases": sum(
            r["raw_max_correct_streak"] > 0 for r in keywords
        ),
        "raw_support_partition": {
            name: raw_support[name] for name in (
                "no_correct_window", "fewer_than_required",
                "enough_but_fragmented", "required_consecutive",
            )
        },
        "raw_correct_windows_histogram": histogram("raw_correct_windows"),
        "raw_max_correct_streak_histogram": histogram("raw_max_correct_streak"),
        "raw_min_span_for_required_histogram": histogram(
            "raw_min_span_for_required"
        ),
        "gated_correct_windows_histogram": histogram("gated_correct_windows"),
        "gated_max_correct_streak_histogram": histogram(
            "gated_max_correct_streak"
        ),
        "gated_min_span_for_required_histogram": histogram(
            "gated_min_span_for_required"
        ),
        "quiet_cases": nq,
        "quiet_false_alarm_streams": quiet_fa,
        "quiet_stream_false_alarm_fraction": quiet_fa / nq if nq else None,
        "notes": [
            "Outcome categories are an ordered partition, not causal attribution.",
            "Any-correct-window coverage uses ground truth; it is not achievable recall.",
            "Clean hit means exactly one correct event and no other events in the case.",
            "Target overlap refers to the inserted clip, not annotated word boundaries.",
            "Minimum span is diagnostic M-of-K geometry, not achievable recall.",
            "Rejected keywords become quiet and may re-arm the existing vote policy.",
        ],
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=1.25)
    ap.add_argument("--required-consecutive", type=int, default=5)
    ap.add_argument("--cooldown-windows", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None,
                    help="output prefix; existing trace is never overwritten")
    args = ap.parse_args()
    cases = read_trace(args.trace)
    if any(len(w.logits) != 12 for c in cases for w in c.windows):
        raise ValueError("expected 12 classes, with silence=10 and unknown=11")
    config = VoteConfig(args.required_consecutive, args.cooldown_windows)
    rows, summary = diagnose_cases(cases, args.margin, config)
    summary["trace"] = str(args.trace)
    summary["trace_sha256"] = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    prefix = args.out or args.trace.with_name(
        args.trace.stem.removesuffix("_windows")
        + f"_diagnosis_n{args.required_consecutive}_m{args.margin:g}"
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_name(prefix.name + "_cases.csv")
    json_path = prefix.with_name(prefix.name + "_summary.json")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"N={args.required_consecutive}; cooldown={args.cooldown_windows}; "
          f"margin={args.margin:g}; keyword cases={summary['keyword_cases']}")
    for outcome, count in summary["keyword_outcomes"].items():
        print(f"{outcome:32s} {count:6d}")
    print(f"clean_single_hit_cases            {summary['clean_single_hit_cases']:6d}")
    print(f"raw_any_correct_window_cases      {summary['raw_any_correct_window_cases']:6d}")
    print("raw correct-window support:")
    for name, count in summary["raw_support_partition"].items():
        print(f"  {name:30s} {count:6d}")
    spans = summary["raw_min_span_for_required_histogram"]
    print(f"raw min span for {args.required_consecutive} correct windows: "
          + ", ".join(f"{span}={count}" for span, count in spans.items()))
    print(f"quiet false streams: {summary['quiet_false_alarm_streams']}"
          f"/{summary['quiet_cases']}")
    print(f"cases: {csv_path}\nsummary: {json_path}")


if __name__ == "__main__":
    main()
