"""Relate streaming-window predictions to actual keyword-span coverage.

The input trace must come from ``eval_streaming`` schema 2 or newer.  Keyword
boundaries are the 10 ms RMS spans recorded there; they are diagnostic
boundaries rather than human phoneme annotations.  This script runs entirely
from the CSV and does not load the AFE, model, checkpoint, or GPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class WordWindow:
    case_id: int
    target: int
    window_id: int
    word_fraction: float
    prediction: int


BIN_LABELS = (
    "0",
    "(0,.25)",
    "[.25,.50)",
    "[.50,.75)",
    "[.75,.90)",
    "[.90,1)",
    "1",
)


def coverage_bin(value: float) -> str:
    if value <= 0.0:
        return "0"
    if value < 0.25:
        return "(0,.25)"
    if value < 0.50:
        return "[.25,.50)"
    if value < 0.75:
        return "[.50,.75)"
    if value < 0.90:
        return "[.75,.90)"
    if value < 1.0 - 1e-6:
        return "[.90,1)"
    return "1"


def longest_run(windows: Sequence[WordWindow], predicate) -> int:
    longest = current = 0
    previous = None
    for window in windows:
        adjacent = previous is None or window.window_id == previous + 1
        selected = predicate(window)
        if adjacent and selected:
            current += 1
        elif selected:
            current = 1
        else:
            current = 0
        longest = max(longest, current)
        previous = window.window_id
    return longest


def read_word_trace(path: Path) -> Dict[int, Tuple[WordWindow, ...]]:
    grouped: Dict[int, List[WordWindow]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "target", "window_id", "word_fraction",
                    "prediction"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "trace lacks word-overlap columns; rerun eval_streaming: "
                + ", ".join(sorted(missing))
            )
        for row in reader:
            if int(row["target"]) >= 10 or not row["word_fraction"]:
                continue
            fraction = float(row["word_fraction"])
            if not 0.0 <= fraction <= 1.0 + 1e-6:
                raise ValueError(f"invalid word_fraction {fraction}")
            window = WordWindow(
                case_id=int(row["case_id"]),
                target=int(row["target"]),
                window_id=int(row["window_id"]),
                word_fraction=min(fraction, 1.0),
                prediction=int(row["prediction"]),
            )
            grouped[window.case_id].append(window)
    if not grouped:
        raise ValueError("trace contains no keyword windows with word boundaries")
    return {
        case_id: tuple(sorted(windows, key=lambda x: x.window_id))
        for case_id, windows in sorted(grouped.items())
    }


def analyze(
    cases: Dict[int, Tuple[WordWindow, ...]],
    required_consecutive: int = 5,
    thresholds: Sequence[float] = (0.50, 0.75, 0.90, 1.0),
) -> Dict[str, object]:
    if required_consecutive <= 0:
        raise ValueError("required_consecutive must be positive")
    bins = {label: {"windows": 0, "correct": 0, "quiet": 0}
            for label in BIN_LABELS}
    for windows in cases.values():
        target = windows[0].target
        if any(window.target != target for window in windows):
            raise ValueError("target changes within a case")
        for window in windows:
            stats = bins[coverage_bin(window.word_fraction)]
            stats["windows"] += 1
            stats["correct"] += int(window.prediction == target)
            stats["quiet"] += int(window.prediction >= 10)

    geometry = []
    for threshold in thresholds:
        coverage_runs = [
            longest_run(windows, lambda w, t=threshold: w.word_fraction >= t)
            for windows in cases.values()
        ]
        correct_runs = [
            longest_run(
                windows,
                lambda w, t=threshold: (
                    w.word_fraction >= t and w.prediction == w.target
                ),
            )
            for windows in cases.values()
        ]
        geometry.append({
            "minimum_word_fraction": threshold,
            "cases": len(cases),
            "coverage_run_at_least_required": sum(
                run >= required_consecutive for run in coverage_runs
            ),
            "correct_run_at_least_required": sum(
                run >= required_consecutive for run in correct_runs
            ),
            "median_longest_coverage_run": float(median(coverage_runs)),
            "median_longest_correct_run": float(median(correct_runs)),
        })

    serial_bins = []
    for label in BIN_LABELS:
        stats = bins[label]
        count = stats["windows"]
        serial_bins.append({
            "word_fraction_bin": label,
            **stats,
            "accuracy": stats["correct"] / count if count else None,
            "quiet_prediction_fraction": stats["quiet"] / count if count else None,
        })
    return {
        "required_consecutive": required_consecutive,
        "keyword_cases": len(cases),
        "coverage_bins": serial_bins,
        "case_geometry": geometry,
        "notes": [
            "RMS word boundaries are diagnostics, not phoneme annotations.",
            "Partial-word windows are not automatically valid keyword labels.",
            "Coverage opportunity uses no prediction or ground-truth event policy.",
        ],
    }


def print_summary(summary: Dict[str, object]) -> None:
    print(f"keyword cases: {summary['keyword_cases']}; "
          f"required consecutive: {summary['required_consecutive']}")
    print("\nword coverage     windows   correct  accuracy   quiet")
    for row in summary["coverage_bins"]:
        accuracy = "n/a" if row["accuracy"] is None else f"{row['accuracy']:.4f}"
        quiet = ("n/a" if row["quiet_prediction_fraction"] is None
                 else f"{row['quiet_prediction_fraction']:.4f}")
        print(f"{row['word_fraction_bin']:>13} {row['windows']:>9} "
              f"{row['correct']:>9} {accuracy:>9} {quiet:>8}")

    print("\nmin coverage   geometry N-run   correct N-run   median geometry/correct")
    for row in summary["case_geometry"]:
        print(f"{row['minimum_word_fraction']:>11.0%} "
              f"{row['coverage_run_at_least_required']:>8}/"
              f"{row['cases']:<8} "
              f"{row['correct_run_at_least_required']:>8}/"
              f"{row['cases']:<8} "
              f"{row['median_longest_coverage_run']:.1f}/"
              f"{row['median_longest_correct_run']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--required-consecutive", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None,
                        help="summary JSON; defaults beside the trace")
    args = parser.parse_args()

    cases = read_word_trace(args.trace)
    summary = analyze(cases, args.required_consecutive)
    summary["trace"] = str(args.trace)
    summary["trace_sha256"] = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    output = args.out or args.trace.with_name(
        args.trace.stem.removesuffix("_windows") + "_word_overlap.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_summary(summary)
    print(f"\nsummary: {output}")


if __name__ == "__main__":
    main()
