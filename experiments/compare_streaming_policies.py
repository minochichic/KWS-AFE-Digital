"""Compare causal decision policies on a saved validation window trace.

No AFE or network inference is run. Labels are used only by the scorer, never
by a policy. These synthetic three-second cases are not field measurements.
The existing consecutive vote remains the baseline and production reference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.diagnose_streaming import diagnose_case
from experiments.streaming_window import VoteConfig, vote_sequence
from experiments.sweep_streaming_gate import (
    TraceCase, TraceWindow, evaluate_gate, gate_predictions, read_trace,
)

QUIET = (10, 11)


@dataclass(frozen=True)
class Policy:
    kind: str
    history: int
    required: int = 1
    margin: float = 0.0
    probability: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("consecutive", "mofk", "mean_logits", "mean_probs"):
            raise ValueError("unknown policy kind")
        if not 1 <= self.required <= self.history:
            raise ValueError("require 1 <= required <= history")
        if self.kind == "consecutive" and self.required != self.history:
            raise ValueError("consecutive policy requires N-of-N")
        if self.kind.startswith("mean_") and self.required != 1:
            raise ValueError("mean policies use one thresholded averaged result")
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be finite and non-negative")
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if self.kind == "mean_probs" and self.margin != 0:
            raise ValueError("mean_probs uses a probability threshold, not logit margin")
        if self.kind != "mean_probs" and self.probability != 0:
            raise ValueError("probability threshold is only used by mean_probs")

    @property
    def name(self) -> str:
        if self.kind == "mean_probs":
            return f"mean_probs_k{self.history}_p{self.probability:g}"
        return f"{self.kind}_k{self.history}_n{self.required}_m{self.margin:g}"


def candidate_sequence(
    windows: Sequence[TraceWindow], policy: Policy,
) -> List[Optional[int]]:
    """Use only past/current scores; None means an incomplete averaging window.

    Threshold-rejected keywords become quiet, preserving the baseline gate's
    convention. Smoothing can change re-arm timing, which is part of the policy
    being compared. No extra raw-quiet re-arm rule is introduced here.
    """
    if policy.kind in ("consecutive", "mofk"):
        return list(gate_predictions(windows, policy.margin))
    history: deque = deque(maxlen=policy.history)
    candidates: List[Optional[int]] = []
    previous = None
    for window in windows:
        if previous is not None and window.window_id != previous + 1:
            history.clear()
        previous = window.window_id
        scores = np.asarray(window.logits, dtype=np.float64)
        if scores.shape != (12,) or not np.isfinite(scores).all():
            raise ValueError("expected 12 finite logits per window")
        if policy.kind == "mean_probs":
            # Softmax separately for each window, then average probabilities.
            scores = np.exp(scores - scores.max())
            scores /= scores.sum()
        history.append(scores)
        if len(history) != policy.history:
            candidates.append(None)
            continue
        averaged = np.mean(history, axis=0)
        label = int(np.argmax(averaged))  # deterministic lowest-index tie break
        if label not in QUIET:
            accepted = (
                averaged[label] >= policy.probability
                if policy.kind == "mean_probs"
                else averaged[label] - max(averaged[i] for i in QUIET) >= policy.margin
            )
            if not accepted:
                label = QUIET[0]
        candidates.append(label)
    return candidates


def confirm_candidates(
    candidates: Sequence[Optional[int]], window_ids: Sequence[int],
    history_size: int, required: int, cooldown: int,
) -> List[Optional[int]]:
    """Confirm M-of-K with the baseline's quiet-plus-cooldown event latch.

    Require a full, adjacent K-window history and current support for the
    candidate. History is discarded while locked and after an event; evidence
    cannot be reused to emit another event. None cannot re-arm the detector.
    """
    if len(candidates) != len(window_ids):
        raise ValueError("candidate and ID lengths differ")
    if not 1 <= required <= history_size or cooldown < 0:
        raise ValueError("invalid confirmation or cooldown")
    history: deque = deque(maxlen=history_size)
    armed, quiet_seen = True, True
    remaining = 0
    previous = None
    events: List[Optional[int]] = []
    for label, window_id in zip(candidates, window_ids):
        if window_id < 0 or (previous is not None and window_id <= previous):
            raise ValueError("window IDs must be non-negative and strictly increasing")
        if label is not None and not 0 <= label < 12:
            raise ValueError("class outside [0, 12)")
        if previous is not None:
            elapsed = window_id - previous
            remaining = max(0, remaining - elapsed)
            if elapsed != 1:
                history.clear()
        previous = window_id
        if not armed and label in QUIET:
            quiet_seen = True
        if not armed and quiet_seen and remaining == 0:
            armed = True
        event = None
        if not armed or label is None:
            history.clear()
        else:
            history.append(label)
            if (label not in QUIET and len(history) == history_size
                    and history.count(label) >= required):
                event = label
                history.clear()
                armed, quiet_seen, remaining = False, False, cooldown
        events.append(event)
    return events


def replay(
    windows: Sequence[TraceWindow], policy: Policy, cooldown: int,
) -> List[Optional[int]]:
    candidates = candidate_sequence(windows, policy)
    ids = [w.window_id for w in windows]
    if policy.kind == "consecutive":
        config = VoteConfig(policy.required, cooldown)
        return [step.detection for step in vote_sequence(candidates, ids, config)]
    return confirm_candidates(
        candidates, ids, policy.history if policy.kind == "mofk" else 1,
        policy.required, cooldown,
    )


def validate_cases(cases: Sequence[TraceCase]) -> None:
    if not cases or len({c.case_id for c in cases}) != len(cases):
        raise ValueError("expected non-empty cases with unique case IDs")
    for case in cases:
        if not case.windows or not 0 <= case.target < 12:
            raise ValueError("invalid case target or empty windows")
        previous = None
        for w in case.windows:
            if (w.window_id < 0 or w.end_frame < 0 or w.target_frames < 0
                    or not 0 <= w.prediction < 12 or len(w.logits) != 12
                    or not all(math.isfinite(x) for x in w.logits)):
                raise ValueError(f"invalid window in case {case.case_id}")
            if previous and (w.window_id <= previous.window_id
                             or w.end_frame <= previous.end_frame):
                raise ValueError("window IDs and end frames must increase strictly")
            previous = w


def score_policy(
    cases: Sequence[TraceCase], policy: Policy, cooldown: int,
    target_start: int = 100, frame_ms: float = 10.0,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Event scorer matching sweep_streaming_gate, plus clean and paired cases."""
    if not math.isfinite(frame_ms) or frame_ms <= 0 or target_start < 0:
        raise ValueError("invalid target start or frame period")
    rows: List[Dict[str, object]] = []
    latencies = []
    for case in cases:
        correct = wrong = outside = 0
        first_end = None
        for w, event in zip(case.windows, replay(case.windows, policy, cooldown)):
            if event is None:
                continue
            if w.target_frames == 0:
                outside += 1
            elif case.target not in QUIET and event == case.target:
                correct += 1
                if first_end is None:
                    first_end = w.end_frame
            else:
                wrong += 1
        if first_end is not None:
            latencies.append((first_end - target_start) * frame_ms)
        rows.append(dict(
            case_id=case.case_id, target=case.target, target_name=case.target_name,
            hit=int(correct > 0), correct_events=correct, wrong_events=wrong,
            outside_events=outside, duplicate_correct_events=max(0, correct - 1),
            clean_single_hit=int(correct == 1 and wrong == 0 and outside == 0),
            quiet_false=int(case.target in QUIET and wrong + outside > 0),
        ))
    keywords = [r for r in rows if r["target"] not in QUIET]
    quiet = [r for r in rows if r["target"] in QUIET]
    if not keywords or not quiet:
        raise ValueError("comparison needs both keyword and quiet cases")
    metrics = dict(
        keyword_cases=len(keywords), keyword_event_hits=sum(r["hit"] for r in keywords),
        keyword_event_recall=sum(r["hit"] for r in keywords) / len(keywords),
        clean_single_hit_cases=sum(r["clean_single_hit"] for r in keywords),
        clean_single_hit_fraction=sum(r["clean_single_hit"] for r in keywords) / len(keywords),
        quiet_cases=len(quiet), quiet_false_alarm_streams=sum(r["quiet_false"] for r in quiet),
        quiet_stream_false_alarm_fraction=sum(r["quiet_false"] for r in quiet) / len(quiet),
        wrong_detections_while_target_overlaps=sum(r["wrong_events"] for r in rows),
        detections_outside_target_overlap=sum(r["outside_events"] for r in rows),
        duplicate_correct_detections=sum(r["duplicate_correct_events"] for r in rows),
        median_first_correct_detection_latency_ms=float(np.median(latencies)) if latencies else None,
    )
    for label in QUIET:
        metrics[f"quiet_{label}_false_streams"] = sum(
            r["quiet_false"] for r in quiet if r["target"] == label
        )
    return metrics, rows


def within_budget(metrics: Dict[str, object], baseline: Dict[str, object]) -> bool:
    # Preserve each quiet class's budget as well as the aggregate error counts.
    keys = ("quiet_false_alarm_streams", "quiet_10_false_streams", "quiet_11_false_streams",
            "wrong_detections_while_target_overlaps", "detections_outside_target_overlap",
            "duplicate_correct_detections")
    latency = metrics["median_first_correct_detection_latency_ms"]
    base_latency = baseline["median_first_correct_detection_latency_ms"]
    return (all(metrics[key] <= baseline[key] for key in keys)
            and latency is not None
            and (base_latency is None or latency <= base_latency))


def paired_changes(
    baseline: Sequence[Dict[str, object]], candidate: Sequence[Dict[str, object]],
    outcomes: Dict[int, str],
) -> Dict[str, object]:
    base = {r["case_id"]: r for r in baseline}
    if set(base) != {r["case_id"] for r in candidate}:
        raise ValueError("paired comparison requires identical case IDs")
    gains: Counter = Counter()
    lost = quiet_added = quiet_removed = 0
    for row in candidate:
        before = base[row["case_id"]]
        if row["clean_single_hit"] and not before["clean_single_hit"]:
            gains[outcomes[row["case_id"]]] += 1
        lost += int(before["clean_single_hit"] and not row["clean_single_hit"])
        quiet_added += int(row["quiet_false"] and not before["quiet_false"])
        quiet_removed += int(before["quiet_false"] and not row["quiet_false"])
    return dict(recovered_clean_cases=sum(gains.values()), lost_clean_cases=lost,
                recovered_by_baseline_outcome=dict(gains),
                new_quiet_false_cases=quiet_added, resolved_quiet_false_cases=quiet_removed)


def policy_grid(margins: Sequence[float], probabilities: Sequence[float]) -> List[Policy]:
    policies = []
    for k, m in ((5, 4), (6, 4), (6, 5), (7, 5)):
        policies.extend(Policy("mofk", k, m, margin) for margin in margins)
    for k in (3, 5, 7):
        policies.extend(Policy("mean_logits", k, margin=margin) for margin in margins)
        policies.extend(Policy("mean_probs", k, probability=p) for p in probabilities)
    # Preserve order while avoiding duplicated trials from repeated CLI values.
    return list(dict.fromkeys(policies))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--baseline-n", type=int, default=5)
    ap.add_argument("--baseline-margin", type=float, default=1.25)
    ap.add_argument("--cooldown-windows", type=int, default=10)
    ap.add_argument("--margins", type=float, nargs="+", default=[0, .5, 1, 1.25, 1.5, 2, 2.5, 3])
    ap.add_argument("--probabilities", type=float, nargs="+", default=[.4, .5, .6, .7, .8, .9])
    ap.add_argument("--target-start-frame", type=int, default=100)
    ap.add_argument("--frame-ms", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=None, help="output prefix")
    args = ap.parse_args()
    with args.trace.open(newline="", encoding="utf-8") as f:
        fields = csv.DictReader(f).fieldnames or []
    if {x for x in fields if x.startswith("logit_")} != {f"logit_{i}" for i in range(12)}:
        raise ValueError("trace must contain exactly logit_0 through logit_11")
    sidecar_path = args.trace.with_name(args.trace.stem.removesuffix("_windows") + "_summary.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.is_file() else {}
    if sidecar.get("split", "val") != "val":
        raise ValueError("policy sweep is for validation only; do not tune on test")
    if sidecar and sidecar.get("target_interval_frames", [args.target_start_frame])[0] != args.target_start_frame:
        raise ValueError("target-start-frame differs from the trace sidecar")
    cases = read_trace(args.trace)
    validate_cases(cases)
    baseline = Policy("consecutive", args.baseline_n, args.baseline_n, args.baseline_margin)
    base_metrics, base_rows = score_policy(cases, baseline, args.cooldown_windows,
                                         args.target_start_frame, args.frame_ms)
    reference = evaluate_gate(cases, args.baseline_margin, args.baseline_n,
                              args.cooldown_windows, target_start_frame=args.target_start_frame,
                              frame_ms=args.frame_ms)
    for key in base_metrics.keys() & reference.keys():
        if base_metrics[key] != reference[key]:
            raise RuntimeError(f"baseline scorer mismatch: {key}")
    config = VoteConfig(args.baseline_n, args.cooldown_windows)
    outcomes = {c.case_id: diagnose_case(c, args.baseline_margin, config)["outcome"] for c in cases}
    print(f"trace: {args.trace}; cases: {len(cases)}; baseline verified against evaluate_gate")
    print(f"baseline: {baseline.name}; cooldown={args.cooldown_windows}")
    print("Budgets: no increase in quiet false streams (also per class), wrong/outside/duplicate events,")
    print("or median detection latency. Ranking favors clean single hits, then any-hit recall.")
    trials = [dict(policy=baseline.name, **asdict(baseline), **base_metrics, within_budget=True)]
    best = {}
    best_rows = {}
    for policy in policy_grid(args.margins, args.probabilities):
        metrics, rows = score_policy(cases, policy, args.cooldown_windows,
                                     args.target_start_frame, args.frame_ms)
        eligible = within_budget(metrics, base_metrics)
        trial = dict(policy=policy.name, **asdict(policy), **metrics, within_budget=eligible)
        trials.append(trial)
        if not eligible:
            continue
        def rank(item: Dict[str, object]) -> tuple:
            return (item["clean_single_hit_cases"], item["keyword_event_hits"],
                    -item["quiet_false_alarm_streams"], -item["wrong_detections_while_target_overlaps"],
                    -item["median_first_correct_detection_latency_ms"])
        if policy.kind not in best or rank(trial) > rank(best[policy.kind]):
            best[policy.kind], best_rows[policy.kind] = trial, rows
    print(f"{'policy':36s} {'recall':>7} {'clean':>7} {'quiet':>7} {'wrong':>6} {'dup':>4} {'delay':>7}")
    for trial in [trials[0]] + list(best.values()):
        print(f"{trial['policy']:36s} {trial['keyword_event_recall']:7.4f} "
              f"{trial['clean_single_hit_fraction']:7.4f} {trial['quiet_false_alarm_streams']:7d} "
              f"{trial['wrong_detections_while_target_overlaps']:6d} "
              f"{trial['duplicate_correct_detections']:4d} "
              f"{str(trial['median_first_correct_detection_latency_ms']):>7}")
    for kind in ("mofk", "mean_logits", "mean_probs"):
        if kind not in best:
            print(f"{kind}: no candidate within the baseline budgets")
    paired = {}
    detail_rows = []
    for kind, rows in best_rows.items():
        paired[kind] = paired_changes(base_rows, rows, outcomes)
        print(f"{kind} paired changes: {json.dumps(paired[kind], sort_keys=True)}")
        base_by_id = {r["case_id"]: r for r in base_rows}
        for row in rows:
            detail_rows.append(dict(policy=best[kind]["policy"], **row,
                                    baseline_outcome=outcomes[row["case_id"]],
                                    baseline_clean_single_hit=base_by_id[row["case_id"]]["clean_single_hit"]))
    prefix = args.out or args.trace.with_name(args.trace.stem.removesuffix("_windows") + "_policy_compare")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_name(prefix.name + "_sweep.csv")
    json_path = prefix.with_name(prefix.name + "_summary.json")
    # All cases are retained even when no alternative satisfies the budgets.
    if not detail_rows:
        detail_rows = [dict(policy=baseline.name, **r, baseline_outcome=outcomes[r["case_id"]],
                            baseline_clean_single_hit=r["clean_single_hit"]) for r in base_rows]
    detail_path = prefix.with_name(prefix.name + "_best_cases.csv")
    write_csv(csv_path, trials)
    write_csv(detail_path, detail_rows)
    summary = dict(
        schema_version=1, trace=str(args.trace),
        trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        source_metadata=sidecar, cooldown_windows=args.cooldown_windows,
        target_start_frame=args.target_start_frame, frame_ms=args.frame_ms,
        trial_count=len(trials), baseline=trials[0], best_within_budget=best, paired_changes=paired,
        notes=["Validation-selected candidates; improvement is not a held-out test result.",
               "Cases are synthetic spliced streams; quiet false fraction is not false alarms/hour.",
               "Latency is from inserted clip start to window end, not word end or FPGA completion.",
               "All temporal means are causal and need K adjacent windows; gaps reset history.",
               "Threshold rejection becomes quiet; averaging may alter re-arm timing.",
               "Logit means and probability means have different thresholds and are separate policies.",
               "Recorded float logits are rounded; repeat chosen policy on fixed-point scores before RTL.",
               "A best-in-family candidate need not beat the baseline; inspect paired gains AND losses."],
    )
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"sweep: {csv_path}\nsummary: {json_path}\ncases: {detail_path}")


if __name__ == "__main__":
    main()
