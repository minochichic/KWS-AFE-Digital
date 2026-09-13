"""Causal replay, event lifecycle, and baseline parity for policy experiments."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.compare_streaming_policies import (
    Policy, candidate_sequence, confirm_candidates, main, paired_changes,
    replay, score_policy, validate_cases, within_budget,
)
from experiments.streaming_window import VoteConfig, vote_sequence
from experiments.sweep_streaming_gate import TraceCase, TraceWindow, evaluate_gate


def make_case(predictions, target=2, case_id=0, ids=None, overlaps=None):
    ids = range(len(predictions)) if ids is None else ids
    overlaps = [100] * len(predictions) if overlaps is None else overlaps
    windows = []
    for prediction, window_id, overlap in zip(predictions, ids, overlaps):
        scores = [-3.0] * 12
        scores[10], scores[11] = 0.0, -1.0
        scores[prediction] = 4.0
        windows.append(TraceWindow(window_id, 110 + window_id * 10, overlap,
                                   prediction, tuple(scores)))
    return TraceCase(case_id, target, str(target), tuple(windows))


class CompareStreamingTests(unittest.TestCase):
    def test_mofk_recovers_one_interruption(self):
        case = make_case([2, 2, 10, 2, 2])
        self.assertEqual(replay(case.windows, Policy("consecutive", 5, 5), 10), [None] * 5)
        self.assertEqual(replay(case.windows, Policy("mofk", 5, 4), 10), [None] * 4 + [2])

    def test_mofk_needs_full_history_and_current_support(self):
        self.assertEqual(confirm_candidates([2, 2, 2, 2], [0, 1, 2, 3], 5, 4, 0), [None] * 4)
        self.assertEqual(confirm_candidates([2, 2, 2, 2, 10], list(range(5)), 5, 4, 0), [None] * 5)

    def test_competing_labels_do_not_share_votes(self):
        self.assertEqual(confirm_candidates([1, 2, 1, 2, 2], list(range(5)), 5, 4, 0), [None] * 5)

    def test_gap_discards_evidence(self):
        self.assertEqual(confirm_candidates([2] * 5, [0, 1, 2, 4, 5], 5, 4, 0), [None] * 5)
        case = make_case([2] * 5, ids=[0, 1, 2, 4, 5])
        self.assertEqual(candidate_sequence(case.windows, Policy("mean_logits", 3)),
                         [None, None, 2, None, None])

    def test_cooldown_and_quiet_both_required(self):
        self.assertEqual(confirm_candidates([2] * 12, list(range(12)), 2, 2, 2).count(2), 1)
        events = confirm_candidates([2, 2, 10, 2, 2, 2], list(range(6)), 2, 2, 3)
        self.assertEqual(events, [None, 2, None, None, None, 2])

    def test_unknown_rearms_but_unavailable_score_does_not(self):
        self.assertEqual(confirm_candidates([2, None, 2], [0, 1, 2], 1, 1, 0), [2, None, None])
        self.assertEqual(confirm_candidates([2, 11, 2], [0, 1, 2], 1, 1, 0), [2, None, 2])

    def test_post_detection_history_is_not_reused(self):
        events = confirm_candidates([2, 2, 10, 2, 2, 10, 2, 2], list(range(8)), 5, 4, 0)
        self.assertEqual(events, [None, None, None, None, 2, None, None, None])

    def test_n_of_n_matches_existing_state_machine(self):
        rng = np.random.default_rng(22)
        for n in (1, 2, 5):
            for cooldown in (0, 3, 10):
                predictions = rng.choice([1, 2, 2, 2, 10, 11], size=500).tolist()
                ids = np.cumsum(rng.choice([1, 1, 1, 2], size=500)).tolist()
                expected = [s.detection for s in vote_sequence(predictions, ids, VoteConfig(n, cooldown))]
                self.assertEqual(confirm_candidates(predictions, ids, n, n, cooldown), expected)

    def test_means_are_causal(self):
        windows = make_case([2, 1, 2, 2, 1, 1, 1]).windows
        for policy in (Policy("mean_logits", 3), Policy("mean_probs", 3, probability=.4)):
            full = candidate_sequence(windows, policy)
            for end in range(1, len(windows) + 1):
                self.assertEqual(candidate_sequence(windows[:end], policy), full[:end])
                self.assertEqual(replay(windows[:end], policy, 2), replay(windows, policy, 2)[:end])

    def test_means_can_use_second_place_evidence(self):
        windows = []
        for i, rival in enumerate((1, 3, 4)):
            scores = [0.0] * 12
            scores[2], scores[rival] = 3.0, 4.0
            windows.append(TraceWindow(i, 110 + 10 * i, 100, rival, tuple(scores)))
        self.assertEqual(candidate_sequence(windows, Policy("mean_logits", 3)), [None, None, 2])

    def test_probability_mean_differs_from_logit_mean(self):
        windows = []
        for i, strength in enumerate((10, -2, -2)):
            scores = [-100.0] * 12
            scores[2], scores[3] = strength, 0.0
            windows.append(TraceWindow(i, 110 + 10 * i, 100, 2 if strength > 0 else 3, tuple(scores)))
        self.assertEqual(candidate_sequence(windows, Policy("mean_logits", 3))[-1], 2)
        self.assertEqual(candidate_sequence(windows, Policy("mean_probs", 3, probability=.5))[-1], 3)

    def test_softmax_is_stable_and_invariant_to_common_offset(self):
        windows = make_case([2, 2, 2]).windows
        shifted = [replace(w, logits=tuple(x + 10000 for x in w.logits)) for w in windows]
        policy = Policy("mean_probs", 3, probability=.8)
        self.assertEqual(candidate_sequence(windows, policy), candidate_sequence(shifted, policy))
        self.assertEqual(candidate_sequence(shifted, policy)[-1], 2)

    def test_probability_threshold_and_quiet_winner(self):
        windows = make_case([2, 2, 10]).windows
        self.assertEqual(candidate_sequence(windows, Policy("mean_probs", 3, probability=.9))[-1], 10)
        self.assertEqual(candidate_sequence(make_case([11] * 3).windows,
                                            Policy("mean_probs", 3, probability=.4))[-1], 11)

    def test_labels_never_change_policy_events(self):
        a = make_case([2, 2, 10, 2, 2])
        b = replace(a, target=10, target_name="silence",
                    windows=tuple(replace(w, target_frames=0) for w in a.windows))
        for policy in (Policy("mofk", 5, 4), Policy("mean_logits", 3),
                       Policy("mean_probs", 3, probability=.5)):
            self.assertEqual(replay(a.windows, policy, 10), replay(b.windows, policy, 10))

    def test_scorer_matches_legacy_and_counts_dirty_hits(self):
        cases = [
            make_case([2, 2, 10, 2, 2], case_id=0),  # two correct events
            make_case([1, 1, 10, 2, 2], case_id=1),  # wrong then correct
            make_case([2, 2], case_id=2, overlaps=[0, 0]),  # outside only
            make_case([10, 10], case_id=3, target=10),
            make_case([1, 1], case_id=4, target=11),
        ]
        metrics, rows = score_policy(cases, Policy("consecutive", 2, 2, 1.25), 0)
        reference = evaluate_gate(cases, 1.25, 2, 0)
        for key in metrics.keys() & reference.keys():
            self.assertEqual(metrics[key], reference[key], key)
        self.assertEqual(metrics["keyword_event_hits"], 2)
        self.assertEqual(metrics["clean_single_hit_cases"], 0)
        self.assertEqual(metrics["duplicate_correct_detections"], 1)
        self.assertEqual(metrics["wrong_detections_while_target_overlaps"], 2)
        self.assertEqual(metrics["detections_outside_target_overlap"], 1)

    def test_budget_rejects_per_class_regression_and_delayed_events(self):
        cases = [make_case([2, 2]), make_case([1, 1], 10, 1), make_case([11, 11], 11, 2)]
        baseline, _ = score_policy(cases, Policy("consecutive", 2, 2), 0)
        self.assertTrue(within_budget(baseline, baseline))
        shifted = dict(baseline, quiet_10_false_streams=0, quiet_11_false_streams=1)
        self.assertFalse(within_budget(shifted, baseline))
        slower = dict(baseline, median_first_correct_detection_latency_ms=9999)
        self.assertFalse(within_budget(slower, baseline))

    def test_paired_changes_reports_losses_as_well_as_gains(self):
        baseline = [dict(case_id=0, clean_single_hit=0, quiet_false=0),
                    dict(case_id=1, clean_single_hit=1, quiet_false=0),
                    dict(case_id=2, clean_single_hit=0, quiet_false=1)]
        candidate = [dict(case_id=2, clean_single_hit=0, quiet_false=0),
                     dict(case_id=1, clean_single_hit=0, quiet_false=0),
                     dict(case_id=0, clean_single_hit=1, quiet_false=0)]
        result = paired_changes(baseline, candidate, {0: "correct_but_not_consecutive", 1: "hit", 2: "quiet"})
        self.assertEqual(result["recovered_clean_cases"], 1)
        self.assertEqual(result["lost_clean_cases"], 1)
        self.assertEqual(result["resolved_quiet_false_cases"], 1)
        self.assertEqual(result["recovered_by_baseline_outcome"], {"correct_but_not_consecutive": 1})

    def test_bad_parameters_and_trace_are_rejected(self):
        for kwargs in (dict(kind="mofk", history=3, required=4),
                       dict(kind="mean_probs", history=3, probability=2),
                       dict(kind="mean_logits", history=3, margin=float("nan"))):
            with self.assertRaises(ValueError):
                Policy(**kwargs)
        case = make_case([2, 2])
        for cases in ([], [case, case], [make_case([2, 2], ids=[1, 1])],
                      [replace(case, windows=(replace(case.windows[0], logits=(float("nan"),) * 12),))]):
            with self.assertRaises(ValueError):
                validate_cases(cases)
        with self.assertRaises(ValueError):
            confirm_candidates([2, 2], [1, 1], 2, 2, 0)

    def test_cli_writes_reproducible_outputs_and_blocks_test_sweep(self):
        cases = [make_case([2, 2, 10, 2, 2]), make_case([10] * 5, 10, 1),
                 make_case([11] * 5, 11, 2)]
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "tiny_windows.csv"
            with trace.open("w", newline="", encoding="utf-8") as f:
                fields = ["case_id", "target", "target_name", "window_id", "end_frame", "target_frames", "prediction"]
                fields += [f"logit_{i}" for i in range(12)]
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for c in cases:
                    for w in c.windows:
                        writer.writerow(dict(case_id=c.case_id, target=c.target, target_name=c.target_name,
                                             window_id=w.window_id, end_frame=w.end_frame,
                                             target_frames=w.target_frames, prediction=w.prediction,
                                             **{f"logit_{i}": x for i, x in enumerate(w.logits)}))
            argv = ["compare", "--trace", str(trace), "--margins", "1.25", "--probabilities", ".5"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads((Path(temp) / "tiny_policy_compare_summary.json").read_text())
            self.assertEqual(result["trial_count"], 11)
            self.assertEqual(len(result["trace_sha256"]), 64)
            self.assertEqual(result["baseline"]["keyword_event_hits"], 0)
            self.assertEqual(result["best_within_budget"]["mofk"]["keyword_event_hits"], 1)
            self.assertTrue((Path(temp) / "tiny_policy_compare_best_cases.csv").is_file())
            (Path(temp) / "tiny_summary.json").write_text(json.dumps({"split": "test"}))
            with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "validation only"):
                main()


if __name__ == "__main__":
    unittest.main()
