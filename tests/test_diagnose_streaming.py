"""Small trace scenarios with independently specified event outcomes."""
import unittest

from experiments.diagnose_streaming import (
    correct_support,
    diagnose_case,
    diagnose_cases,
)
from experiments.streaming_window import VoteConfig
from experiments.sweep_streaming_gate import TraceCase, TraceWindow, evaluate_gate


def make_case(predictions, target=2, margins=None, ids=None, overlaps=None):
    margins = [2.0] * len(predictions) if margins is None else margins
    ids = list(range(len(predictions))) if ids is None else ids
    overlaps = [100] * len(predictions) if overlaps is None else overlaps
    windows = []
    for i, (prediction, margin) in enumerate(zip(predictions, margins)):
        logits = [-3.0] * 12
        logits[10], logits[11] = 0.0, -1.0
        logits[prediction] = margin
        windows.append(TraceWindow(ids[i], 110 + ids[i] * 10, overlaps[i],
                                   prediction, tuple(logits)))
    return TraceCase(0, target, str(target), tuple(windows))


class DiagnoseStreamingTests(unittest.TestCase):
    def setUp(self):
        self.config = VoteConfig(required_consecutive=3, cooldown_windows=10)

    def outcome(self, case, margin=1.0):
        return diagnose_case(case, margin, self.config)["outcome"]

    def test_no_correct_window_and_fragmented_are_distinct(self):
        self.assertEqual(self.outcome(make_case([10, 1, 1, 11])), "no_correct_window")
        self.assertEqual(self.outcome(make_case([2, 2, 10, 2])),
                         "correct_but_not_consecutive")

    def test_gap_breaks_correct_streak(self):
        case = make_case([2, 2, 2], ids=[0, 1, 3])
        self.assertEqual(self.outcome(case), "correct_but_not_consecutive")
        self.assertEqual(correct_support(case, [2, 2, 2], 3), (3, 2, None))

    def test_correct_support_separates_count_streak_and_minimum_span(self):
        case = make_case([2, 10, 2, 10, 2, 2])
        predictions = [2, 10, 2, 10, 2, 2]
        self.assertEqual(correct_support(case, predictions, 3), (4, 2, 4))
        row = diagnose_case(case, 1.0, self.config)
        self.assertEqual(row["raw_correct_windows"], 4)
        self.assertEqual(row["raw_max_correct_streak"], 2)
        self.assertEqual(row["raw_min_span_for_required"], 4)

    def test_gate_support_is_reported_separately(self):
        case = make_case([2, 2, 2], margins=[2, .5, 2])
        row = diagnose_case(case, 1.0, self.config)
        self.assertEqual((row["raw_correct_windows"],
                          row["raw_max_correct_streak"],
                          row["raw_min_span_for_required"]), (3, 3, 3))
        self.assertEqual((row["gated_correct_windows"],
                          row["gated_max_correct_streak"],
                          row["gated_min_span_for_required"]), (2, 1, None))

    def test_gate_breaks_an_otherwise_sufficient_correct_streak(self):
        self.assertEqual(self.outcome(make_case([2, 2, 2], margins=[2, .5, 2])),
                         "margin_broke_correct_streak")

    def test_early_wrong_event_blocks_later_correct_streak(self):
        row = diagnose_case(make_case([1, 1, 1, 2, 2, 2]), 1.0, self.config)
        self.assertEqual(row["outcome"], "blocked_after_detection")
        self.assertEqual(row["wrong_events"], 1)
        self.assertEqual(row["gated_max_correct_streak"], 3)

    def test_hit_with_wrong_event_is_not_clean(self):
        case = make_case([1, 1, 1] + [10] * 10 + [2, 2, 2])
        row = diagnose_case(case, 1.0, self.config)
        self.assertEqual(row["outcome"], "hit")
        self.assertEqual(row["clean_single_hit"], 0)
        self.assertEqual((row["correct_events"], row["wrong_events"]), (1, 1))

    def test_clean_single_hit_and_duplicate(self):
        row = diagnose_case(make_case([2, 2, 2]), 1.0, self.config)
        self.assertEqual(row["clean_single_hit"], 1)
        duplicate = diagnose_case(make_case([2] * 3 + [10] * 10 + [2] * 3),
                                  1.0, self.config)
        self.assertEqual(duplicate["correct_events"], 2)
        self.assertEqual(duplicate["clean_single_hit"], 0)

    def test_correct_class_outside_target_is_not_a_hit(self):
        row = diagnose_case(make_case([2, 2, 2], overlaps=[0, 0, 0]),
                            1.0, self.config)
        self.assertEqual(row["outcome"], "no_correct_window")
        self.assertEqual(row["outside_events"], 1)

    def test_gated_rejection_can_rearm_existing_policy(self):
        config = VoteConfig(required_consecutive=2, cooldown_windows=0)
        case = make_case([2] * 5, margins=[2, 2, .1, 2, 2])
        self.assertEqual(diagnose_case(case, 0.0, config)["correct_events"], 1)
        self.assertEqual(diagnose_case(case, 1.0, config)["correct_events"], 2)

    def test_partition_and_counts_match_sweep(self):
        cases = [
            make_case([10, 11]), make_case([2, 2, 10]),
            make_case([2, 2, 2], margins=[2, .5, 2]),
            make_case([1, 1, 1, 2, 2, 2]), make_case([2, 2, 2]),
            make_case([10, 10], target=10), make_case([1, 1, 1], target=11),
        ]
        rows, summary = diagnose_cases(cases, 1.0, self.config)
        self.assertEqual(sum(summary["keyword_outcomes"].values()), 5)
        self.assertTrue(all(n == 1 for n in summary["keyword_outcomes"].values()))
        reference = evaluate_gate(cases, 1.0, 3, 10)
        self.assertEqual(summary["keyword_event_recall"], .2)
        self.assertEqual(summary["quiet_stream_false_alarm_fraction"], .5)
        for key in ("keyword_event_recall", "quiet_false_alarm_streams"):
            self.assertEqual(summary[key], reference[key])
        self.assertEqual(sum(r["wrong_events"] for r in rows),
                         reference["wrong_detections_while_target_overlaps"])
        self.assertEqual(sum(summary["raw_support_partition"].values()), 5)
        self.assertEqual(summary["schema_version"], 2)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            diagnose_cases([], 1.0, self.config)
        with self.assertRaises(ValueError):
            diagnose_case(make_case([2]), float("nan"), self.config)
        with self.assertRaises(ValueError):
            correct_support(make_case([2]), [2], 0)


if __name__ == "__main__":
    unittest.main()
