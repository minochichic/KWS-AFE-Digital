from experiments.analyze_streaming_word_overlap import (
    WordWindow,
    analyze,
    coverage_bin,
    longest_run,
)


def _window(case, target, window, fraction, prediction):
    return WordWindow(case, target, window, fraction, prediction)


def test_coverage_bins_keep_full_word_separate():
    assert coverage_bin(0.0) == "0"
    assert coverage_bin(0.5) == "[.50,.75)"
    assert coverage_bin(0.999) == "[.90,1)"
    assert coverage_bin(1.0) == "1"


def test_longest_run_resets_on_window_id_gap():
    windows = (
        _window(0, 2, 0, 1.0, 2),
        _window(0, 2, 1, 1.0, 2),
        _window(0, 2, 3, 1.0, 2),
    )
    assert longest_run(windows, lambda w: w.prediction == w.target) == 2


def test_analyze_separates_geometry_from_classifier_success():
    cases = {
        0: tuple(_window(0, 2, i, 1.0, 2 if i < 4 else 3)
                 for i in range(5)),
        1: tuple(_window(1, 4, i, 0.5, 4) for i in range(5)),
    }
    summary = analyze(cases, required_consecutive=5, thresholds=(0.5, 1.0))

    half, full = summary["case_geometry"]
    assert half["coverage_run_at_least_required"] == 2
    assert half["correct_run_at_least_required"] == 1
    assert full["coverage_run_at_least_required"] == 1
    assert full["correct_run_at_least_required"] == 0
