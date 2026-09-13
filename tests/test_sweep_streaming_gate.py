from experiments.sweep_streaming_gate import (
    TraceCase,
    TraceWindow,
    evaluate_gate,
    keyword_quiet_margin,
)


def _window(window_id, prediction, keyword_logit, target_frames=100):
    logits = [0.0] * 12
    logits[10] = 0.8
    logits[11] = 0.7
    logits[prediction] = keyword_logit
    return TraceWindow(
        window_id=window_id,
        end_frame=110 + 10 * window_id,
        target_frames=target_frames,
        prediction=prediction,
        logits=tuple(logits),
    )


def test_keyword_quiet_margin_uses_strongest_quiet_class():
    window = _window(0, prediction=2, keyword_logit=1.05)
    assert abs(keyword_quiet_margin(window, (10, 11)) - 0.25) < 1e-12


def test_margin_trades_keyword_hit_for_quiet_false_alarm():
    keyword = TraceCase(
        case_id=0,
        target=2,
        target_name="up",
        windows=tuple(_window(i, 2, 1.0) for i in range(2)),
    )
    quiet = TraceCase(
        case_id=1,
        target=10,
        target_name="_silence_",
        windows=tuple(_window(i, 2, 0.95) for i in range(2)),
    )

    open_gate = evaluate_gate([keyword, quiet], 0.0, 2, 10)
    strict_gate = evaluate_gate([keyword, quiet], 0.25, 2, 10)

    assert open_gate["keyword_event_recall"] == 1.0
    assert open_gate["quiet_stream_false_alarm_fraction"] == 1.0
    assert strict_gate["keyword_event_recall"] == 0.0
    assert strict_gate["quiet_stream_false_alarm_fraction"] == 0.0


def test_zero_margin_preserves_recorded_keyword_prediction():
    # The stored logits are rounded, so a near-tie can look inconsistent with
    # the full-precision prediction recorded by eval_streaming.
    inconsistent = TraceCase(
        case_id=0,
        target=2,
        target_name="up",
        windows=tuple(_window(i, 2, 0.79) for i in range(2)),
    )
    result = evaluate_gate([inconsistent], 0.0, 2, 10)
    assert result["keyword_event_recall"] == 1.0
