"""board_thresholds reads swing_board.md by parsing, not copying -- so a
rewrite of that document breaks it, and the place to find out is here, not on
the training box (it happened: 4f35691 renamed the table and the tool stopped
with "절이 없다")."""
import pytest

from experiments.board_thresholds import board_rows, measured


@pytest.mark.parametrize("mic", [0.1, 0.4])
def test_swing_table_parses_all_16_channels(mic):
    dc, sw, upper = measured(mic)
    assert dc == pytest.approx(938.10)
    assert sorted(sw) == list(range(16)) and all(v > 0 for v in sw.values())
    assert upper <= set(range(16))


def test_known_rows_of_the_swing_table():
    _, sw01, up01 = measured(0.1)
    _, sw04, up04 = measured(0.4)
    assert (sw01[14], sw01[15]) == (2.75, 1.65)
    assert (sw04[14], sw04[15]) == (27.10, 15.99)
    assert up01 == {1, 2, 9} and up04 == {0, 3, 4, 12}     # the `≤` upper bounds


def test_unknown_mic_column_is_refused():
    with pytest.raises(SystemExit):
        measured(0.2)


def test_board_bom_has_16_channels():
    _, rows = board_rows()
    assert sorted(rows) == list(range(16))


def test_measured_dc_csv():
    from experiments.board_thresholds import measured_dc
    dc = measured_dc()
    assert sorted(dc) == list(range(16))
    assert dc[10] == (903.9, 870.6) and dc[12] == (892.4, 886.2)


def test_per_channel_targets_use_each_channels_dc_and_swing():
    from experiments.board_thresholds import measured_dc, per_channel_targets
    dc = {c: v[1] for c, v in measured_dc().items()}
    _, sw, _ = measured(0.4)
    thr = [0.0] * 16
    thr[15], thr[0] = 0.5, 1.0
    rows = per_channel_targets(thr, dc, sw)
    assert rows[15]["target"] == pytest.approx(888.1 + 0.5 * 15.99)
    assert rows[0]["target"] == pytest.approx(rows[0]["peak"])        # thr 1.0 = peak
    assert rows[3]["target"] == pytest.approx(dc[3])                  # thr 0 = its own DC


def test_awg_swing_interpolation():
    from experiments.awg_thresholds import swing_at
    assert swing_at(0.0, 2.0, 16.0) == (0.0, "")
    assert swing_at(0.5, 2.0, 16.0) == (1.0, "lin<1mV")             # origin to 1 mV
    assert swing_at(2.5, 2.0, 16.0)[0] == pytest.approx(9.0)         # 1..4 mV segment
    v, how = swing_at(5.0, 2.0, 16.0)
    assert how == "extrap>4mV" and v == pytest.approx(16.0 + 14.0 / 3)
