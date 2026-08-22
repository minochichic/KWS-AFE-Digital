"""The confusion reading, checked on matrices whose answer is known by hand.

The point of these is that experiments/fixed_accuracy.py cannot run without a
checkpoint and a dataset, so its diagnostics would otherwise be unverified --
and a diagnostic that is silently wrong aims the next experiment at the wrong
problem, which costs far more than the bug.
"""
from __future__ import annotations

from experiments.confusion import (class_stats, false_wakes, pairs, payoff,
                                   report)

NAMES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop",
         "go", "_silence_", "_unknown_"]


def eye(n: int, diag: int = 100) -> list[list[int]]:
    return [[diag if i == j else 0 for j in range(n)] for i in range(n)]


def test_a_perfect_matrix_is_perfect_on_both_axes() -> None:
    m = eye(len(NAMES))
    for s in class_stats(m, NAMES):
        assert s.recall == 1.0 and s.precision == 1.0
        assert not s.absorbing
    assert pairs(m) == []
    assert false_wakes(m, NAMES)[0] == 0
    assert payoff(m, NAMES) == []


def test_recall_is_the_row_and_precision_is_the_column() -> None:
    """The two are different numbers, and swapping them inverts the diagnosis.

    `down` here gets every one of its own AND twenty from `no`, so it is a
    perfect-recall, poor-precision class -- the absorbing shape.
    """
    m = eye(len(NAMES))
    m[1][1] -= 20
    m[1][3] += 20                                   # no -> down, one way
    by = {s.name: s for s in class_stats(m, NAMES)}

    assert by["no"].recall == 80 / 100
    assert by["no"].precision == 1.0                # nothing wrongly called no
    assert by["down"].recall == 1.0                 # down loses nothing
    assert by["down"].precision == 100 / 120        # but takes 20 it should not
    assert by["down"].absorbing
    assert not by["no"].absorbing


def test_a_class_that_is_merely_weak_is_not_called_absorbing() -> None:
    """Losing clips in all directions drops precision too -- but that is the
    ordinary weak case and must not raise the absorbing flag."""
    m = eye(len(NAMES))
    m[1][1] -= 22
    for p in (0, 2, 4, 6, 8, 9, 11):
        m[1][p] += 2                                # no leaks everywhere
        m[p][1] += 2                                # and takes some back
    by = {s.name: s for s in class_stats(m, NAMES)}
    assert by["no"].recall < 1.0
    assert not by["no"].absorbing


def test_mutual_and_one_way_confusion_are_told_apart() -> None:
    m = eye(len(NAMES))
    m[9][9] -= 34; m[9][1] = 34                     # go -> no
    m[1][1] -= 26; m[1][9] = 26                     # no -> go, nearly as much
    m[2][2] -= 30; m[2][7] = 30                     # up -> off, nothing back

    found = {(min(NAMES[p.lo], NAMES[p.hi]),
              max(NAMES[p.lo], NAMES[p.hi])): p for p in pairs(m)}
    go_no = found[("go", "no")]
    up_off = found[("off", "up")]

    assert go_no.total == 60 and go_no.mutual
    assert up_off.total == 30 and not up_off.mutual
    assert NAMES[up_off.preferred()] == "off"


def test_pairs_are_ordered_by_total_not_by_one_direction() -> None:
    m = eye(len(NAMES))
    m[0][4] = 40                                    # 40 one way
    m[2][7] = 25; m[7][2] = 25                      # 50 across two directions
    first = pairs(m)[0]
    assert {NAMES[first.lo], NAMES[first.hi]} == {"up", "off"}


def test_symmetry_of_a_one_sided_pair_is_zero_not_a_crash() -> None:
    m = eye(len(NAMES))
    m[0][4] = 7
    p = pairs(m)[0]
    assert p.symmetry == 0.0 and not p.mutual


def test_false_wakes_count_only_non_keyword_scored_as_keyword() -> None:
    m = eye(len(NAMES))
    m[11][5] = 25                                   # unknown -> right: a wake
    m[10][0] = 5                                    # silence -> yes:   a wake
    m[11][10] = 9                                   # unknown -> silence: not
    m[1][11] = 40                                   # no -> unknown: a MISS
    woke, total = false_wakes(m, NAMES)
    assert woke == 30
    assert total == sum(m[10]) + sum(m[11])


def test_payoff_scales_with_class_size_not_with_the_gap() -> None:
    """Two classes the same distance below target pay out in proportion to how
    many clips they hold -- that is the whole reason to rank them."""
    m = eye(len(NAMES), diag=100)
    m[1] = [0] * len(NAMES); m[1][1] = 80; m[1][3] = 20      # 100 clips, 0.80
    m[2] = [0] * len(NAMES); m[2][2] = 160; m[2][7] = 40     # 200 clips, 0.80
    got = dict((NAMES[c], g) for g, c in payoff(m, NAMES, target=0.85))
    assert got["up"] > got["no"]
    assert abs(got["up"] / got["no"] - 2.0) < 1e-9


def test_report_names_the_real_finding_on_the_measured_shape() -> None:
    """The xl_g12 shape: the named confusions ON TOP OF a diffuse background.

    The background matters. A matrix carrying only the confusions we name sits
    at 96% and the report says we are above target -- which is how the first
    version of this fixture was wrong. The measured split is 84.45%, so most of
    the errors are spread thin, and the named pairs have to stand out FROM that
    rather than be all of it.
    """
    sizes = [419, 405, 425, 406, 412, 396, 396, 402, 411, 402, 407, 407]
    named = {(9, 1): 34, (1, 9): 26, (1, 3): 28, (9, 3): 24,
             (2, 7): 33, (2, 6): 13, (11, 5): 25, (11, 1): 14}
    n_c = len(NAMES)
    m = [[0] * n_c for _ in range(n_c)]
    for t in range(n_c):
        for (a, b), v in named.items():
            if a == t:
                m[t][b] = v
        # spread ~10% of what is left evenly over the other classes, which is
        # the diffuse floor every class sits on
        rest = sizes[t] - sum(m[t])
        leak = round(rest * 0.12)          # lands the fixture on .8445
        each, extra = divmod(leak, n_c - 1)
        for p in range(n_c):
            if p != t:
                m[t][p] += each + (1 if extra > 0 else 0)
                extra -= 1
        m[t][t] = sizes[t] - sum(m[t])

    for t in range(n_c):
        assert sum(m[t]) == sizes[t]
    total = sum(sizes)
    acc = sum(m[c][c] for c in range(n_c)) / total
    assert 0.840 < acc < 0.850, f"fixture at {acc:.4f}, not the measured .8445"

    out = report(m, NAMES)
    assert "mutual" in out                          # the go/no pair
    assert "down" in out and "absorbing" in out     # down takes from both
    assert "false wakes" in out
    assert "short of the 0.85 target" in out
    # the mutual pair must outrank the diffuse background, or the diagnostic
    # is drowning in noise and says nothing
    head = out.split("most confused pairs")[1].splitlines()[1]
    assert "go" in head and "no" in head and "mutual" in head


def test_the_worst_sink_is_picked_by_precision_not_by_recall() -> None:
    """fixed_accuracy names one class as THE sink and compares it against what
    the network answers for an empty input, so the pick has to be the class
    taking in the most it should not -- lowest precision. Ranking by recall
    would pick whichever absorbing class happens to also be good at its own
    job, which is the opposite of the question.
    """
    m = eye(len(NAMES), diag=300)
    m[1][1] -= 96; m[1][9] += 96                    # no -> go, the big one
    m[6][6] -= 34; m[6][9] += 34                    # on -> go
    m[11][11] -= 38; m[11][9] += 38                 # unknown -> go
    m[0][0] -= 20; m[0][3] += 20                    # a milder sink at down

    sinks = [s for s in class_stats(m, NAMES) if s.absorbing]
    assert {s.name for s in sinks} == {"go", "down"}
    assert min(sinks, key=lambda s: s.precision).name == "go"
