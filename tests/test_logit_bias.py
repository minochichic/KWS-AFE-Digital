"""The per-class offset search, on a problem whose answer is planted.

These need torch but not a checkpoint or the dataset, so they run anywhere the
training environment does. The point is that the search is the only part of the
correction that could be silently wrong: everything else is arithmetic that
export/tailfmt.py already tests. If the search were broken it would still
return twelve numbers and still print a table.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from experiments.logit_bias import accuracy, balanced, fit_delta  # noqa: E402


def planted(n: int = 3000, n_c: int = 4, bias_on: int = 2, bias: float = 5.0,
            seed: int = 0):
    """Logits where the right answer usually wins, but one class cheats."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(n_c, (n,), generator=g)
    logits = torch.randn(n, n_c, generator=g, dtype=torch.double) * 3.0
    logits[torch.arange(n), y] += 6.0            # the truth usually leads
    logits[:, bias_on] += bias                   # and this one is inflated
    return logits, y


def test_the_search_recovers_a_planted_offset() -> None:
    """A constant added to one class should be found and cancelled, up to the
    overall shift the problem is invariant to."""
    logits, y = planted()
    d = fit_delta(logits, y, 4)
    rel = d - d[2]                               # only differences matter
    for c in (0, 1, 3):
        assert 3.5 < float(rel[c]) < 6.5, f"class {c} not near the planted 5.0"


def test_the_search_improves_the_metric_it_is_given() -> None:
    logits, y = planted()
    zero = torch.zeros(4, dtype=torch.double)
    d = fit_delta(logits, y, 4)
    assert accuracy(logits, y, d) > accuracy(logits, y, zero) + 0.10


def test_predicted_counts_move_toward_the_true_counts() -> None:
    """The symptom this exists to fix: one class predicted far too often."""
    logits, y = planted()
    d = fit_delta(logits, y, 4)
    true = torch.bincount(y, minlength=4)
    before = torch.bincount(logits.argmax(1), minlength=4)
    after = torch.bincount((logits + d).argmax(1), minlength=4)
    assert int(before[2]) > 2 * int(true[2]), "fixture did not inflate class 2"
    err0 = float((before - true).abs().sum())
    err1 = float((after - true).abs().sum())
    assert err1 < err0 / 2


def test_a_shift_shared_by_every_class_changes_nothing() -> None:
    """argmax is invariant to it, so the metric must be too -- otherwise the
    search is chasing a direction that cannot matter."""
    logits, y = planted()
    flat = torch.full((4,), 7.5, dtype=torch.double)
    assert accuracy(logits, y, flat) == accuracy(
        logits, y, torch.zeros(4, dtype=torch.double))


def test_an_already_fair_problem_is_left_alone() -> None:
    """No planted bias: the search must not manufacture a correction, or it is
    fitting noise and will not survive the move from val to test."""
    logits, y = planted(bias=0.0)
    zero = torch.zeros(4, dtype=torch.double)
    d = fit_delta(logits, y, 4)
    gain = accuracy(logits, y, d) - accuracy(logits, y, zero)
    assert gain < 0.02, f"invented {gain:.3f} of accuracy from nothing"


def test_balanced_weighs_a_swallowed_small_class_like_a_large_one() -> None:
    """Plain accuracy barely notices a small class disappearing; balanced is
    there for exactly that case, so the difference has to be real."""
    n_c = 3
    y = torch.tensor([0] * 900 + [1] * 90 + [2] * 10)
    logits = torch.zeros(len(y), n_c, dtype=torch.double)
    logits[:, 0] = 1.0                           # everything predicted class 0
    zero = torch.zeros(n_c, dtype=torch.double)
    assert accuracy(logits, y, zero) == pytest.approx(0.9)
    assert balanced(logits, y, zero, n_c) == pytest.approx(1 / 3)


def test_balanced_as_the_objective_lifts_the_small_class() -> None:
    g = torch.Generator().manual_seed(1)
    n_c = 3
    y = torch.cat([torch.zeros(800), torch.ones(150),
                   torch.full((50,), 2.0)]).long()
    logits = torch.randn(len(y), n_c, generator=g, dtype=torch.double)
    logits[torch.arange(len(y)), y] += 1.2       # weak but real signal
    logits[:, 0] += 2.0                          # and a large class cheating
    zero = torch.zeros(n_c, dtype=torch.double)
    d = fit_delta(logits, y, n_c, objective="balanced")
    assert balanced(logits, y, d, n_c) > balanced(logits, y, zero, n_c) + 0.05


# ---- the one-parameter alternative, added after twelve overfitted ----------

def test_prior_recovers_a_planted_offset_too() -> None:
    """With only a strength to fit, the direction still has to be right."""
    from experiments.logit_bias import fit_prior
    logits, y = planted()
    d = fit_prior(logits, y, 4)
    rel = d - d[2]
    for c in (0, 1, 3):
        assert float(rel[c]) > 0, f"class {c} should be pushed above class 2"
    assert accuracy(logits, y, d) > accuracy(
        logits, y, torch.zeros(4, dtype=torch.double)) + 0.10


def test_prior_direction_comes_from_the_count_mismatch() -> None:
    """An over-predicted class goes down, an under-predicted one goes up. That
    ordering is the whole reason this generalises better than a free search."""
    from experiments.logit_bias import fit_prior
    logits, y = planted()
    d = fit_prior(logits, y, 4)
    pred = torch.bincount(logits.argmax(1), minlength=4)
    true = torch.bincount(y, minlength=4)
    over = [c for c in range(4) if pred[c] > true[c]]
    under = [c for c in range(4) if pred[c] < true[c]]
    assert over and under, "fixture must have both directions"
    assert max(float(d[c]) for c in over) < min(float(d[c]) for c in under)


def test_prior_is_a_no_op_when_the_counts_already_match() -> None:
    """No mismatch means no direction, so nothing to travel along -- the guard
    against inventing a correction out of a fair problem."""
    from experiments.logit_bias import fit_prior
    n_c = 3
    y = torch.tensor([0, 1, 2] * 300)
    logits = torch.zeros(len(y), n_c, dtype=torch.double)
    logits[torch.arange(len(y)), y] = 1.0        # every clip already correct
    d = fit_prior(logits, y, n_c)
    assert float(d.abs().max()) == pytest.approx(0.0, abs=1e-9)


def test_split_half_exposes_a_method_that_only_fits_what_it_saw() -> None:
    """Pure noise: nothing to learn, so a free search should show a gain on the
    half it fitted and about nothing on the half it did not."""
    from experiments.logit_bias import fit_delta, fit_prior, halves
    g = torch.Generator().manual_seed(3)
    n_c = 6
    y = torch.randint(n_c, (1200,), generator=g)
    logits = torch.randn(1200, n_c, generator=g, dtype=torch.double)

    seen, held = halves(logits, y, n_c, fit_delta, "accuracy")
    assert seen > 0.02, "twelve-ish free params should fit noise it can see"
    assert held < seen / 2, "and should not carry to the half it cannot"

    seen_p, held_p = halves(logits, y, n_c, fit_prior, "accuracy")
    assert seen_p <= seen, "one parameter cannot fit more noise than many"
