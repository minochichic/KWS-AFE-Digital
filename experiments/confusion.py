"""Reading a confusion matrix, as plain lists so it can be tested anywhere.

experiments/fixed_accuracy.py needs torch and a checkpoint and the dataset, so
it only ever runs on the training box. The ARITHMETIC here needs none of that,
and a wrong index in a diagnostic is worse than no diagnostic: it sends the
next month of work at the wrong problem. So the reading is separated from the
measuring and unit tested (tests/test_confusion.py).

Convention throughout: `m[t][p]` is the number of clips whose true class is `t`
and which were predicted `p`. Rows are truth, columns are predictions.

The three questions this answers, and why each is a different job:

  recall vs precision   A class can look weak because it is hard, or because a
                        NEIGHBOUR is greedy. Low precision on the neighbour
                        says the decision boundary moved, not the acoustics.

  mutual vs one way     `go`->`no` as often as `no`->`go` says the input does
                        not carry what separates the two sounds -- a front end
                        finding. Lopsided says one class is simply preferred --
                        a prior finding.

  false wakes           Accuracy weighs every error the same; an always-on
                        spotter does not. A non-keyword scored as a keyword
                        wakes the device.
"""
from __future__ import annotations

from typing import List, NamedTuple, Sequence

Matrix = Sequence[Sequence[int]]


class ClassStat(NamedTuple):
    idx: int
    name: str
    recall: float
    precision: float
    n_true: int
    n_pred: int

    @property
    def absorbing(self) -> bool:
        """Takes in more than it gets right -- a neighbour's losses land here.

        Guarded on recall because a class that is simply bad at everything has
        low precision too, and that is the ordinary weak case, not this one.
        """
        return self.precision < self.recall - 0.05


class Pair(NamedTuple):
    lo: int
    hi: int
    lo_to_hi: int
    hi_to_lo: int

    @property
    def total(self) -> int:
        return self.lo_to_hi + self.hi_to_lo

    @property
    def symmetry(self) -> float:
        """1.0 = perfectly mutual, 0.0 = entirely one direction."""
        big = max(self.lo_to_hi, self.hi_to_lo)
        return min(self.lo_to_hi, self.hi_to_lo) / big if big else 0.0

    @property
    def kind(self) -> str:
        """"mutual", "leaning" or "one way".

        Two bands were not enough. fx_d0's no/go runs 96 against 49, and a
        single cut at 0.5 called that mutual -- reporting a two-to-one flow as
        "the input cannot separate these sounds" when half of it is one class
        winning. xl_g12's genuinely mutual go/no sits at 0.765 for comparison.

        The middle band is where both readings are true at once: the sounds do
        overlap AND one side takes more, so neither a front end fix nor a
        boundary fix alone accounts for it.
        """
        if self.symmetry > 0.6:
            return "mutual"
        return "leaning" if self.symmetry > 0.3 else "one way"

    @property
    def mutual(self) -> bool:
        return self.kind == "mutual"

    def preferred(self) -> int:
        """The class the confusion flows TOWARD. Meaningless when mutual."""
        return self.hi if self.lo_to_hi > self.hi_to_lo else self.lo


def class_stats(m: Matrix, names: Sequence[str]) -> List[ClassStat]:
    """Recall and precision per class, weakest recall first."""
    out = []
    for c, name in enumerate(names):
        n_true = sum(m[c])
        n_pred = sum(m[t][c] for t in range(len(names)))
        out.append(ClassStat(c, name, m[c][c] / n_true if n_true else 0.0,
                             m[c][c] / n_pred if n_pred else 0.0,
                             n_true, n_pred))
    return sorted(out, key=lambda s: s.recall)


def pairs(m: Matrix) -> List[Pair]:
    """Every pair that confuses at all, most confused first."""
    out = []
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            if m[i][j] or m[j][i]:
                out.append(Pair(i, j, m[i][j], m[j][i]))
    return sorted(out, key=lambda p: p.total, reverse=True)


def false_wakes(m: Matrix, names: Sequence[str],
                non_keyword: Sequence[str] = ("_unknown_", "_silence_")
                ) -> tuple[int, int]:
    """(non-keyword clips scored as a keyword, non-keyword clips).

    Both silence and unknown count: a spotter that wakes on background noise is
    as broken as one that wakes on the wrong word.
    """
    nk = [c for c, n in enumerate(names) if n in non_keyword]
    kw = [c for c, n in enumerate(names) if n not in non_keyword]
    return (sum(m[t][p] for t in nk for p in kw),
            sum(sum(m[t]) for t in nk))


def payoff(m: Matrix, names: Sequence[str], target: float = 0.85
           ) -> List[tuple[float, int]]:
    """(percentage points gained, class) for lifting one class to `target`.

    Raising a class costs roughly the same work whatever its size, so the
    payoff is proportional to its share of the split. Sorted best first.
    """
    n = sum(sum(row) for row in m)
    out = []
    for s in class_stats(m, names):
        if s.recall < target and n:
            out.append(((target - s.recall) * s.n_true / n * 100, s.idx))
    return sorted(out, reverse=True)


def report(m: Matrix, names: Sequence[str], target: float = 0.85,
           n_weak: int = 4, n_pairs: int = 5) -> str:
    """The whole reading, as text. Pure, so the test can assert on it."""
    L: List[str] = []
    stats = class_stats(m, names)
    by_idx = {s.idx: s for s in stats}
    n = sum(sum(row) for row in m)
    correct = sum(m[c][c] for c in range(len(names)))

    L.append("weakest classes, and what they lose to:")
    for s in stats[:n_weak]:
        row = [(v, p) for p, v in enumerate(m[s.idx]) if p != s.idx and v]
        top = sorted(row, reverse=True)[:2]
        to = ", ".join(f"{names[p]} {v}" for v, p in top)
        L.append(f"  {s.name:<12} {s.recall:.3f}  "
                 f"({m[s.idx][s.idx]}/{s.n_true})   -> {to}")

    L.append("")
    L.append("per class, recall vs precision "
             "(low precision = absorbing its neighbours):")
    L.append(f"  {'':<12}{'recall':>8}{'prec':>8}{'true':>7}{'pred':>7}")
    for s in stats:
        L.append(f"  {s.name:<12}{s.recall:>8.3f}{s.precision:>8.3f}"
                 f"{s.n_true:>7}{s.n_pred:>7}"
                 f"{'  <- absorbing' if s.absorbing else ''}")

    L.append("")
    L.append("most confused pairs, and whether the confusion is mutual:")
    said = {
        "mutual": "mutual -- the input does not separate these two sounds",
        "leaning": "leaning to {} -- the sounds overlap AND one side wins",
        "one way": "one way -- {} is preferred",
    }
    for p in pairs(m)[:n_pairs]:
        L.append(f"  {names[p.lo]:<10} <-> {names[p.hi]:<10} {p.total:>4}  "
                 f"({p.lo_to_hi} / {p.hi_to_lo})  {p.symmetry:.2f}  "
                 f"{said[p.kind].format(names[p.preferred()])}")

    L.append("")
    L.append(f"what reaching {target:.2f} on one class alone would add:")
    for g, c in payoff(m, names, target)[:n_weak]:
        s = by_idx[c]
        L.append(f"  {s.name:<12} {g:+.2f} pp   ({s.n_true} clips, "
                 f"{s.n_true / n:.1%} of the split)" if n else "")

    woke, n_nk = false_wakes(m, names)
    if n_nk:
        L.append("")
        L.append(f"false wakes: {woke} of {n_nk} non-keyword clips "
                 f"({woke / n_nk:.1%}) scored as a keyword.")
        L.append("  This is the error an always-on spotter pays for, and "
                 "accuracy hides it among the rest.")

    if n:
        short = (target - correct / n) * 100
        L.append("")
        L.append(f"currently {short:.2f} pp short of the {target:.2f} target "
                 f"in CLAUDE.md section 1." if short > 0 else
                 f"{-short:.2f} pp above the {target:.2f} target.")
    return "\n".join(L)
