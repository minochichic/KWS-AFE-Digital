"""Reading a run's history, on the shape the file actually has.

The bug this locks down is small and cost a round trip: history.json is
{"history": [...], "wall_time_s": ..., "device": ...}, not a bare list, so
iterating the top level yields dict KEYS and crashes with "string indices must
be integers" -- a message that says nothing about the real problem.
"""
from __future__ import annotations

import json

import pytest

from experiments.show_history import load


def write(tmp_path, tag, doc):
    d = tmp_path / tag
    d.mkdir()
    (d / "history.json").write_text(json.dumps(doc))
    return str(tmp_path)


def rows(n=3):
    return [dict(epoch=i, train_loss=1.0, train_acc=0.5, val_acc=0.5,
                 lr=1e-3) for i in range(1, n + 1)]


def test_the_wrapped_shape_is_what_the_trainer_writes(tmp_path) -> None:
    runs = write(tmp_path, "a", {"history": rows(), "wall_time_s": 1.0,
                                 "device": "cuda"})
    got = load(runs, "a")
    assert len(got) == 3 and got[0]["epoch"] == 1


def test_a_bare_list_still_reads(tmp_path) -> None:
    """Older files, and anything hand-made, should not need converting."""
    runs = write(tmp_path, "b", rows())
    assert len(load(runs, "b")) == 3


def test_a_missing_run_says_which_path(tmp_path) -> None:
    with pytest.raises(SystemExit, match="history.json"):
        load(str(tmp_path), "nope")


def test_an_empty_history_is_refused(tmp_path) -> None:
    runs = write(tmp_path, "c", {"history": [], "device": "cpu"})
    with pytest.raises(SystemExit, match="history"):
        load(runs, "c")
