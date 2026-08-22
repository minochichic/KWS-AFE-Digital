"""Conventions that are easy to state and easy to forget.

Each of these cost a failed run at least once, and two of them cost it twice --
the guard is written down in rtl/README.md section 4 and then broken anyway,
because remembering a rule at the moment you break it is exactly what does not
happen. A test does not have that problem.

Runs with no simulator and no torch.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

RTL = Path(__file__).resolve().parents[1] / "rtl"
SOURCES = sorted(RTL.glob("*.v")) + sorted((RTL / "tb").glob("*.v"))


def _lines(p):
    return p.read_text().splitlines()


def test_no_string_literal_is_split_across_lines():
    """Verilog-2005 has no adjacent-literal concatenation.

    C and Python join "a" "b" into "ab"; Verilog treats the second one as a
    syntax error. Every $display here must keep its format string on one line
    and wrap only the argument list.
    """
    bad = []
    for p in SOURCES:
        lines = _lines(p)
        for i, ln in enumerate(lines[:-1]):
            if re.search(r'"\s*$', ln) and re.match(r'\s*"', lines[i + 1]):
                bad.append(f"{p.relative_to(RTL.parent)}:{i + 1}")
    assert not bad, (
        "a format string is split across lines, which Verilog-2005 will not "
        f"compile: {bad}")


def test_rst_n_is_asynchronous_everywhere():
    """Mixing the two makes rst_n both an async reset and a synchronous term,
    which lint flags (SYNCASYNCNET) and which leaves a synthesis tool guessing
    how to build the reset tree. Assertions gate on a valid signal instead."""
    bad = []
    for p in sorted(RTL.glob("*.v")):
        txt = p.read_text()
        for m in re.finditer(r"always @\(posedge clk\)(?!\s*or)", txt):
            seg = txt[m.end():m.end() + 400].split("always @")[0]
            if "rst_n" in seg:
                bad.append(f"{p.name}:{txt[:m.start()].count(chr(10)) + 1}")
    assert not bad, f"rst_n used synchronously: {bad}"


def test_the_design_never_names_an_export_tag():
    """An analog change costs a retrain and new .hex, never an edit to a .v
    (docs/ICD.md). A tag in the RTL would break that the moment a second track
    exists -- and it is why there is no rtl_fixed/ directory."""
    bad = [p.name for p in sorted(RTL.glob("*.v"))
           if re.search(r"rtl/gen/\w+/", p.read_text())]
    assert not bad, f"RTL source names an export path: {bad}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_every_file_sets_and_restores_default_nettype(path):
    """`default_nettype none` turns a typo'd signal name into an error instead
    of an implicit one-bit wire. Restoring it at the end keeps the setting from
    leaking into whatever the tool compiles next."""
    txt = path.read_text()
    assert "`default_nettype none" in txt, "missing `default_nettype none"
    assert txt.rstrip().endswith("`default_nettype wire"), (
        "must end with `default_nettype wire")


@pytest.mark.parametrize("path", sorted(RTL.glob("*.v")), ids=lambda p: p.name)
def test_assertions_are_compiled_out_of_the_bitstream(path):
    """Every $display must sit inside `ifdef KWS_ASSERT. They are worth their
    weight in simulation and worth zero gates on the board."""
    txt = path.read_text()
    if "$display" not in txt:
        pytest.skip("no assertions")
    depth, guarded, total = 0, 0, 0
    for ln in txt.splitlines():
        s = ln.strip()
        if s.startswith("`ifdef KWS_ASSERT"):
            depth += 1
        elif s.startswith("`endif") and depth:
            depth -= 1
        if "$display" in s:
            total += 1
            guarded += depth > 0
    assert guarded == total, f"{total - guarded} of {total} outside KWS_ASSERT"


@pytest.mark.parametrize("path", sorted((RTL / "tb").glob("tb_*.v")),
                         ids=lambda p: p.name)
def test_clip_count_comes_from_the_export_not_a_literal(path):
    """A testbench must not decide how many golden clips there are.

    export/golden.py writes `KWS_GOLD_CLIPS` and defaults to --clips 8. Every
    testbench used to hardcode 2, so an 8-clip export was checked a quarter of
    the way and still printed ok -- the failure mode of a literal here is
    silent under-testing, which is the kind a green run hides.
    """
    txt = path.read_text()
    if "CLIPS" not in txt:
        pytest.skip("no clip loop")
    m = re.search(r"localparam integer\s+CLIPS\s*=\s*(.+?);", txt)
    assert m, "CLIPS is used but never declared as a localparam"
    assert m.group(1).strip() == "`KWS_GOLD_CLIPS", (
        f"CLIPS = {m.group(1).strip()} -- read it from the export instead")
