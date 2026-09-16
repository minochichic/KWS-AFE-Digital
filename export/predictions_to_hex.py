"""predictions_fixed.txt (decimal) -> expected.hex ($readmemh, hex).

Why this exists at all: `predictions_fixed.txt` holds ONE DECIMAL PER LINE, and
`$readmemh` reads HEX. For classes 0-9 the two agree, so the bug hides. Class 11
appears in the very first two-clip dump, and `$readmemh` reads "11" as 0x11 = 17.

`rtl/tb/tb_top.v` dodges this with `$fscanf(fh, "%d", ...)`, which is correct and
**not synthesizable**. The self-test harness has to hold the answer key in a ROM
that lives in the bitstream, so it needs the same numbers in hex.

Keeping it a separate file rather than changing golden.py's output:
`predictions_fixed.txt` is read by tb_top, by the eval scripts and by people.
Turning it into hex would make "11" mean eleven in one place and seventeen in
another -- the exact ambiguity this script exists to remove.

Stdlib only, so it runs anywhere -- including Vivado's bundled interpreter on a
machine with no project Python:

    python -m export.predictions_to_hex rtl/gen/<tag>/selftest

    'C:/AMDDesignTools/2026.1/tps/win64/python-3.13.0/python.exe' \
        -m export.predictions_to_hex rtl/gen/<tag>/selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

# 12 classes: 10 keywords + silence + unknown (CLAUDE.md 1). A value outside
# that range means the file is not what we think it is, and a silently wrong
# answer key would make every hardware mismatch look like an RTL bug.
N_CLASSES = 12


def convert(src: Path, dst: Path, digits: int = 1) -> int:
    """Write one hex word per line. Returns the count."""
    vals = []
    for lineno, line in enumerate(src.read_text().split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            v = int(line, 10)                    # base 10 EXPLICITLY
        except ValueError:
            raise SystemExit(f"{src}:{lineno}: not a decimal integer: {line!r}")
        if not 0 <= v < N_CLASSES:
            raise SystemExit(
                f"{src}:{lineno}: class {v} outside 0..{N_CLASSES - 1}. "
                f"Either the file is not predictions_fixed.txt or the model "
                f"has a different class count than this script assumes.")
        vals.append(v)
    if not vals:
        raise SystemExit(f"{src}: no values")
    dst.write_text("\n".join(f"{v:0{digits}x}" for v in vals) + "\n")
    return len(vals)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip().split("\n\n")[-1].strip())
    d = Path(sys.argv[1])
    src, dst = d / "predictions_fixed.txt", d / "expected.hex"
    if not src.exists():
        raise SystemExit(f"{src} not found")
    n = convert(src, dst)

    # Round-trip: read the hex back as hex and require the decimals again. The
    # failure this catches is the whole point of the script, so it is checked
    # rather than assumed.
    back = [int(w, 16) for w in dst.read_text().split()]
    want = [int(w, 10) for w in src.read_text().split()]
    if back != want:
        raise SystemExit(f"{dst}: round-trip mismatch -- refusing to ship it")

    hi = max(want)
    print(f"{n} predictions -> {dst}")
    print(f"max class {hi}"
          + ("  (>9, so the decimal/hex distinction was load-bearing)"
             if hi > 9 else "  (all <= 9; decimal and hex coincide here)"))


if __name__ == "__main__":
    main()
