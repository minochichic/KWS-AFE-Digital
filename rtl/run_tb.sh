#!/usr/bin/env bash
# Lint, then build and run one testbench. Run from anywhere in the repo.
#
#   ./rtl/run_tb.sh bin_mac
#
#   sudo apt install -y iverilog verilator     # Debian/Ubuntu/WSL
#   brew install icarus-verilog verilator      # macOS
#
# Lint and simulation catch different things, so both run. Lint reads the code
# and finds width truncation or an unintended latch without any stimulus;
# simulation runs it and finds values that disagree with the golden vectors,
# but only for data that reaches the bug. Verilator is used lint-only -- as a
# simulator it is 2-state and would not propagate X, so an unreset register
# reads as 0 and the bug survives to the board.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

NAME="${1:-bin_mac}"
DUT="rtl/kws_${NAME}.v"
TB="rtl/tb/tb_${NAME}.v"
[ -f "$DUT" ] || { echo "no such module: $DUT" >&2; exit 1; }
[ -f "$TB" ]  || { echo "no testbench: $TB" >&2; exit 1; }

if command -v verilator >/dev/null 2>&1; then
    echo "== lint =="
    # -Wall minus the style-only ones that fight Verilog-2001 conventions
    verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-VARHIDDEN \
              --top-module "kws_${NAME}" "$DUT"
    echo "lint clean"
else
    echo "== lint skipped (verilator not installed) =="
fi

echo "== simulate =="
OUT="$(mktemp -d)/tb_${NAME}"
# -DKWS_ASSERT arms the +-n_terms accumulator bound check inside the DUT.
# -I. so the testbench can include the generated vectors/expect.vh by repo path.
iverilog -g2005 -Wall -DKWS_ASSERT -I. -o "$OUT" "$DUT" "$TB"
vvp "$OUT"
