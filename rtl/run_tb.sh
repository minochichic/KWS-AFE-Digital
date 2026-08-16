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
    # -DKWS_ASSERT here too, and not only for the simulator: without it the
    # assertion block is invisible to lint, so the upper bits of the wide
    # accumulator look unused and lint is checking a different configuration
    # than the one that runs. Lint should see the code the simulator sees.
    # -Wall minus the style-only ones that fight Verilog-2001 conventions.
    verilator --lint-only -Wall -DKWS_ASSERT \
              -Wno-DECLFILENAME -Wno-VARHIDDEN \
              --top-module "kws_${NAME}" "$DUT"
    echo "lint clean"
else
    echo "== lint skipped (verilator not installed) =="
fi

echo "== simulate =="
OUT="$(mktemp -d)/tb_${NAME}"
LOG="${OUT}.log"
# -DKWS_ASSERT arms the accumulator bound checks inside the DUT.
# -I. so the testbench can include the generated vectors/expect.vh by repo path.
iverilog -g2005 -Wall -DKWS_ASSERT -I. -o "$OUT" "$DUT" "$TB"
vvp "$OUT" | tee "$LOG"

# Do not rely on $fatal to set the exit status: it is a SystemVerilog task and
# what a Verilog-2005 simulator does with it varies. The log is the contract.
if grep -qE '^(FAIL|ASSERT)' "$LOG"; then
    echo "FAILED -- see above" >&2
    exit 1
fi
# ", 0 failures" and not "0 failures": the latter also matches "10 failures".
grep -qE ', 0 failures' "$LOG" || { echo "no pass line in output" >&2; exit 1; }
