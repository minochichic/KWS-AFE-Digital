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
# every module, not just the DUT: modules instantiate each other
# (dw_conv wraps bin_mac) and a missing one is a link error, not a
# design question worth asking the caller about
ALL_RTL=$(ls rtl/*.v)
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
              --top-module "kws_${NAME}" $ALL_RTL
    echo "lint clean"
else
    echo "== lint skipped (verilator not installed) =="
fi

echo "== simulate =="
OUT="$(mktemp -d)/tb_${NAME}"
LOG="${OUT}.log"
# -DKWS_ASSERT arms the accumulator bound checks inside the DUT.
# -I. so the testbench can include the generated vectors/expect.vh by repo path.
iverilog -g2005 -Wall -DKWS_ASSERT -I. -o "$OUT" $ALL_RTL "$TB"
vvp "$OUT" | tee "$LOG"

# Do not rely on $fatal to set the exit status: it is a SystemVerilog task and
# what a Verilog-2005 simulator does with it varies. The log is the contract.
if grep -qE '^(FAIL|ASSERT)' "$LOG"; then
    echo "FAILED -- see above" >&2
    exit 1
fi
# ", 0 failures" and not "0 failures": the latter also matches "10 failures".
grep -qE ', 0 failures' "$LOG" || { echo "no pass line in output" >&2; exit 1; }

# Record it. Until now a run's result lived only in terminal scrollback, so
# "what has been verified against what" was something a person remembered
# rather than something the repo knew -- and that is exactly the kind of claim
# that rots. docs/make_site.py reads this file; nothing else may write it.
RES="rtl/results.json"
CHECKED=$(grep -oE '[0-9]+ frames checked' "$LOG" | tail -1 | grep -oE '^[0-9]+')
python3 - "$RES" "$NAME" "${CHECKED:-0}" "$LOG" <<'EOPY'
import json, pathlib, subprocess, sys, datetime
path, name, checked, log = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
p = pathlib.Path(path)
data = json.loads(p.read_text()) if p.is_file() else {}
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
data[name] = {
    "passed": True,
    "checked": checked,
    "commit": sha,
    "when": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "lines": [l for l in pathlib.Path(log).read_text().splitlines()
              if l.startswith("ok ") or "failures" in l],
}
p.write_text(json.dumps(dict(sorted(data.items())), indent=2) + "\n")
print(f"recorded {name}: {checked} checked -> {path}")
EOPY
