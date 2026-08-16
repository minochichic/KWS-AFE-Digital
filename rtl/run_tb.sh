#!/usr/bin/env bash
# Build and run one testbench. Run from the repo root.
#   ./rtl/run_tb.sh bin_mac
# Needs iverilog:  sudo apt install iverilog   /   brew install icarus-verilog
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
NAME="${1:-bin_mac}"
OUT="$(mktemp -d)/tb_${NAME}"
iverilog -g2005 -Wall -DKWS_ASSERT -I. -o "$OUT" \
    "rtl/kws_${NAME}.v" "rtl/tb/tb_${NAME}.v"
vvp "$OUT"
