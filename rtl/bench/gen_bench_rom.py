"""Write rtl/bench/bench_rom.mem: the weight ROM of bench_mac's memory KINDs.

    python rtl/bench/gen_bench_rom.py

256 words x 256 bits of fixed pseudo-random data, for $readmemh -- the way the
design loads its weights (CLAUDE.md 5). The first version computed the words
in an initial loop; iverilog evaluated it, Vivado folded most of it to
constants (2026-10-30: 1.5 of 4 BRAM tiles, a 32-bit ROM gone entirely).
"""
import random
from pathlib import Path

rng = random.Random(20261030)
words = [rng.getrandbits(256) for _ in range(256)]
out = Path(__file__).resolve().parent / "bench_rom.mem"
out.write_text("".join(f"{w:064x}\n" for w in words))
print("wrote", out)
