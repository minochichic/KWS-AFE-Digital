"""rtl/bench/bench_mac.v: every KIND runs, matches its reference, and its
output toggles (so Vivado cannot optimise the engine away). Needs iverilog."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IV = shutil.which("iverilog")


@pytest.mark.skipif(IV is None, reason="iverilog not installed")
@pytest.mark.parametrize("kind", list(range(13)))
def test_bench_kind(tmp_path, kind):
    exe = tmp_path / "b"
    subprocess.run([IV, "-g2012", f"-DKIND_UNDER_TEST={kind}", "-s", "tb_bench_mac",
                    "-o", str(exe), str(ROOT / "rtl/kws_bin_mac.v"),
                    str(ROOT / "rtl/bench/bench_mac.v"), str(ROOT / "rtl/bench/tb_bench_mac.v")],
                   check=True, capture_output=True)
    out = subprocess.run(["vvp", str(exe)], check=True, capture_output=True, text=True,
                         cwd=ROOT / "rtl/bench").stdout   # bench_rom.mem
    assert "20 frames checked, 0 failures" in out, out
