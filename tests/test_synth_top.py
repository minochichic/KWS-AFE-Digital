"""합성 래퍼가 테스트벤치와 같은 파라미터를 꽂는지.

`rtl/synth/kws_top_synth.v` 는 `rtl/tb/tb_top.v` 의 파라미터 블록을 기계적으로
추출해 만든다. 왜 검사가 필요한가: 둘이 어긋나도 **합성은 성공한다.** 다른 회로가
조용히 나올 뿐이고, 시뮬레이션은 테스트벤치 쪽만 보므로 끝까지 통과한다.

특히 ROM 경로 파라미터가 하나라도 빠지면 kws_top 의 기본값 ""가 들어가고, RTL 이
`if (ROM_FILE != "")` 로 걸러 **그 층의 가중치만 없는** 회로가 된다.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TB = ROOT / "rtl/tb/tb_top.v"
SYNTH = ROOT / "rtl/synth/kws_top_synth.v"


def param_block(text: str) -> str:
    """`kws_top #(` 의 괄호가 닫히는 데까지."""
    i = text.index("kws_top #(")
    j = text.index("#(", i) + 1
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    raise AssertionError("괄호가 안 닫힌다")


def bindings(block: str) -> dict:
    """.NAME(VALUE) -> {NAME: VALUE}, 공백 무시."""
    out = {}
    for m in re.finditer(r"\.(\w+)\s*\(([^()]*)\)", block):
        out[m.group(1)] = re.sub(r"\s+", "", m.group(2))
    return out


def test_synth_wrapper_matches_testbench() -> None:
    tb = bindings(param_block(TB.read_text()))
    sy = bindings(param_block(SYNTH.read_text()))
    assert tb, "tb_top 에서 파라미터를 못 읽었다"
    missing = sorted(set(tb) - set(sy))
    extra = sorted(set(sy) - set(tb))
    differ = sorted(k for k in set(tb) & set(sy) if tb[k] != sy[k])
    assert not missing, (
        f"합성 래퍼에 빠진 파라미터: {missing}. ROM 경로가 빠지면 그 층만 "
        f"가중치 없이 합성된다 -- 에러 없이.")
    assert not extra, f"합성 래퍼에만 있는 파라미터: {extra}"
    assert not differ, (
        "같은 파라미터에 다른 값: "
        + ", ".join(f"{k}: tb={tb[k]!r} synth={sy[k]!r}" for k in differ))


def test_rom_paths_all_reach_the_wrapper() -> None:
    """ROM 경로가 하나도 안 빠졌는지.

    이름으로 고르지 않는다 -- TL_C3_W 는 경로가 아니라 비트폭이라 이름 규칙으로는
    구별이 안 된다. **무엇에 묶였는가**로 본다: `KWS_ROM_* 에 묶인 것만 경로다.
    """
    tb = bindings(param_block(TB.read_text()))
    sy = bindings(param_block(SYNTH.read_text()))
    tb_rom = {k for k, v in tb.items() if v.startswith("`KWS_ROM")}
    sy_rom = {k for k, v in sy.items() if v.startswith("`KWS_ROM")}
    assert tb_rom, "테스트벤치에서 ROM 경로를 하나도 못 찾았다"
    assert tb_rom == sy_rom, (
        f"ROM 경로 불일치. 래퍼에만 없는 것: {sorted(tb_rom - sy_rom)}, "
        f"래퍼에만 있는 것: {sorted(sy_rom - tb_rom)}")
