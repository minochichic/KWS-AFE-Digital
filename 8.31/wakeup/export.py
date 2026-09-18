"""학습 결과 -> 회로 상수.

이 파일이 내는 표가 곧 배선표다. 형판은 저항의 연결 여부와 극성, 허용 오차는
비교기 기준전압, 시각은 디코더 결선이 된다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch

from .model import WakeupModel


def template_table(M: torch.Tensor) -> str:
    """형판을 사람이 읽는 표로. 1 = 켜져야 함, 0 = 꺼져야 함, X = 미연결."""
    S, C = M.shape
    rows = ["  " + "".rjust(5) + "".join(f"ch{c}".rjust(5) for c in range(C))]
    for s in range(S):
        cells = "".join(
            ("1" if int(M[s, c]) > 0 else ("0" if int(M[s, c]) < 0 else "X")).rjust(5)
            for c in range(C))
        rows.append("  " + f"s{s+1}".rjust(5) + cells)
    return "\n".join(rows)


def divider(vth: float, vdd: float = 1.8, r_total: float = 1e6) -> Dict[str, float]:
    """기준전압을 분압 저항 두 개로. V_TH = VDD * R8 / (R7 + R8)."""
    if vth <= 0 or vth >= vdd:
        return {"R7": float("nan"), "R8": float("nan")}
    r8 = r_total * vth / vdd
    return {"R7": r_total - r8, "R8": r8}


def build_report(model: WakeupModel, vdd: float = 1.8) -> Dict:
    e = model.export(vdd=vdd)
    cfg = model.cfg
    M = e["M"]
    S, C = M.shape
    states = []
    for s in range(S):
        d = divider(float(e["V_TH"][s]), vdd)
        states.append({
            "state": s + 1,
            "active": bool(e["active"][s]),
            "tau_frames": int(e["tau"][s]),
            "tau_ms": int(e["tau"][s]) * cfg.frontend.envelope_win_ms,
            "template": [int(v) for v in M[s]],
            "m": int(e["m"][s]),
            "k": int(e["k"][s]),
            "V_TH": round(float(e["V_TH"][s]), 4),
            "margin_V": round(float(e["margin_V"][s]), 4),
            "R7_ohm": None if d["R7"] != d["R7"] else round(d["R7"]),
            "R8_ohm": None if d["R8"] != d["R8"] else round(d["R8"]),
        })
    return {
        "target_word": cfg.train.target_word,
        "n_channels": C,
        "n_states": S,
        "start_k": int(e["start_k"]),
        "timeout_frames": int(e["timeout"]),
        "timeout_ms": int(e["timeout"]) * cfg.frontend.envelope_win_ms,
        "wake_width_ms": e["wake_width_ms"],
        "frame_ms": cfg.frontend.envelope_win_ms,
        "vdd": vdd,
        "n_resistors_template": int(e["n_resistors"]),
        "n_active_states": int(e["n_active_states"]),
        "channel_thresholds": [round(float(v), 4) for v in e["theta"]],
        "states": states,
    }


def format_report(rep: Dict, test: Optional[Dict] = None) -> str:
    L = []
    a = L.append
    a("=" * 66)
    a(f"  대상 단어 '{rep['target_word']}'  |  {rep['n_channels']}채널  "
      f"|  상태 {rep['n_states']}개")
    a("=" * 66)
    if test:
        a(f"\n검출 성능 (시험 분할)")
        a(f"  TPR          {test['tpr']:.3f}")
        a(f"  FPR          {test['fpr']:.3f}")
        a(f"  START 재현율 {test['start_recall']:.3f}")

    a(f"\n타이밍  (1 프레임 = {rep['frame_ms']:.0f} ms)")
    a(f"  START 조건    채널 {rep['start_k']}개 이상 동시 ON")
    for s in rep["states"]:
        a(f"  tau({s['state']})        {s['tau_frames']:3d} 프레임 = {s['tau_ms']:.0f} ms")
    a(f"  timeout       {rep['timeout_frames']:3d} 프레임 = {rep['timeout_ms']:.0f} ms")
    a(f"  WAKE 폭       {rep['wake_width_ms']:.0f} ms")

    a(f"\n형판  (1 = ON 이어야, 0 = OFF 여야, X = 미연결)")
    M = torch.tensor([s["template"] for s in rep["states"]])
    a(template_table(M))

    a(f"\n상태별 비교기")
    a(f"  {'상태':<5} {'m':>3} {'k':>3} {'V_TH [V]':>9} {'여유 [V]':>9} "
      f"{'R7 [kΩ]':>9} {'R8 [kΩ]':>9}")
    for s in rep["states"]:
        r7 = "-" if s["R7_ohm"] is None else f"{s['R7_ohm']/1000:.1f}"
        r8 = "-" if s["R8_ohm"] is None else f"{s['R8_ohm']/1000:.1f}"
        a(f"  s{s['state']:<4} {s['m']:>3} {s['k']:>3} {s['V_TH']:>9.3f} "
          f"{s['margin_V']:>9.3f} {r7:>9} {r8:>9}")
    dead = [s["state"] for s in rep["states"] if not s.get("active", True)]
    if dead:
        a(f"\n  ! 상태 {dead} 는 k=0 또는 m=0 이라 항상 통과한다 -- 판정에 기여하지")
        a(f"    않으면서 비교기와 플립플롭만 먹는다. 상태 수를 {rep['n_active_states']}"
          f" 로 줄이고 다시 학습할 것.")
    a(f"\n  형판 저항 {rep['n_resistors_template']}개 "
      f"(전 채널 사용 시 {rep['n_channels'] * rep['n_states']}개)")
    a(f"  기준전압 분압 저항 {2 * rep['n_states']}개 "
      f"(합 1 MΩ 기준, 실제 값은 전력 예산에 맞춰 조정)")

    a(f"\n채널 문턱 theta_c (정규화 스케일)")
    a("  " + "  ".join(f"ch{i}:{v:.3f}"
                       for i, v in enumerate(rep["channel_thresholds"])))
    a("\n  * theta_c 는 아날로그 전단의 비교기 기준전압이 된다. 실제 전압으로")
    a("    옮기려면 동료의 VTHR 격자(10 mV 간격)에 양자화한 뒤 다시 평가할 것.")
    a("=" * 66)
    return "\n".join(L)


def write_report(model: WakeupModel, out_dir: Path, *,
                 test: Optional[Dict] = None, history: Optional[list] = None,
                 vdd: float = 1.8) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = build_report(model, vdd=vdd)
    if test:
        rep["test"] = test
    if history:
        rep["history"] = history
    (out_dir / "circuit.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2))
    txt = format_report(rep, test)
    (out_dir / "circuit.txt").write_text(txt)
    print("\n" + txt)
    return rep
