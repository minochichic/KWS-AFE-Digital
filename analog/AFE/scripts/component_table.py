"""Emit the 16-channel component table (artifacts/component_table.md).

Per-channel values (RA, C, R1) come from filterbank_design.csv; the detector /
comparator components are shared across channels. Values are the continuous
design targets -- snapping to E12/E24 series (and the resulting tolerance) is a
later fabrication step (CLAUDE.md 5).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

AFE = Path(__file__).resolve().parents[1]


def main():
    d = np.loadtxt(AFE / "artifacts" / "filterbank_design.csv",
                   delimiter=",", skiprows=1)
    lines = []
    lines.append("# AFE 16채널 부품값 (SPICE 설계)\n")
    lines.append("연속(continuous) 설계값. E12/E24 스냅·공차는 제작 단계에서 반영.\n")
    lines.append("## 채널별 (GIC 필터: RA1=RA2=RA, C1=C2=C, R1)\n")
    lines.append("| ch | f_c [Hz] | Q | RA [kΩ] | C [nF] | R1 [kΩ] | 이득 [dB] |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in d:
        ch, fc, q, gain, RA, C, R1 = int(r[0]), r[2], r[4], r[5], r[6], r[7], r[8]
        lines.append(f"| {ch} | {fc:.0f} | {q:.2f} | {RA/1e3:.2f} | "
                     f"{C*1e9:.2f} | {R1/1e3:.1f} | {gain:.1f} |")
    lines.append("\n## 채널 공통 (모든 채널 동일)\n")
    lines.append("| 부품 | 값 | 역할 |")
    lines.append("|---|---|---|")
    lines.append("| R2 = R3 | 100 kΩ | GIC 피드백 쌍 |")
    lines.append("| R4 | 10 kΩ | 검출기 입력 |")
    lines.append("| R5 | 47 kΩ | 검출기 피드백 (τ ≈ R5·C3) |")
    lines.append("| R6 | 8.25 kΩ | 검출기 기준 |")
    lines.append("| C3 | 100 nF | 엔벨로프 평활 → τ ≈ 4.7 ms |")
    lines.append("| 공급 | 0 – 1.8 V (바이어스 0.9 V) | 단일 공급 |")
    lines.append("\n## 채널별 튜닝값 (하드웨어에선 R7/R8 분압 = ML 학습 threshold)\n")
    lines.append("비교기 임계 V- 는 각 채널 엔벨로프 범위 중앙에 맞춘다(신호 레벨 의존).")
    lines.append("run_transient.py 가 2-pass로 자동 설정.")
    txt = "\n".join(lines) + "\n"
    (AFE / "artifacts" / "component_table.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
