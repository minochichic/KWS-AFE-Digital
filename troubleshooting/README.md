# troubleshooting — 미해결 3대 문제 (격리 분석)

프로젝트가 부딪힌 **세 가지 열린 문제**를 격리해서 분석·설계하는 폴더. 지금까지의 결과
(AFE 2차 GIC + spice+√ = 0.80, 4차 기각, 프리앰프 채택)는 그대로 두고, **여기서는 분석·계획만**
한다.

## 격리 규칙

- **이 폴더 안에서만 문서·스크립트 작성.** `AFE/`, `AFE_highorder/`, `AFE_micamp/`,
  `data/`, `models/`, `train/`, `configs/`는 **동결**.
- ML/RTL 코드 변경이 필요해지면 **착수 전 승인**을 받는다.
- 복귀점: `git checkout phase-b-2ndorder` (Phase B 완료 상태).

## 세 문제

| # | 문제 | 문서 | 한 줄 요약 |
|---|---|---|---|
| 1 | **AFE: GBW·오프셋** | [`01_afe_gbw_offset.md`](01_afe_gbw_offset.md) | 고주파 스윙이 작아 오프셋에 묻힘. OPA379(GBW 90k)의 한계 → op-amp 교체(TLV 등) 검토 |
| 2 | **ML: 하드웨어 인지 학습 + 구조** | [`02_ml_hardware_aware.md`](02_ml_hardware_aware.md) | 오프셋 없이 학습해 실물서 붕괴(0.80→0.47). HAT 필수. DS-CNN/GRU 등 대안 검토 |
| 3 | **FPGA 최적화** | [`03_fpga_optimization.md`](03_fpga_optimization.md) | 완전펼침은 KWS 속도(2–10 Hz)에 과대. folding/재사용 설계 필요 |

세 문제는 **연결**돼 있다: ①이 오프셋 크기를 정하고 → ②가 그 오프셋을 학습으로 흡수하며
→ ③이 그 모델을 실제 자원 안에 넣는다.

## 지금까지 확정된 사실 (이 폴더의 출발점)

- AFE 2차 GIC + √압축 + global min-max = **test 0.80** (이상 mel은 0.85).
- **4차 필터 기각**: 블러 −49%인데 정확도 하락 → 채널 겹침은 정확도 원인이 아님.
- **프리앰프(G≈10) 채택**: HF V+ 스윙 0.5–1.4 mV → 20–58 mV (오프셋 ×4–29).
  단 OPA379 GBW 때문에 **G≈10이 상한**(8 kHz 평탄 유지).
- **오프셋 취약성 (핵심)**: 오프셋 없이 학습한 모델은 vos=0.02만 줘도 0.80→0.62,
  프리앰프 수준(0.035)에서도 **0.47**. → 프리앰프만으론 불충분, **오프셋-인지 학습 필수**.

## 참고 문헌

- [`refs/HelloEdge_1711.07128v3.pdf`](refs/) — Zhang et al. (Arm), "Hello Edge: KWS on
  Microcontrollers". DNN/CNN/RNN/CRNN/**DS-CNN** 비교, MCU 자원 제약 하 설계.
- 프로젝트 루트: `MatchboxNet.pdf`, `Sub-mW_Keyword_Spotting...pdf` (Cerutti).
