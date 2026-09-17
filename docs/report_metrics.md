# 평가 지표 체계 — 중간보고서용 (2026-09-17)

> 목적: 우리 수치와 선행연구 수치를 **같은 조건끼리만** 비교한다. KWS 논문의 "정확도"
> 한 줄은 서로 다른 세 가지를 가리킬 수 있고, 섞으면 비교가 성립하지 않는다.

---

## 1. 세 층위 (T1–T3)

| 층위 | 이름 | 무엇을 재는가 | 입력 | 정답 판정 | 우리 도구 |
|---|---|---|---|---|---|
| **T1** | 클립 정확도 (소프트웨어) | 모델 자체의 분류 능력 | 1 초 클립, 단어가 창 안에 온전 | 클립당 argmax 1 회 == 라벨 | `train` (float), `experiments.fixed_accuracy` (정수) |
| **T2** | 하드웨어 재현성 (칩) | RTL/칩이 T1 의 정수 경로를 **그대로** 계산하는가 | T1 과 같은 클립을 칩 ROM 에 | 칩 class == 파이썬 정수 예측, 클립마다 | `rtl/selftest/run_selftest.tcl` |
| **T3** | 연속 동작 (always-on) | 단어 시점을 모르는 상황의 검출 | 무음 1 s + 대상 1 s + 무음 1 s, 100 ms hop | N=5 연속 + margin 통과 = 검출 | `experiments.eval_streaming` |

- **T2 는 정확도가 아니라 동치다.** T2 가 100 % 일치하면 칩 정확도 = T1 정수 정확도로
  환원된다. 그래서 T2 표본은 "칩이 틀리지 않음" 을 보이는 데 충분한 크기면 되고,
  정확도의 통계적 정밀도는 T1 전체 test 셋이 담당한다.
- **T3 는 문헌에 통일된 규약이 없다.** 우리 T3 는 잘라 붙인 3 초 스트림 위의 자체
  규약이다(`docs/windowing_plan.md` §8). 현장 오검출률(FA/h) 이 아니다.

## 2. 데이터셋 규약

| 항목 | 우리 | 표준 설정 (TensorFlow speech_commands 예제) |
|---|---|---|
| 데이터 | Google Speech Commands **v2** (105,829 발화, 35 단어) | 같음 |
| 12 클래스 | yes no up down left right on off stop go + silence + unknown | 같음 |
| 분할 | 공식 `validation_list.txt` / `testing_list.txt` (80:10:10) | 같음 |
| unknown / silence 비율 | 키워드 수의 10 % / 10 % (`data.unknown_fraction`, `silence_fraction`) | 같은 비율 → test **4,890 클립** (Rybakov 2020 이 명시) |
| test 클립 수 | **원격에서 확인 필요** (`fixed_accuracy` 출력 `n_clips`) | 4,890 |

→ 우리 test 가 4,890 에 가까우면 표준 12-class v2 수치와 직접 비교 가능하다. 다르면
보고서에 그 차이를 적는다.

## 3. 선행연구 — 조건별 정리

"측정" 열: **SW** = 소프트웨어 시뮬, **Chip** = 제작 칩 실측, **MCU** = 상용 MCU 실측/추정.
"평가" 열: **Clip** = 1 초 클립당 판정(T1 형), **Stream** = 연속 입력 규약.

### 3.1 참고 논문 (직접 기반)

| 연구 | 특징 추출 | 분류기 | 클래스 | 정확도 | 측정 | 평가 | test 크기 |
|---|---|---|---|---|---|---|---|
| Cerutti et al., arXiv 2201.03386 (2022) | **아날로그 필터뱅크 + 비교기 → 이진 이미지** | BNN (GAP8 MCU) | GSC v2 12 | 64-ch **86.0 %**, 8-ch **76.3 %** | SW 시뮬 + MCU 추정 | Clip | 명시 없음 (공식 80:10:10) |
| 〃 | 〃 | 〃 | GSC v2 10 | 64-ch 88.8 %, 8-ch 85.8 % | 〃 | Clip | 〃 |
| Majumdar & Ginsburg, MatchboxNet (2020) | log-mel (디지털) | 1D TCS CNN, FP | GSC v1/v2 | **확인 필요** (PDF 재독 후 기입) | SW | Clip | Rybakov: "다른 train/test 설정이라 직접 비교 불가" |

### 3.2 같은 과제(12-class GSC)를 하드웨어로 구현한 연구

| 연구 | 공정 | 특징 추출 | 분류기 | 클래스 | 정확도 | 전력 | 측정 |
|---|---|---|---|---|---|---|---|
| Giraldo, Vocell, JSSC 2020 | 65 nm | 디지털 MFCC | NN | 12 | 90.87 % | 16 µW (KWS 모드) | Chip |
| Kim et al., ISSCC 2022 / JSSC 2022 | 65 nm | **아날로그 링오실레이터 시간영역** | RNN | 12 | 86.03 % | 23 µW (AFE+FE+NN) | Chip |
| Chen et al., DeltaKWS, TCASAI 2025 | 65 nm | 디지털 IIR BPF | ΔGRU | 12 / 11 | 89.5 % / 90.5 % | 5.22 µW, 36 nJ/dec | Chip (FPGA 호스트가 클립 주입) |
| Tan et al., ISSCC 2024 | 28 nm | — | Transfer computing | 12 | 91.8 % | 1.73 µW | Chip |
| Seol et al., ISSCC 2023 | 28 nm | 아날로그 프론트엔드 | Skip-RNN | **7** | 92.8 % | 1.48 µW | Chip |
| Kosuge et al., VLSI 2023 | 40 nm | — | CNN (wired-logic) | 35 / 10 | 78.2 % / 88.0 % | 152.8 µW | Chip |

(Tan·Seol·Kosuge·Kim 수치는 DeltaKWS Table II, Vocell 수치는 Cerutti §II-C 에서 인용.)

### 3.3 소프트웨어 상한 (같은 데이터셋·규약)

| 연구 | 모델 | GSC v2 12-label | 비고 |
|---|---|---|---|
| Rybakov et al., Interspeech 2020 | MHAtt-RNN | 98.0 % | FP, SpecAugment, test 4,890 |
| 〃 | DS-CNN (stride) | 97.0 % | 〃 |
| 〃 | TC-ResNet | 97.4 % | 〃 |
| Zhang et al., Hello Edge (2017) | DS-CNN | Cerutti Table III 기준 84.6 % (MCU 에너지 비교 행) | v1 기준 원 논문 수치는 별도 |

### 3.4 스트리밍(T3) 을 어떻게 다뤘나

- **Rybakov 2020**: 스트리밍 **추론**(20 ms 마다 갱신)을 구현했지만 정확도는 여전히
  "test 클립을 돌려 라벨과 비교" — 즉 T1 형이다. 연속 녹음의 검출률/오검출률은 보고하지 않는다.
- **칩 논문들(Table 3.2)**: 전부 1 초 GSC 클립 단위 정확도. DeltaKWS 는 FPGA 가 1 초
  클립 스트림을 칩에 흘려 넣고 클립당 결정을 잰다.
- **Cerutti**: 첫 인터럽트 후 1 초를 모아 이미지 재구성 → 1 회 분류. 트리거형이며
  sliding window 판정이 없다.

→ **12-class GSC 에서 T3 형 수치를 표준 규약으로 보고한 연구는 위 목록에 없다.**
우리 T3 는 비교표가 아니라 별도 절에서 "자체 규약" 으로 보고한다.

## 4. 우리 수치 — 현재 채워진 것 / 채울 것

| 층위 | 항목 | `bd_base` | `bd_base_ft20_partial75` | 출처 |
|---|---|---|---|---|
| T1 | test 전체, float | 0.825 (학습 시 기록) | 확인 필요 | `docs/ABLATIONS.md` / 원격 |
| T1 | test 전체, **정수** | **원격 실행 필요** | **원격 실행 필요** | `fixed_accuracy --json-out` |
| T1 | 균형 1200 클립, 정수 | 0.833 | 0.839 | `rtl/gen/*/selftest_bal` |
| T1 | val 중앙 클립 (T3 스트림의 가운데 창) | 확인 필요 | 82.68 % (2540/3072) | `windowing_plan.md` §10.9 |
| T2 | 칩 일치 (앞 600, 2 클래스) | **600/600 ×2** | **600/600 ×2** | `selftest_report.md` §5 |
| T2 | 칩 일치 (균형 600, 12 클래스 × 50) | **600/600 ×2** | 빌드/실행 대기 | 〃 |
| T3 | val keyword recall (N5, margin) | 67.73 % (m 1.25) | **69.53 %** (m 1.05) | `windowing_plan.md` §10.9 |
| T3 | val quiet 오검출 스트림 | 52/512 | 41/512 | 〃 |
| T3 | test | 미실행 (정책 고정 후 1 회) | 미실행 | — |
| T3 | 칩 | 미실행 | 미실행 | 다음 단계 |

## 5. 표본 크기에 대해

- 선행 칩 논문들은 **test 클립 수를 명시하지 않는다** (DeltaKWS, Kim, Cerutti 모두).
  표준 설정이면 4,890 이다.
- 우리 T1 은 전체 test 를 쓴다 → 표준과 같은 크기.
- 우리 T2 (칩) 는 1,200 클립(3 비트스트림 × 600, 일부 중복 모델) 이다. T2 는 동치 검증이므로
  "몇 개 틀렸나" 의 신뢰구간이 의미를 가진다: 600 중 0 오류이면 오류율 95 % 상한
  ≈ 3/600 = 0.5 % (rule of three). 1,200 이면 0.25 %.
- 전체 4,890 을 칩에서 돌리려면 16 비트 ROM 기준 비트스트림 약 5 개(1,000 클립씩), 보드 실행
  각 1 분. 최종보고서 항목으로 둔다.
