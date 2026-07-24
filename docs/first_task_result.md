# FIRST_TASK 결과 요약 — base 실험 (sc_v2)

FIRST_TASK.md 단계 1~6를 완료하고, base 설정으로 Google Speech Commands v2
12-class 학습을 1회 수행한 결과 요약. (2026-07-21 기준)

---

## 1. 마일스톤 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 스캐폴드·config 레이어 | ✅ |
| 2 | 이진 연산 원자 + 단위 테스트 | ✅ |
| 3 | AFE 이진화 모듈 (학습가능 threshold) | ✅ |
| 4 | BinaryMatchboxNet 조립 | ✅ |
| 5 | 학습 루프 + 합성 오버핏 검증 | ✅ |
| 6 | Speech Commands v2 파이프라인 + 학습 | ✅ (첫 실행 완료) |

단위 테스트 118개 통과. 학습/데이터는 Colab(T4), 코드는 로컬→GitHub→Colab pull.

---

## 2. 실험 사양 (configs/base.yaml)

**입력 (AFE)** — 파형 → `[16, 128]` ∈ {-1,+1}
- STFT 25 ms / 10 ms, n_fft 512
- **Mel 필터뱅크 16개**, 50–7500 Hz, **Mel 도메인 등간격** (Hz로는 저주파 촘촘,
  Fig.3과 일치). 실제 center: 156, 281, 438, …, 5500, 6438 Hz
- 엔벨로프 = 10 ms 윈도우 max → native_T 100 → 128 zero-pad
- per-clip 전역 min-max → 채널별 학습가능 threshold(16개, sign-STE)

**모델 (BinaryMatchboxNet-3x2x64, 총 96,524 params, 이진 58%)**

| 스테이지 | 정밀도 | 커널 | 출력 | params |
|---|---|---|---|---|
| Conv1 | INT8 | 11,s2 | 128 | 22,784 |
| B1 | 이진 TCS×2 +res | 13 | 64 | 23,616 |
| B2 | 이진 TCS×2 +res | 15 | 64 | 10,624 |
| B3 | 이진 TCS×2 +res | 17 | 64 | 10,880 |
| Conv2 | 이진 separable | 29,dil2 | 128 | 10,432 |
| Conv3 | INT8 | 1 | 128 | 16,640 |
| Conv4 | fixed | 1 | 12 | 1,548 |

**학습**: Adam lr 1e-3, CrossEntropy, ReduceLROnPlateau(factor 0.1, patience 10),
batch 128, 100 epoch, seed 1234, **증강 없음**. QAT(latent fp + STE), AFE threshold
동시 학습.

**데이터**: GSC v2, 공식 80:10:10 split. 12-class = 키워드 10
(yes no up down left right on off stop go) + silence + unknown.
train 36,923 / val 4,443 / test 4,888. unknown은 키워드 클래스 규모로 subsample,
silence는 background noise에서 합성.

---

## 3. 결과

### 🎯 목표 달성: f_max=8000에서 test 0.850 (85% 돌파)

`f_max`를 7500→8000으로 올린 것만으로 **test 0.809 → 0.850 (+4.1%p)**, 목표 85%
돌파. seed 동일(1234), f_max만 다름 → 순수 f_max 효과. 16-filter AFE에서는 필터
배치가 정확도를 크게 좌우한다는 Cerutti Fig.6(8-filter에서 범위만 바꿔 53.9→76.3%)와
일치.

| 지표 | f_max=7500 (sc_v2) | **f_max=8000** |
|---|---|---|
| test (last, ep100) | 0.809 | **0.850  ✓ MEETS** |
| test (best.pt) | 0.8236 (ep64) | — |
| final train acc | 0.849 | 0.872 |
| best val | 0.840 (ep64) | 0.862 (ep79) |
| final val | 0.833 | 0.858 |

이하 상세 곡선/분석은 sc_v2(7500) 기준. wall ≈ 3.3 h (T4).

| 지표 | 값 (sc_v2, f_max=7500) |
|---|---|
| best val | 0.8397 (epoch 64) |
| final val | 0.8328 (epoch 100) |
| final train | 0.8493 |
| test (last 모델) | 0.809 |
| wall time | 11,820 s ≈ 3.28 h (T4) |

곡선: [`diagrams/07_sc_v2_training_curve.svg`](diagrams/07_sc_v2_training_curve.svg)

- lr 드롭(epoch 51, 64)에서 계단식 상승, **약 epoch 67 수렴** (이후 평탄).
- **과적합 아님, 약한 과소적합**: train−val +1.6%, val−test +2.4%.
- test는 val보다 낮은 게 정상(다른 split). 또한 test 평가에 **best.pt(ep64)가
  아니라 last(ep100) 모델**을 씀 → best.pt로 재평가 시 소폭 상승 여지.

### 벤치마크 대비 (논문 확인 수치)

| 구성 | 12-class acc |
|---|---|
| Cerutti 8ch AFE | 76.3% |
| **우리 16ch AFE** | **80.9%** |
| Cerutti 64ch AFE | 86.0% |
| Cerutti BNN(binary-input, Mel) | 85.6% |
| AFE 방식 상한(논문) | ~85% |

→ 16채널에서 80.9%(7500)/**85.0%(8000)**. f_max=8000이 64ch(86.0%)에 근접 —
소수 필터에선 배치가 채널 수만큼 중요함을 보여줌.

---

## 4. 다음 단계 (85% 돌파 이후)

이제 85%를 넘겼으므로 순서가 바뀐다:

1. **best.pt로 f_max=8000 test 재평가** (best val 0.862 ep79 → test ~0.855 예상).
2. **재현 확인** — 다른 seed 1회로 0.850이 우연이 아님을 확인(권장).
3. **(C, T) sweep** — 이제 85% 넘는 설정이 있으니 "**최소** 크기" 탐색이 의미 있음.
   T=128 고정, C를 16/32/48로 낮춰 어디까지 85%가 유지되는지 (broken-T 자동 skip됨).
4. 더 밀어올리기: conv2 dense, 증강(SpecAugment), f_min도 함께 탐색.

효율 메모: 수렴이 ~67 epoch에서 끝나므로 sweep은 65~70 epoch로 단축 가능.

---

## 5. 발견·수정한 이슈

- **base.yaml keywords YAML Norway 버그**: `yes/no/on/off`가 불리언으로 파싱됨.
  단 데이터 파이프라인은 `data/speech_commands.py`의 `KEYWORDS` 상수로 라벨링하므로
  **이번 학습에는 영향 없음**(코드로 확인). base.yaml에서 따옴표로 수정 + 회귀 테스트 추가.
- **Mel 간격 검토 완료**: "등간격"은 Hz 등차수열이 아니라 Mel 척도 등간격 → Hz로는
  저주파 촘촘(Fig.3 일치). 구현 정상, 수정 없음.
- **test 평가에 last 모델 사용**(best 아님) — best.pt(ep64) 재평가 결과 **0.8236**
  (last 0.809보다 +1.5%p). 방법론상 best-val 모델 사용이 표준.
- **f_max 7500 → 8000 상향** (Cerutti 최적 50 Hz–8 kHz에 맞춤). sc_v2 baseline은
  7500이었으므로, 다음 실행의 정확도 변화에는 이 차이가 섞인다. 전체 AFE 변수 설명:
  [`afe_config.md`](afe_config.md).
