# 디지털 모델

AFE가 만드는 `[16, 128]` 이진 이미지를 받아 12-class를 분류하는 네트워크와 그 학습 방법.

---

## 1. 어느 버전인가

| 항목 | 값 |
|---|---|
| 모델 | **BinaryMatchboxNet-3x2x64** |
| 파라미터 | **96,524** (논문 MatchboxNet-3x2x64의 93K와 같은 급) |
| 입력 | `[16, 128]`, 값 {−1, +1} |
| 출력 | 12 class (키워드 10 + silence + unknown) |
| 설정 파일 | [`configs/base.yaml`](../configs/base.yaml) — 모든 크기의 단일 출처 |
| 코드 | [`models/binary_matchboxnet.py`](../models/binary_matchboxnet.py) |

## 2. 구조 — 부분 이진화

Cerutti Fig.5의 교훈(입력·양 끝단을 이진화하면 정확도 급락)을 반영해 **전부 이진화하지
않는다.**

| 스테이지 | 종류 | 정밀도 | 이유 |
|---|---|---|---|
| Conv1 (prologue) | k=11, stride 2 | **INT8** | 첫 층 이진화 금지. 입력이 이미 ±1이라 곱셈 없이 부호 누산으로 구현 가능 |
| B1 | TCS ×2, k=13 | **이진** | 연산·파라미터의 대부분 |
| B2 | TCS ×2, k=15 | **이진** | |
| B3 | TCS ×2, k=17 | **이진** | |
| Conv2 (epilogue) | k=29, dilation 2, separable | **이진** | |
| Conv3 | 1×1 | INT8 | |
| Conv4 | 1×1 → 12 | **fixed-point** | 마지막 층 이진화 금지 |
| Head | avg pool + softmax | — | |

### 이진 TCS 블록 내부

```
x ∈ {−1,+1}
  → Depthwise conv (이진 가중치)     : XNOR + popcount
  → BatchNorm → Sign                 (pointwise가 XNOR이려면 ±1 입력 필요)
  → Pointwise 1×1 (이진 가중치)      : XNOR + popcount
  → BatchNorm → Sign
  + residual (정수 누산 단계에서 더한 뒤 한 번만 threshold)
```

- 이진 내적 = `2·popcount(XNOR) − N`. **곱셈기 없음.**
- BatchNorm은 추론 시 **정수 threshold 비교 하나로 융합**된다 (Cerutti 식 3).
- 하드웨어 비대칭 주의: XNOR-popcount 이득은 **pointwise(1×1)에서 극대화**되고
  depthwise는 채널을 안 섞어 팩킹 이득이 작다. **두 PE를 통합하지 말 것.**

**결정 기록 (2026-07-21, 사용자 승인)**: Conv2는 기본 **separable**. dense k=29는 conv2
혼자 238K로 논문 전체(93K)를 초과한다. 정확도 미달 시 `separable: false`가 1순위 ablation.

## 3. 프론트엔드 설정 (확정)

```yaml
afe:
  filterbank_source: spice      # 실제 GIC 응답 (mel 삼각형 아님)
  compression: sqrt             # 검출기가 진폭에 선형 → √이 회로 충실
  normalize: xmix               # 회로 형태의 채널간 상대 임계
  xmax_floor_frac: 0.05         # δ = 전형 음성 상승의 2.4%
  envelope_win_ms: 10.0         # native T = 100 → 제로패딩 128
  f_min: 50.0
  f_max: 8000.0
  stft_win_ms: 25.0
  stft_hop_ms: 10.0
  n_fft: 512
```

> **T = 128을 바꾸지 말 것.** conv2(k=29, dilation 2)의 유효 span이 57이고 conv1
> stride 2를 거치면 프레임이 절반이 되므로 **conv1 이후 ≥ 57 프레임**이 필요하다.
> T=96(→48), T=64(→32)는 전부 conv2가 padding을 훑어 성능이 붕괴한다(실측 확인).
> `Config.time_axis_report()`가 이 조건을 경고한다.

## 4. 학습 설정

| 항목 | 값 |
|---|---|
| 데이터셋 | Google Speech Commands **v2**, 12-class, 공식 split |
| epochs | 100 |
| batch size | 128 |
| optimizer | Adam, lr 1e-3 |
| seed | 1234 |
| STE clip | 1.0 |
| 증강 | 없음 (게인 증강은 시도했다가 기각 — [EXPERIMENTS.md](EXPERIMENTS.md)) |

## 5. α는 어떻게 학습되는가

**두 단계다.**

### 5-1. 초기화 — 채널 평균 (Cerutti IV-A)

```python
afe.init_fixed_scale(batch)   # δ (xmax_floor) 결정 — 프레임 분위수
afe.init_thresholds(batch)    # α 초기값 = 채널별 정규화 엔벨로프 평균
```

**순서가 중요하다** — δ가 정해져야 정규화값이 나오고, 그래야 평균을 낼 수 있다.

`xmix`에서는 초기값을 **[0,1]로 클램프**한다. α는 저항비라 그 범위만 만들 수 있고,
클램프가 없으면 이 배치에서 한 번도 기준을 못 넘은 채널이 α = −1로 초기화되어
`sign(0) = +1` 때문에 **전 프레임 발화**한다 (실측: 전체 비트의 4.9%).

### 5-2. 학습 — STE로 end-to-end

임계 비교는 계단 함수라 미분이 0이다. **STE(직통 추정기)** 로 기울기를 흘린다:

```python
out = sign_ste(정규화값 − α, ste_clip)
```

- forward: `sign(·)`
- backward: `|x| < ste_clip` 구간에서만 기울기를 통과

α는 **네트워크 가중치와 동시에** 갱신된다. 즉 네트워크가 "이 채널은 좀 더 예민해야
한다"고 판단하면 α가 내려간다.

### 5-3. 동결

학습이 끝나면 α는 **16개의 상수**다. 체크포인트에 저장되고, 추론 중에는 변하지 않으며,
**저항비로 그대로 납땜된다.**

> **클립마다 변하는 것은 α가 아니라 분모(V_max)** 이고, 그건 회로가 실시간으로 만든다.
> 자세한 구분은 [MATH.md](MATH.md) §5.

채널별 초기값과 실측 통계: [artifacts/channel_table.md](artifacts/channel_table.md).
학습된 최종값 추출: [REPRODUCE.md](REPRODUCE.md) §α 추출.

## 6. 하드웨어 내보내기 (export)

학습이 끝나면 두 가지를 뽑는다.

| 대상 | 결과물 | 상태 |
|---|---|---|
| **α ×16** | 저항비 $R_b/(R_a+R_b)$ → 아날로그 BOM | 셀 있음 |
| **BN → 정수 threshold** | `floor(β′/γ′)` (Cerutti 식 3) → FPGA | [`export/`](../export/) 미구현 |
| **이진 가중치 bit-packing** | XNOR-popcount용 | 미구현 |

α의 스케일 인자(XNOR-Net의 per-output α)는 뒤따르는 BN이 흡수해 정수 threshold로
접히므로 **하드웨어에 남지 않는다.** 단, residual 정수 누산에 들어가는 마지막
pointwise·skip은 스케일 미적용(정수 유지).
