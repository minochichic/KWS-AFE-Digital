# BinaryMatchboxNet — 컨볼루션 연산 구조 상세

이 문서는 base 설정(**C=64, T=128**, 총 **96,524 파라미터**) 기준으로, AFE 입력부터
분류기까지 각 스테이지가 **어떤 정밀도로 무슨 연산을 하는지** 정리한다. 다이어그램은
[`docs/diagrams/`](diagrams/)에 SVG로 있다.

| 그림 | 파일 |
|---|---|
| 전체 파이프라인 개요 | [`01_pipeline_overview.svg`](diagrams/01_pipeline_overview.svg) |
| 이진 TCS 블록 내부 | [`02_tcs_block.svg`](diagrams/02_tcs_block.svg) |
| 이진 dot product (XNOR/popcount) | [`03_binary_conv_primitive.svg`](diagrams/03_binary_conv_primitive.svg) |
| INT8 fake-quant 층 | [`04_int8_conv.svg`](diagrams/04_int8_conv.svg) |
| α(가중치 스케일) 적용 위치 지도 | [`05_alpha_scaling_map.svg`](diagrams/05_alpha_scaling_map.svg) |
| sub-block 전 과정 (아이소메트릭, 채널별 α) | [`06_subblock_pipeline.svg`](diagrams/06_subblock_pipeline.svg) |

근거: `models/binary_ops.py`, `models/quant_ops.py`, `models/binary_matchboxnet.py`,
`data/afe.py`, `configs/base.yaml`. 논문 근거는 `CLAUDE.md`와 두 PDF.

---

## 0. 한눈에 보는 데이터 흐름

```
원시 파형 [B,16000] @16kHz
  └▶ AFE (STFT→16 mel→log→엔벨로프→min-max→채널별 threshold) ─▶ [B,16,128] ∈ {-1,+1}
      └▶ Conv1  INT8   k=11,s2   16→128   ─BN─sign─▶ [B,128,64] {-1,+1}
          └▶ B1  이진 TCS×2 +res  k=13   128→64   ─▶ [B,64,64] {-1,+1}
              └▶ B2  이진 TCS×2 +res  k=15   64→64   ─▶ [B,64,64] {-1,+1}
                  └▶ B3  이진 TCS×2 +res  k=17  64→64  ─▶ [B,64,64] {-1,+1}
                      └▶ Conv2  이진 separable  k=29,dil2  64→128  ─BN─ReLU─▶ [B,128,64] 연속
                          └▶ Conv3  INT8  1×1  128→128  ─BN─ReLU─▶ [B,128,64] 연속
                              └▶ Conv4  fixed  1×1  128→12  ─▶ [B,12,64]
                                  └▶ avg pool(시간) ─▶ [B,12] 로짓
```

| 스테이지 | 종류 | 가중치 정밀도 | 커널 | 출력 ch | 출력 shape | params |
|---|---|---|---|---|---|---|
| Conv1 | 1D conv, stride 2 | **INT8** | 11 | 128 | [B,128,64] | 22,784 |
| B1 | TCS ×2 +residual | **이진** | 13 | 64 | [B,64,64] | 23,616 |
| B2 | TCS ×2 +residual | **이진** | 15 | 64 | [B,64,64] | 10,624 |
| B3 | TCS ×2 +residual | **이진** | 17 | 64 | [B,64,64] | 10,880 |
| Conv2 | separable conv, dil 2 | **이진** | 29 | 128 | [B,128,64] | 10,432 |
| Conv3 | 1×1 conv | **INT8** | 1 | 128 | [B,128,64] | 16,640 |
| Conv4 | 1×1 conv | **fixed-point** | 1 | 12 | [B,12,64] | 1,548 |
| Head | avg pool + softmax | — | — | 12 | [B,12] | 0 |

이진 스테이지(B1~B3+Conv2)가 전체 파라미터의 **58%**. 시간축은 Conv1의 stride 2에서
128→64로 한 번 줄고 이후 유지, 마지막에 pooling으로 1이 된다.

---

## 1. 입력: AFE (그림 01의 파란 블록)

파형을 1비트 시간-주파수 이미지로 바꾼다. `data/afe.py`.

1. **STFT** — 25 ms 윈도우 / 10 ms hop, n_fft 512
2. **Mel 필터뱅크 16개** — 삼각 필터, 50–7500 Hz를 mel 도메인에서 등간격.
   (분석용 64-mel이 아니라 AFE 필터 수 16개를 직접 구성. CLAUDE.md 3)
3. **log** 압축
4. **엔벨로프** — 10 ms 윈도우 내 **max** → native_T = 100 (Cerutti IV-A)
5. **per-clip 전역 min-max** → [0,1] (채널별 아님: 채널 간 레벨 차이를 보존해야
   threshold가 그 차이를 흡수)
6. **채널별 threshold 비교** — 16개 학습가능 threshold, `sign-STE`로 {-1,+1}
7. **128 프레임으로 zero-pad** (MatchboxNet 4.1)

출력 **[B, 16, 128] ∈ {-1,+1}**. threshold 16개는 네트워크와 **end-to-end 동시 학습**되고,
첫 학습 배치의 채널별 평균 엔벨로프로 초기화한다(Cerutti IV-A).

---

## 2. Conv1 / Conv3 — INT8 (그림 04)

**첫 층·마지막 층은 절대 이진화하지 않는다**(CLAUDE.md 2.2). Conv1은 이미 1비트인
입력을 128채널 연속공간으로 펼쳐 정보 손실을 완충한다. `models/quant_ops.py`.

**QAT fake-quantization** (forward에서만 양자화, latent 실수 가중치 유지):

```
s   = max|Wₒ| / 127            # 출력채널별 대칭 스케일
q   = clip(round(W / s), ±127) # 8-bit 정수 격자 (255 레벨)
W_q = q · s                    # 역양자화, 양자화 오차 ≤ s/2
```

- **backward: round-STE** — `round`를 항등으로 통과시켜 latent W가 연속으로 유지되며
  미세 갱신이 누적된다.
- **채널별 스케일** — 출력 필터마다 s 하나. 필터별 크기 차이를 반영.
- Conv1 흐름: `[B,16,128]{-1,+1} → conv(int8) → [B,128,64] 연속 → BN → sign → {-1,+1}`.
  즉 Conv1의 **가중치는 8비트지만 출력은 다음 이진 스테이지(B1)를 위해 다시 sign으로
  1비트화**된다. 핵심은 conv 연산 자체가 연속 정밀도로 128채널을 학습한다는 점.

검증: `tests/test_quant_ops.py` (정수 격자 위치, 오차 상한 s/2, STE, 채널별 스케일).

---

## 3. 이진 dot product — 모든 이진 conv의 핵심 (그림 03)

이진 가중치·이진 입력의 내적을 곱셈기·누산기 없이 계산한다. `models/binary_ops.py`.

```
x, w ∈ {-1,+1}
x̂ = (x+1)/2 ∈ {0,1}                      # -1을 0으로 인코딩
Σ xᵢ·wᵢ  ≡  2·popcount( XNOR(x̂, ŵ) ) − N   # N = 누산 탭 수
```

즉 **곱셈 = XNOR, 누산 = popcount**. 예시(N=5):

| | 탭1 | 탭2 | 탭3 | 탭4 | 탭5 | |
|---|---|---|---|---|---|---|
| x | +1 | −1 | +1 | +1 | −1 | |
| w | +1 | +1 | −1 | +1 | −1 | |
| x·w | +1 | −1 | −1 | +1 | +1 | Σ = **+1** |
| XNOR(x̂,ŵ) | 1 | 0 | 0 | 1 | 1 | popcount = **3** |

`2·3 − 5 = +1` — 일반 ±1 내적과 정확히 일치.

- **하드웨어**: 32채널을 32-bit 정수로 팩킹. XNOR 미지원 플랫폼은 `32 − 2·popcnt(x̂⊕ŵ)`.
  FPGA 완전 펼침은 곱셈기 0개, LUT popcount + 비교기.
- **소프트웨어**: `F.conv1d`(float)로 계산하되 값은 위 정수와 **비트-정확 동일**.
  `tests/test_binary_ops.py`가 bool/popcount 레퍼런스와 dense·depthwise·pointwise
  모두 일치함을 검증 → 학습된 SW 모델과 FPGA가 어긋나지 않음을 보장.

### 가중치 스케일 α (선택) — 그림 05

`sign()`이 버린 크기를 보정. `α = mean|W|`(출력채널별)이 `‖W − α·sign(W)‖²`를
최소화하는 최적값(XNOR-Net). 전체 출력 = `α · (2·popcount − N)`.

**지배 규칙**: 출력을 바로 뒤 BN이 정규화하면 α 적용(BN이 흡수 → export 시 소멸, 공짜);
출력이 정수 residual 누산기로 가면 α 끔(정수 유지 필요).

base 모델의 이진 conv 15개 중 (그림 05):

- **α 적용 (11)**: `b1/b2/b3` 각 `subs.0.dw`, `subs.0.pw`, `subs.1.dw` + `conv2.dw`, `conv2.pw`
- **α 없음 (4)**: `b1/b2/b3` `subs.1.pw`(각 블록 마지막 pointwise) + `b1.skip`(투영).
  B2·B3 skip은 identity라 conv 없음.
- **무관**: `conv1`·`conv3`(INT8, 자체 스케일 `s=max|W|/127`은 α와 별개), `conv4`(fixed).

**핵심**: 마지막 sub-block에서도 depthwise(`subs.1.dw`)는 α 적용. dw 출력은 residual로
직행하지 않고 `내부 BN→sign→pw`를 거치므로 내부 BN이 흡수하기 때문. 오직 마지막
pointwise(`subs.1.pw`)만 정수 누산기로 들어가 `scale=False`(§4).

`scale` 플래그 위치: `_TCSSub`(마지막 pw = `scale_binary_weights and not final`),
`BinaryTCSBlock`(skip = 항상 `scale=False`).

### α는 하드웨어 연산이 아니다 (그림 06 확대 패널)

- **계산·곱셈은 학습 소프트웨어에만 존재**: `binary_weight()`에서 `α=mean|W|`를 latent
  실수 가중치로부터 계산해 `sign(W)`에 곱함. latent 가중치는 학습 중에만 존재.
- **추론/FPGA에는 α가 없음**: α는 BN의 γ,β와 함께 export 시 **정수 threshold 하나로
  융합**(Cerutti 식 3). 하드웨어는 `popcount → 정수 비교`만 하며, α 곱셈기도 α 저장도
  없다(비용 0). (이 융합=export 단계는 아직 미구현, `export/`는 다음 마일스톤용 자리.)

### threshold 융합 식 유도 (scale=True vs False)

기호 (출력채널 `o`):

- `aₒ = 2·popcount(XNOR) − N` — 정수 누산기 (하드웨어가 popcount로 계산)
- BN 추론 파라미터: `μₒ`(mean), `sₒ = √(Varₒ+ε)`, `γₒ`(scale), `βₒ`(shift)
- 활성: `sign(BN(z))`, `z` = conv 출력

`BN(z) = γₒ·(z − μₒ)/sₒ + βₒ = 0` 인 지점 (conv 출력 z 기준 임계값):

```
z*ₒ = μₒ − βₒ·sₒ / γₒ
```

`z`가 α 스케일 여부에 따라 달라지므로, **정수 누산기 aₒ 기준** threshold도 갈린다:

| | conv 출력 z | aₒ 기준 threshold |
|---|---|---|
| **scale=True**  | `z = αₒ·aₒ` | `Tₒ = z*ₒ / αₒ`  (α로 나눔) |
| **scale=False** | `z = aₒ`    | `Tₒ = z*ₒ`       (안 나눔) |

`αₒ > 0`이라 부등호 방향은 안 바뀌고 `γₒ`의 부호만 방향을 정하므로, 하드웨어 비교는:

```
outₒ = +1  ⟺  sgn(γₒ)·aₒ  ≥  sgn(γₒ)·⌊Tₒ⌋
```

`aₒ`는 정수, `⌊Tₒ⌋`는 학습 후 미리 계산해 둔 정수 상수 → `popcount → 정수 비교` 한 번
(Cerutti 식 3, {-1,+1} vs {0,1} 인코딩 차이만 있음).

**왜 "BN이 α를 흡수"인가**: scale=True로 학습하면 BN이 `αₒ·aₒ`를 정규화하므로 통계가
α를 머금는다 (`μₒ ≈ αₒ·mean(a)`, `sₒ ≈ αₒ·std(a)`). 이를 `Tₒ = z*ₒ/αₒ`에 넣으면
`Tₒ ≈ mean(a) − (βₒ/γₒ)·std(a)` 로 **α가 상쇄**된다. 결국 정수 threshold는 α 유무와
사실상 같아지고, α는 학습 시 gradient 스케일에만 영향을 준다.

---

## 4. 이진 TCS 블록 B1~B3 (그림 02)

MatchboxNet의 1D time-channel separable conv를 이진화. 각 블록 = **R=2 sub-block**
+ 블록 전체를 감싸는 residual. `models/binary_matchboxnet.py`.

### sub-block 내부

```
x{-1,+1} ─▶ Depthwise(이진,k) ─▶ BN ─▶ sign ─▶ Pointwise(이진,1×1) ─▶ [정수 누산기]
                                                    │(마지막 sub-block만) + residual
                                                    ▼
                                                   BN ─▶ sign ─▶ {-1,+1}
```

- **Depthwise**: 채널별 시간축 conv(groups=채널수), 이진. XNOR+popcount.
- **Pointwise**: 1×1 채널 믹싱, 이진. XNOR+popcount.
- **내부 BN→sign**은 헌장 2.3의 "dw·pw 모두 XNOR/popcount" 요구의 필연적 귀결이다:
  pointwise가 XNOR이려면 입력이 {-1,+1}이어야 하는데 depthwise 출력은 정수 누산기이므로,
  사이에 반드시 이진화가 들어간다.

### residual (그림 02의 빨간 경로)

- 블록 입력에서 분기 → **마지막 sub-block의 정수 누산기에 합류**한 뒤, 그 뒤에서
  **BN→sign 단 한 번**. → 하드웨어에서 threshold가 한 번만 필요(CLAUDE.md 2.3).
- **B1**: 채널 128→64로 바뀌므로 skip = 이진 1×1 투영(`scale=False`), 출력도 정수 popcount.
- **B2·B3**: 채널 동일(64→64) → skip = **identity**(원시 ±1을 정수로 그대로 가산).
- 마지막 pointwise와 skip이 `scale=False`인 이유: α(예: 0.37)를 곱하면 누산기가
  비정수가 되어 "정수+정수" 덧셈과 threshold 융합이 깨진다. 테스트로 잠금
  (`test_residual_accumulator_is_integer_domain`, `test_final_pw_and_skip_are_unscaled`).

### export 시 (FPGA)

BN의 γ,β와 α는 학습 후 `floor(β′/γ′)` **정수 threshold 하나로 융합**(Cerutti 식 3).
블록의 하드웨어 연산 = `popcount → threshold → popcount → 정수덧셈(+res) → threshold`,
**곱셈기 0개**.

---

## 5. Conv2 — 이진 separable epilogue

- separable k=29, dilation 2, 64→128. `separable: true`가 **기본**(2026-07-21 사용자 승인):
  dense k=29는 conv2 혼자 238K로 MatchboxNet 전체(93K)를 초과하므로 원 논문 epilogue는
  separable로 판단. separable 기준 총 96.5K로 논문과 일치.
  정확도 미달 시 `separable: false`(dense, 총 324K)가 1순위 ablation.
- 활성: 다음 스테이지(Conv3)가 비이진이므로 **sign이 아니라 ReLU** → 여기서부터 연속
  특징공간. Cerutti IV-D가 마지막 이진 층의 값을 그대로 분류기에 넘기는 방식과 일치.

---

## 6. Conv3 / Conv4 / Head

- **Conv3**: 1×1 INT8, 128→128, BN→ReLU. conv2의 연속 출력을 섞음.
- **Conv4**: 1×1 **fixed-point**(학습 중 full-precision), 128→12. 마지막 층 이진화 금지.
  고정소수점 변환은 export 단계 소관.
- **Head**: 시간축 adaptive avg pool → [B,12] 로짓. softmax는 CrossEntropyLoss 내부.

---

## 7. 활성 함수 전이 규칙

`PlainStage`는 **다음 스테이지의 정밀도**로 활성을 고른다:

- 다음이 **이진** → `sign`(1비트) : Conv1(→B1), 각 이진 sub-block 내부
- 다음이 **비이진** → `ReLU`(연속) : Conv2(→Conv3), Conv3(→Conv4)
- **마지막**(Conv4) → 활성·BN 없음, bias만 → 로짓

이 규칙이 "가운데만 이진, 양 끝단은 연속"을 자동으로 만든다.

---

## 8. 정밀도·비트수 요약

| 위치 | 값 도메인 | 비트 |
|---|---|---|
| AFE 출력 (모델 입력) | {-1, +1} | 1 |
| Conv1 가중치 | int8 격자 | 8 |
| B1~B3 가중치·활성 | {-1, +1} | 1 |
| Conv2 가중치 | {-1, +1} | 1 |
| Conv2~Conv3 활성 | 연속(ReLU) | fp/fixed |
| Conv3 가중치 | int8 격자 | 8 |
| Conv4 가중치 | full-precision(학습)→fixed(export) | — |
| 로짓 | 연속 | fp |

**절대 규칙**(config `validate()`가 로드 시점에 강제): 첫 층(Conv1)·마지막 층(Conv4)은
절대 binary로 설정할 수 없다. 이를 어긴 YAML은 `ValueError`로 실패한다.
