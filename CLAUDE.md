# CLAUDE.md — BinaryMatchboxNet KWS 프로젝트

이 파일은 Claude Code가 매 세션 시작 시 자동으로 읽는 프로젝트 헌장이다.
여기 적힌 아키텍처 결정, 제약, 컨벤션을 **항상** 따른다. 임의로 바꾸지 말고,
바꿔야 할 이유가 생기면 먼저 나에게 물어본다.

---

## 0. 프로젝트 한 줄 요약

Cerutti et al.의 아날로그 프론트엔드(AFE)가 만드는 **이진 시간-주파수 이미지**를
입력으로 받아, **부분 이진화된 MatchboxNet**으로 12-class 키워드 스포팅(KWS)을
수행한다. 최종 목표는 **KC705(XC7K325T) FPGA 구현**이며, **현재 단계는 RTL 설계와
검증**이다. 소프트웨어 모델(정확도)은 `xlse` + `aug_gain_db=[-12,+12]`로 확정됐다.

**결정 기록 (2026-08-15, 사용자 승인):** 데이터패스는 **folded**(MAC 엔진 시분할).
원래 §0은 fully-unrolled였으나, 추론이 10 Hz라 요구 처리량이 **0.6 MAC/cycle**에
불과해 완전 펼침은 과잉이다(KC705 점유 추정 1.3%). folded는 합성·디버깅이 빠르고,
C·T가 바뀌어도 컨트롤러 파라미터만 바뀐다 — 아날로그가 아직 확정 전인 지금에 맞다.

## 1. 목표와 성공 기준

- **정확도 목표**: Google Speech Commands v2, 12-class 기준 **≥ 85%**.
  - 12-class = 10개 키워드(yes, no, up, down, left, right, on, off, stop, go)
    + "silence" + "unknown".
- **전력 목표**: 분류 1회당 ≤ 15 mJ (지금은 추정치로만 리포트, 실측 아님).
  - FPGA 실측이 아니라 **연산 카운트 기반 추정**으로 다룬다.
- **현재 마일스톤**: PyTorch에서 (채널 폭 C, 시간 윈도우 T)를 sweep하여
  85%를 넘기는 **최소 (C, T)** 를 찾는다. 이 값이 이후 하드웨어 크기를 확정한다.

---

## 2. AFE 신호 체인 — 정확한 이해 (필독)

이 절은 이 프로젝트에서 가장 오해하기 쉬운 부분이다. 코드를 쓰기 전에 반드시 읽는다.

> **중요 — 하드웨어 vs 우리 소프트웨어 시뮬**: 이 절은 대부분 *실제 아날로그
> 하드웨어*의 신호 체인을 설명한다. **우리 코드는 그 하드웨어를 스펙트로그램
> 수준에서 시뮬**하며(Cerutti IV-A가 쓴 것과 같은 방법), SPICE급 아날로그 회로
> 시뮬이 아니다. 어떤 개념이 "실제 회로" 얘기이고 어떤 게 "우리 코드" 얘기인지
> 2.10 / 3.2에서 명확히 구분한다. 참고 그림: `docs/diagrams/09_stft_two_paths.svg`,
> `docs/diagrams/10_discretization.svg`.

### 2.1 AFE의 존재 이유

표준 KWS 파이프라인은 `마이크 → ADC(고속 샘플링) → FFT/STFT → Mel → MFCC → NN`이다.
여기서 **전력의 대부분은 분류 연산이 아니라 데이터 취득(ADC)과 전처리(FFT/MFCC)** 가 먹는다.
Cerutti 논문의 핵심 주장이 이것이며, 강하게 최적화된 분류기를 쓰면 분류 에너지는
취득·전처리 다음인 3순위로 밀린다.

**AFE는 이 FFT/MFCC 계산을 아날로그 회로로 대체한다.** 아날로그 대역통과 필터 뱅크가
주파수 분해를 물리적으로 수행하므로 디지털 FFT가 필요 없다.
(논문: 8필터 AFE는 1초 취득에 0.28 mJ, 표준 마이크 5.4 mJ의 5.2%.)

### 2.2 아날로그 신호 체인 (채널 1개 기준)

```
마이크 → GIC 대역통과 필터 → active detector(정밀정류+이득+C3 평활) → comparator → 이진 펄스
   V_in            V_filt                    V_+ (= envelope)              빨간 사각파
```

논문 Fig.1 오른쪽 파형의 세 단이 정확히 이 순서다.

- **V_in (마이크)**: 진폭 약 0.4 mV(그림 판독 추정). DC 바이어스 위 미세 음성 신호.
- **V_filt (필터 후)**: 이 채널 통과대역 성분만 남아 매끄럽고 규칙적. **아직 진동한다.**
  중심주파수는 식 (1) `f_c = 1/(2πR_A·C)`.
- **V_+ (active detector 출력) = envelope**: 정류+평활된 **톱니 파형**.
- **빨간 사각파**: comparator가 `V_+ > V_-`(R7/R8 분압 기준전압)일 때 1.

### 2.3 active detector가 하는 일

회로도의 "active detector" 블록 = op-amp + D1/D2 + R4/R5 + C3.

- **정밀 정류기(precision rectifier)**: 다이오드 순방향 강하(~0.6 V) 때문에 수동 정류로는
  0.4 mV 신호를 못 다룬다. op-amp 피드백 안에 다이오드를 넣어 이를 상쇄한다.
  이것이 "active"의 의미이며, 이 미세 신호에는 **필수**다.
- **이득**: 식 (2) `G = R5/R4`. comparator가 판정 가능한 크기로 증폭.
- **C3 평활**: **빠른 충전 + 느린 방전** → 톱니 모양 envelope.
  신호가 세면 급상승, 약해지면 지수적으로 감쇠한다.

### 2.4 envelope의 정확한 정의

**envelope = 정류 + 평활된 에너지 윤곽.** 개별 진동을 따라가지 않고 "이 대역에 지금
에너지가 있는가"를 나타낸다. 하나의 개념이다.

| 신호 | 성격 | 비고 |
|---|---|---|
| V_filt | **진동하는** 대역 신호 | envelope 아님 (정류 전) |
| V_+ | 정류+평활된 톱니 | **이것이 envelope** |
| 빨간 사각파 | envelope ≷ threshold | 이진 특징 |

threshold 학습에서 말하는 "full-precision 엔벨로프 값"은 **V_+ (연속값)** 을 가리킨다.
우리 코드에서 그 대응물은 log-mel 스펙트로그램의 창별 max값(연속)이다(2.10 참조).

### 2.5 envelope에는 "길이"가 아니라 시상수 τ가 있다

**실제 회로의 envelope는 고정 길이를 갖지 않는다.** C3와 방전 저항이 만드는
**RC 시상수 τ**가 감쇠 속도를 정한다.

- τ가 너무 짧으면: 개별 진동을 따라 출렁여 평활이 안 됨
- τ가 너무 길면: 음소가 뭉개져 시간 해상도 상실
- 타당 범위: 채널 주기(수 ms)보다 길고 음소 길이(수십 ms)보다 짧음 → **수 ms 규모**

**논문에 τ 수치는 없다.** Fig.1 파형에서 하강 구간을 읽어 수 ms로 추정될 뿐이다.
추정임을 항상 명시하고, 논문 수치인 것처럼 쓰지 말 것. (우리 baseline 시뮬에는 τ가
없다 — 3.2 참조.)

### 2.6 시간 척도 계층 (핵심)

```
채널 주기        ~1.5 ms    (예: 650 Hz 대역)
envelope τ       ~수 ms      ← 아날로그 회로가 결정 (논문에 값 없음)
comparator 펄스  <1 ms 폭    ← threshold 교차 순간들
재구성 window    10~25 ms    ← 디지털이 결정 (envelope_win_ms)
음소 길이        ~수십 ms
```

**τ < window** 이므로 하나의 window 안에 펄스가 여러 개 들어간다.
Fig.1의 20 ms 구간에 펄스가 4개(약 1.2 / 4.6 / 13.4 / 18.1 ms, 그림 판독) 찍힌 것이 그 예다.

### 2.7 취득은 샘플링이 아니라 이벤트 기반이다 (실제 하드웨어)

**실제 AFE는 신호를 주기적으로 샘플링하지 않는다.** comparator 출력이 0↔1로 바뀌는
**순간(이벤트)** 만 MCU/FPGA에 인터럽트로 전달된다.

- 평소 sleep → 첫 인터럽트 시점을 **time 0** 으로 잡고 타이머 시작
- 이후 각 채널 이벤트의 **타임스탬프만 기록**
- **1초 후** 모아둔 타임스탬프로 시간-주파수 이미지를 재구성

**논문의 32 kHz는 오디오 샘플링 클럭이 아니다.** 타임스탬프를 재는 **저전력 타이머의
분해능**이다(STM32L476RG). 이벤트가 없으면 아무 동작도 하지 않는다. 이것이
event-based 취득이며 전력 절감의 핵심이다. "32 kHz 샘플링"이라고 쓰지 말 것.
(우리 소프트웨어 시뮬은 이 이벤트 타이밍을 재현하지 않고 스펙트로그램 프레임 수준으로
근사한다 — 2.10 참조.)

### 2.8 이산화: 연속 이벤트 → [16, T] 이진 이미지

아날로그 이벤트는 **서브밀리초 분해능**이다. 신경망에 넣으려면 시간 격자로 뭉쳐야 한다.

- window 폭 = `envelope_win_ms` (기본 10 ms, 겹침 없음)
- 축약 규칙 = `envelope_reduce` (기본 `max`)
  - `max` = 논문의 "그 window 안에서 **한 번이라도** 1이었으면 1"

**구체 예 (Fig.1의 20 ms, 10 ms window):**

| 칸 | 시간 | 포함 펄스 | max 결과 |
|---|---|---|---|
| 0 | 0–10 ms | 1.2, 4.6 ms (2개) | **1** |
| 1 | 10–20 ms | 13.4, 18.1 ms (2개) | **1** |

→ **펄스 개수·발생 시각·지속 시간이 전부 소실되고 "1"만 남는다.** 펄스가 1개여도
결과는 같다. 이산화는 실질적 정보 손실을 동반한다. (그림: `10_discretization.svg`.)

1초 전체: 1000/10 = 100칸 → MatchboxNet 4.1대로 **T = 128** 로 제로패딩.
16채널 각각 동일 처리 → **`[16, 128]` 이진 이미지**가 신경망 입력.

**우리 코드에서의 등가성**: 우리 시뮬은 `max-pool(연속 envelope) → threshold` 순서인데,
max가 단조라 이는 **`threshold(comparator) → OR(=이진 max)`와 수학적으로 동일**하다.
즉 우리 max baseline은 이미 "한 번이라도 1이면 1" 의미를 정확히 구현한다.

**대안 축약(ablation 후보, baseline 아님)**: `count`(칸 내 펄스 개수),
`mean`(활성 시간 비율). 정보를 보존하지만 **입력이 더 이상 이진이 아니게 되어
Conv1의 하드웨어 이점이 줄어든다.** (그리고 max와 달리 threshold와 순서 교환이
성립하지 않는다.) 트레이드오프를 의식하고 실험할 것.

### 2.9 두 종류의 파라미터를 혼동하지 말 것

STFT 파라미터와 envelope 파라미터는 **역할이 다르다.**

| 파라미터 | 역할 | 겹침 |
|---|---|---|
| `stft_win_ms` / `stft_hop_ms` / `n_fft` | STFT 프론트엔드(주파수 분석) | win/hop 겹침 있음 |
| `n_mels` (64) | **경로 A(full-precision baseline)** 의 mel 필터 수 | — |
| `n_channels` (16) | **경로 B(AFE)** 의 필터 수 | — |
| `envelope_win_ms` / `envelope_reduce` | **경로 B**의 이벤트 이산화 칸 | 겹침 없음 |

**중요 (구현 사실)**: 실제 하드웨어에선 아날로그 필터가 주파수를 쪼개므로 경로 B에
STFT가 없다. **그러나 우리 소프트웨어 시뮬은 아날로그 필터뱅크를 STFT+mel로 흉내내므로
`stft_*`는 두 경로가 공유한다.** 두 경로의 실제 차이는 STFT 유무가 아니라 mel 필터 수
(`n_mels`=64 vs `n_channels`=16)와 그 뒤 처리(경로 B만 envelope+threshold+이진화)다.
따라서 `stft_*`를 "경로 A 전용"이라고 표기하지 말 것 — 코드와 어긋난다.

`stft_hop_ms`와 `envelope_win_ms`가 둘 다 10.0인 것은 **우연**이며 서로 독립이다.
T sweep에서 `envelope_win_ms`만 25로 바꿔도 `stft_hop_ms`는 그대로 둘 수 있어야 한다
(현재 코드가 이미 그렇다: `native_T = clip_ms/envelope_win_ms`).

### 2.10 두 경로 요약 (우리 소프트웨어 시뮬 기준)

```
공통 프론트엔드
  오디오 → STFT (stft_win_ms, stft_hop_ms, n_fft)
             │
   경로 A ───┴─→ mel(n_mels=64) → log → NN            (비교용 full-precision baseline)
   경로 B ─────→ mel(n_channels=16) → log
                  → [τ 평활: 2단계 ablation, baseline 비활성]
                  → envelope_win 이산화(envelope_reduce)
                  → 채널별 학습 threshold → 이진화 → NN   (AFE 시뮬 = 실제 우리 모델)
```

경로 B가 우리가 학습·평가하는 실제 모델이다. STFT+mel(16)은 아날로그 대역필터뱅크의
소프트웨어 대응물이다.

---

## 3. 핵심 아키텍처 결정 (변경 금지 — 바꾸려면 먼저 질문할 것)

### 3.1 입력
- AFE 출력 = **16채널 이진 이미지**, shape `[16, T]`.
- 값은 {0,1}이지만 네트워크 내부에서는 **{-1,+1}** 로 인코딩.
- **baseline: `envelope_win_ms = 10 ms` → native T = 100 → 제로패딩 T = 128.**
- **T sweep 주의 (중요)**: 현재 conv2(k=29, dilation=2)의 유효 커널 span은 57이다.
  conv1 stride 2를 거치면 프레임 수가 절반이 되므로, **conv1 이후 프레임 수 ≥ 57**
  이어야 conv2가 정상 동작한다. 즉 **10 ms 기준 T=128(→64프레임)만 유효**하고,
  T=96(→48), T=64(→32), T=40(25 ms, →20)은 전부 conv2가 padding을 훑어 성능이
  붕괴한다(실측 확인). 작은 T를 sweep하려면 conv2 커널/dilation을 먼저 줄여야 한다.
  `Config.time_axis_report()`가 이 조건을 경고한다.
- **AFE는 지금 실제 회로가 아니라 소프트웨어 시뮬레이션**이다(2.10 경로 B).

### 3.2 AFE 시뮬 충실도 단계 (중요)

두 단계로 나누어 진행한다. **1단계를 먼저 확정한 뒤에만 2단계로 간다.**

- **1단계 (baseline, 지금)**: 대역 에너지 → `envelope_win_ms` 이산화 → 학습 threshold 이진화.
  **τ 평활 없음** (`envelope_tau_ms: 0.0` = 비활성). 이 방법은 Cerutti IV-A의 소프트웨어
  AFE 시뮬(스펙트로그램 창별 max)과 동일하다.
- **2단계 (ablation, 나중)**: 위에 **지수 감쇠(EMA) 평활**을 추가해 C3 톱니를 모사.
  `envelope_tau_ms > 0`. (EMA는 STFT 프레임 위에 적용되는 소프트웨어 근사임을 명시.)

**τ를 baseline에 넣지 않는 이유**: (a) 논문에 τ 수치가 없어 임의값이 되고,
(b) 뒤따르는 10 ms + max 이산화가 이미 강한 평활이라 τ < window면 효과가 거의 없으며,
(c) baseline에 임의값이 섞이면 정확도 미달 시 원인 분리가 불가능해진다.
τ는 **재현 파라미터가 아니라 설계 자유도**로 취급한다. 인터페이스만 미리 만들어 둔다.

### 3.3 모델: BinaryMatchboxNet (부분 이진화)

MatchboxNet의 파라미터 효율(1D TCS conv + residual)은 살리되, 연산을 XNOR/popcount
친화적으로 이진화한다. **단, 전부 이진화하지 않는다.** Cerutti Fig.5의 교훈
(입력·양 끝단 이진화 시 정확도 급락)을 반영한다.

MatchboxNet-3x2xC 골격 + 정밀도 오버레이:

| 스테이지 | 종류 | 정밀도 | 비고 |
|---|---|---|---|
| Conv1 (prologue) | k=11, stride=2 | **INT8 가중치, 이진 입력** | 첫 층 이진화 금지 |
| B1 | TCS sub-block ×2, k=13 | **이진** | depthwise + pointwise 모두 |
| B2 | TCS sub-block ×2, k=15 | 이진 | |
| B3 | TCS sub-block ×2, k=17 | 이진 | |
| Conv2 (epilogue) | k=29, dilation=2, separable | 이진 (ablation: int8/dense) | |
| Conv3 | 1×1 | INT8 | |
| Conv4 | 1×1 → 12 classes | **fixed-point** | 마지막 층 이진화 금지 |
| Head | avg pool + softmax | — | |

**결정 기록 (2026-07-21, 사용자 승인):** Conv2는 기본 **separable**. dense k=29는
conv2 혼자 238K로 논문 전체(93K)를 초과하므로 원 논문 epilogue는 separable로 판단.
정확도 미달 시 `separable: false`(dense, 총 324K)가 1순위 ablation.

**절대 규칙:**
1. **첫 층(Conv1)과 마지막 층(Conv4)은 절대 이진화하지 않는다.**
   입력이 이미 이진이므로 Conv1을 INT8로 두어 연속 특징 공간으로 펼쳐 손실을 완충한다.
2. **중간 TCS 블록(B1~B3)만 이진화한다.** 여기가 연산·파라미터의 대부분.
3. Conv1은 이름은 INT8이지만 **입력이 ±1이라 실제로는 곱셈 없이 부호 있는 누산**으로
   구현 가능하다. 하드웨어 리소스 추정 시 이 점을 반영할 것.

### 3.4 이진 TCS sub-block 내부

```
입력 x ∈ {-1,+1}
  → Depthwise conv (채널별, 시간축 커널, 이진 가중치)  : XNOR + popcount
  → BatchNorm → Sign (내부 재이진화; pointwise가 XNOR이려면 ±1 입력 필요)
  → Pointwise 1×1 conv (채널 믹싱, 이진 가중치)        : XNOR + popcount
  → BatchNorm  (학습 후 정수 threshold 하나로 융합)
  → Sign 활성화 → 다음 층 이진 입력 {-1,+1}
  + residual: sub-block 입력을 popcount 정수 누산 단계에서 정수로 더한 뒤 한 번만 threshold
```

- 이진 dot product = `2 * popcount(XNOR) - N`. 곱셈기 없음.
- BatchNorm은 추론 시 정수 threshold 비교 한 번으로 융합(Cerutti 식 3).
  export 단계에서 `floor(beta'/gamma')` 정수 threshold를 미리 계산.
- α(가중치 스케일)는 뒤 BN이 흡수해 정수 threshold로 접히므로 하드웨어에 안 남는다.
  단, residual 정수 누산에 들어가는 마지막 pointwise·skip은 α 미적용(정수 유지).
- **하드웨어 비대칭 주의**: XNOR-popcount의 이득은 "1비트 팩킹 병렬"에서 나오며,
  이는 **pointwise(1×1)에서 극대화**된다. depthwise는 채널을 섞지 않아 팩킹 이득이 작다.
  두 PE를 하나로 통합하지 말 것.

### 3.5 AFE threshold 학습

- 채널마다 threshold **하나씩**(16개). 학습 가능 파라미터.
- 초기화: 채널별 envelope 값의 평균. min-max 정규화 후 threshold도 같은 스케일.
- threshold 비교는 계단 함수라 미분이 0 → **STE 필수**.
- threshold와 네트워크를 **end-to-end 함께 학습**한다.
- **threshold는 정확도 파라미터일 뿐 아니라 이벤트 발생률(=인터럽트=전력) 파라미터**다.
  threshold를 낮추면 펄스가 급증한다. 추후 이벤트 수 정규화 항을 고려할 수 있다.
  (지금은 정확도만 최적화. 이 항목은 메모로만 유지.)

---

## 4. 학습 방법론
- **데이터셋**: Google Speech Commands **v2** (35 words, 105k utterances), 12-class 설정.
  - train:val:test = 80:10:10, 데이터셋 공식 권장 split 사용.
- **전처리(공통 STFT 프론트엔드)**: STFT 25 ms 윈도우 / 10 ms hop. 그 위에서
  - 경로 A(baseline): **64 Mel-filters**, 50–8000 Hz.
  - 경로 B(AFE, 실제 모델): **16채널** 필터, 50–8000 Hz, Mel 도메인 등간격 corner
    frequency. 그 뒤 envelope 이산화 + 채널별 threshold 이진화 → `[16, T]`.
  - 주의: Mel-filter 개수(경로 A=64)와 AFE 채널 수(경로 B=16)는 다르다.
  - `f_max`는 8000 Hz(= Nyquist). Cerutti Fig.6의 최적 범위(50 Hz–8 kHz)에 맞춤.
    (초기 sc_v2 baseline은 7500이었고, 8000으로 올려 정확도가 유의미하게 상승했다.)
- **증강(가능하면)**: time shift ±5 ms, noise, SpecAugment(time/freq mask). 처음엔
  생략 가능하되 85%가 안정적으로 안 나오면 도입. (파형 단계 증강=time-shift/noise가
  AFE 강건성에 특히 의미.)
- **최적화**: 처음엔 Adam(lr 1e-3~1e-4)으로 단순하게 시작. 재현 안정성 확보 후
  필요하면 NovoGrad + Warmup-Hold-Decay로 튜닝.
- **BNN 학습**: Quantization-Aware Training(QAT). 이진 가중치는 latent full-precision을
  유지하고 forward에서 sign, backward는 STE.

## 5. 지금 하지 말 것 (범위 밖)

- **아날로그 파라미터를 RTL에 넣지 말 것** — frac·δ·threshold·f_c·τ는 전부 비교기
  **왼쪽**에 있다. FPGA로 넘어오는 것은 **비교기 16가닥 + 프레임 타이밍뿐**이다
  (`docs/ICD.md`). 아날로그가 바뀌면 바뀌는 것은 **가중치**지 RTL이 아니다.
- **가중치를 RTL에 상수로 박지 말 것** — BRAM에 `$readmemh`로 싣는다. 재학습 때마다
  합성을 다시 돌리게 되면 아날로그 확정 전까지 반복 비용이 감당이 안 된다.
- **골든 벡터 없이 RTL을 쓰지 말 것** — 층별 중간 활성 없이는 디버깅이 불가능하다.
- **실제 아날로그 회로(SPICE) 설계 금지** — AFE는 스펙트로그램 수준 소프트웨어 시뮬로만.
  실물 회로 설계는 **동료 담당**이다.
- **저항 양자화(E12/E24) 재학습 금지** — 실제 AFE 제작 단계에서만.
- **τ 평활을 baseline에 켜지 말 것** — 3.2의 2단계에서만.

## 6. 코드 컨벤션

- 언어: Python 3, PyTorch. 타입 힌트 사용.
- 구조:
  - `data/` — 데이터셋 다운로드·전처리·이진화 파이프라인
  - `models/` — BinaryMatchboxNet, 이진 레이어(BinaryConv, DepthwiseBin, PointwiseBin),
    STE, BN-threshold 융합
  - `train/` — 학습 루프, config, 로깅
  - `experiments/` — (C, T) sweep 스크립트, 결과 집계
  - `export/` — 학습된 모델을 정수 threshold + bit-packed 가중치로 변환(하드웨어 대비)
  - `tests/` — 단위 테스트
- **설정은 하드코딩하지 말고 config(YAML 또는 dataclass)로.** 특히 C, T, 윈도우 크기,
  각 층 커널·정밀도는 전부 config에서 조절 가능해야 한다.
- 재현성: seed 고정, 결과(정확도·config·checkpoint 경로)를 파일로 기록.
- 이진 레이어는 **단위 테스트 필수**: XNOR-popcount dot product가
  일반 ±1 행렬곱과 수치적으로 일치하는지 검증. STE gradient가 흐르는지 검증.
- **AFE 파이프라인 단위 테스트 필수**:
  - 이산화 규칙(2.8 표)을 **독립 함수**로 떼서 검증할 것. 우리 전체 파이프라인은
    STFT 10 ms 프레임 해상도라 서브-ms 펄스열을 그대로 넣을 수 없으므로,
    `discretize(events, window_ms, reduce)` 같은 순수 함수에 2.8의 케이스를 넣는다.
  - `envelope_tau_ms = 0`일 때 τ 평활이 완전히 비활성(항등)인지.

## 7. 작업 방식 (Claude Code에게)

- **큰 작업은 작은 단위로 쪼개고, 각 단계 후 실행·테스트해서 동작을 확인**한 뒤 다음으로.
  한 번에 전체를 쓰고 끝내지 말 것.
- 새 파일을 만들거나 주요 설계를 정할 때는 **왜 그렇게 했는지 짧게 설명**하고,
  이 CLAUDE.md의 규칙과 충돌하면 진행 전에 질문.
- 데이터셋 다운로드처럼 오래 걸리거나 네트워크가 필요한 작업은 먼저 알리고 진행.
- 정확도 결과가 나오면 (C, T, 정확도, 주요 하이퍼파라미터)를 표로 요약.
- 추측으로 논문 수치를 지어내지 말 것. 불확실하면 "확인 필요"로 표시.
  특히 τ 값은 논문에 없다 — 추정임을 명시할 것.

## 8. 참고 논문 (프로젝트 루트 PDF)

- `MatchboxNet.pdf` — Majumdar & Ginsburg, 1D TCS conv KWS 아키텍처. 골격·커널·학습 설정 출처.
- `Sub-mW_Keyword_Spotting...pdf` — Cerutti et al., AFE + BNN. 이진 입력·threshold 학습·
  BN 융합(식 3)·12-class 정확도 baseline(8ch 76.3%, 64ch 86.0%)·AFE 소프트웨어 시뮬
  방법(IV-A: 스펙트로그램 창별 max) 출처.

이 두 논문의 수치를 인용할 때는 위 파일을 근거로 삼는다.
