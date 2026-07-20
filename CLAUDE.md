# CLAUDE.md — BinaryMatchboxNet KWS 프로젝트

이 파일은 Claude Code가 매 세션 시작 시 자동으로 읽는 프로젝트 헌장이다.
여기 적힌 아키텍처 결정, 제약, 컨벤션을 **항상** 따른다. 임의로 바꾸지 말고,
바꿔야 할 이유가 생기면 먼저 나에게 물어본다.

---

## 0. 프로젝트 한 줄 요약

Cerutti et al.의 아날로그 프론트엔드(AFE)가 만드는 **이진 시간-주파수 이미지**를
입력으로 받아, **부분 이진화된 MatchboxNet**으로 12-class 키워드 스포팅(KWS)을
수행한다. 최종 목표는 FPGA(완전 펼침, fully-unrolled 파이프라인) 구현이지만,
**현재 단계는 PyTorch 소프트웨어 모델의 학습과 정확도 검증**이다.

## 1. 목표와 성공 기준

- **정확도 목표**: Google Speech Commands v2, 12-class 기준 **≥ 85%**.
  - 12-class = 10개 키워드(yes, no, up, down, left, right, on, off, stop, go)
    + "silence" + "unknown".
- **전력 목표**: 분류 1회당 ≤ 15 mJ (지금은 추정치로만 리포트, 실측 아님).
  - FPGA 실측이 아니라 **연산 카운트 기반 추정**으로 다룬다.
- **현재 마일스톤**: PyTorch에서 (채널 폭 C, 시간 윈도우 T)를 sweep하여
  85%를 넘기는 **최소 (C, T)** 를 찾는다. 이 값이 이후 하드웨어 크기를 확정한다.

## 2. 핵심 아키텍처 결정 (변경 금지 — 바꾸려면 먼저 질문할 것)

### 2.1 입력
- AFE 출력 = **16채널 이진 이미지**, shape `[16, T]`.
  - 16 = 주파수 채널(필터 뱅크), T = 시간 윈도우 수.
  - 값은 {0, 1}이지만 네트워크 내부에서는 **{-1, +1}**로 인코딩해서 다룬다.
- T는 실험 변수. sweep 대상: **T ∈ {40, 64, 96, 128}**, 윈도우 {10 ms, 25 ms}.
- **AFE는 지금 실제 회로가 아니라 소프트웨어 시뮬레이션**으로 만든다:
  MFCC/log-Mel 스펙트로그램을 계산 → 채널별 학습 가능 threshold로 이진화.

### 2.2 모델: BinaryMatchboxNet (부분 이진화)
MatchboxNet의 파라미터 효율(1D time-channel separable conv + residual)은 살리되,
연산을 XNOR/popcount 친화적으로 이진화한다. **단, 전부 이진화하지 않는다.**
Cerutti Fig.5의 교훈(입력·양 끝단 이진화 시 정확도 급락)을 반영한다.

기본 골격은 MatchboxNet-3x2xC (B=3 blocks, R=2 sub-blocks, 채널 폭 C):

| 스테이지 | 종류 | 정밀도 | 비고 |
|---|---|---|---|
| Conv1 (prologue) | 1D conv, k=11, stride=2 | **INT8 가중치, 이진 입력** | 첫 층 이진화 금지 |
| B1 | TCS sub-block ×2, k=13 | **이진 (XNOR/popcount)** | depthwise + pointwise 모두 이진 |
| B2 | TCS sub-block ×2, k=15 | 이진 | |
| B3 | TCS sub-block ×2, k=17 | 이진 | |
| Conv2 (epilogue) | 1D conv, k=29, dilation=2 | 이진 or INT8 (ablation) | 리소스 초과 시 첫 축소 후보 |
| Conv3 | 1×1 conv | INT8 | |
| Conv4 | 1×1 conv → 12 classes | **fixed-point** | 마지막 층 이진화 금지 |
| Head | avg pool + softmax | — | |

**절대 규칙:**
1. **첫 층(Conv1)과 마지막 층(Conv4)은 절대 이진화하지 않는다.**
   입력이 이미 이진이라, Conv1을 INT8로 두어 연속 특징 공간으로 펼쳐 정보 손실을 완충한다.
   Conv4는 fixed-point로 12-class 분리력을 확보한다.
2. **중간 TCS 블록(B1~B3)만 이진화한다.** 여기가 연산·파라미터의 대부분.

### 2.3 이진 TCS sub-block 내부
```
입력 x ∈ {-1,+1}
  → Depthwise conv (채널별, 시간축 커널, 이진 가중치)  : XNOR + popcount
  → Pointwise 1×1 conv (채널 믹싱, 이진 가중치)        : XNOR + popcount
  → BatchNorm  (학습 후 정수 threshold 하나로 융합)
  → Sign 활성화 → 다음 층 이진 입력 {-1,+1}
  + residual: sub-block 입력을 popcount 정수 누산 단계에서 정수로 더한 뒤 한 번만 threshold
```
- 곱셈기 없음: XNOR로 곱, popcount로 누산.
  이진 dot product = `2 * popcount(XNOR) - N`.
- BatchNorm은 추론 시 **정수 threshold 비교 한 번**으로 융합
  (Cerutti 식 3 방식). 학습 후 export 단계에서 `floor(beta'/gamma')` 정수 threshold를 계산.
- Dropout은 학습 시에만.

### 2.4 AFE threshold 학습
- 채널마다 threshold **하나씩** (16개). 학습 가능 파라미터.
- 초기화: 각 채널 엔벨로프 값의 평균. min-max 정규화 후 threshold도 같은 스케일로.
- threshold 비교는 계단 함수 → 미분 0 → **STE(straight-through estimator)** 필수.
  forward는 하드 이진화, backward는 gradient 통과(또는 threshold 주변 완화 근사).
- threshold와 네트워크는 **end-to-end로 함께 학습**한다. threshold만 따로 고정하면 차선.

## 3. 학습 방법론 (Cerutti / MatchboxNet 논문 기준, 재현 가능하게)

- **데이터셋**: Google Speech Commands **v2** (35 words, 105k utterances), 12-class 설정.
  - train:val:test = 80:10:10, 데이터셋 공식 권장 split 사용.
- **전처리**: log-Mel / MFCC, 64 Mel-filters, 50–7500 Hz, 25 ms 윈도우 / 10 ms hop
  (윈도우/hop은 T sweep과 연동). 그 뒤 채널별 threshold로 이진화 → `[16, T]`.
  - 주의: Mel-filter 개수(64)와 AFE 채널 수(16)는 다르다. 16채널 AFE를 시뮬할 때
    Mel 도메인에서 등간격 16개 corner frequency로 필터 뱅크를 구성한다.
- **증강(가능하면)**: time shift ±5 ms, SpecAugment(time/freq mask). 처음엔 생략 가능하되
  85%가 안 나오면 도입.
- **최적화**: 처음엔 Adam(lr 1e-3~1e-4)으로 단순하게 시작. 재현 안정성 확보 후
  필요하면 NovoGrad + Warmup-Hold-Decay로 튜닝.
- **BNN 학습**: Quantization-Aware Training(QAT). 이진 가중치는 latent full-precision을
  유지하고 forward에서 sign, backward는 STE.

## 4. 지금 하지 말 것 (범위 밖)

- **실제 RTL/Verilog/HDL 작성 금지** — 현재 마일스톤은 PyTorch 소프트웨어뿐.
- **실제 아날로그 회로 설계 금지** — AFE는 소프트웨어 시뮬로만.
- **저항 양자화(E12/E24) 재학습 금지** — 실제 AFE 제작 단계에서만.
- FPGA 보드는 아직 미확정. **완전 펼침이 리소스에 들어간다고 가정**하고 진행.

## 5. 코드 컨벤션

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

## 6. 작업 방식 (Claude Code에게)

- **큰 작업은 작은 단위로 쪼개고, 각 단계 후 실행·테스트해서 동작을 확인**한 뒤 다음으로.
  한 번에 전체를 쓰고 끝내지 말 것.
- 새 파일을 만들거나 주요 설계를 정할 때는 **왜 그렇게 했는지 짧게 설명**하고,
  이 CLAUDE.md의 규칙과 충돌하면 진행 전에 질문.
- 데이터셋 다운로드처럼 오래 걸리거나 네트워크가 필요한 작업은 먼저 알리고 진행.
- 정확도 결과가 나오면 (C, T, 정확도, 주요 하이퍼파라미터)를 표로 요약.
- 추측으로 논문 수치를 지어내지 말 것. 불확실하면 "확인 필요"로 표시.

## 7. 참고 논문 (프로젝트 루트의 PDF)

- `MatchboxNet.pdf` — Majumdar & Ginsburg, 1D TCS conv KWS 아키텍처. 골격·커널·학습 설정 출처.
- `SubmW_Keyword_Spotting...pdf` — Cerutti et al., AFE + BNN. 이진 입력·threshold 학습·
  BN 융합(식 3)·12-class 정확도 baseline(8ch 76.3%, 64ch 86.0%) 출처.

이 두 논문의 수치를 인용할 때는 위 파일을 근거로 삼는다.
