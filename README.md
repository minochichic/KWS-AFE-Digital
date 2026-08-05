# KWS-AFE-Digital — BinaryMatchboxNet

Cerutti et al.의 아날로그 프론트엔드(AFE)가 만드는 **이진 시간-주파수 이미지**
`[16, T]`를 입력으로 받아, **부분 이진화된 MatchboxNet**으로 12-class 키워드
스포팅을 수행한다. 최종 목표는 FPGA 완전 펼침 구현이지만 현재 마일스톤은
**PyTorch 소프트웨어 모델의 학습과 정확도 검증**이다.

설계 결정과 제약은 [`CLAUDE.md`](CLAUDE.md)에 있다 — 그쪽이 프로젝트 헌장이다.

## 현재 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 디렉터리 구조, config 레이어, Colab 부트스트랩 | ✅ 완료 |
| 2 | 이진 연산 원자(`sign_ste`, `BinaryConv1d`) + 단위 테스트 | ✅ 완료 |
| 3 | AFE 이진화 모듈 (학습 가능 threshold) | ✅ 완료 |
| 4 | BinaryMatchboxNet 조립 | ✅ 완료 |
| 5 | 학습 루프 + 합성 데이터 오버핏 검증 | ✅ 완료 |
| 6 | Speech Commands v2 파이프라인, (C, T) sweep | ✅ 완료 |
| 7 | Phase B — SPICE 실측 필터뱅크 접목 | ✅ 완료 |
| 8 | 정규화 확정 (`xmax` floor=0.05) + 손실 완전 분해 | ✅ 완료 — [Stage 3](docs/experiments_log.md) |
| 9 | 채널당 비교기 2개(2비트)로 −8pp 비교기 손실 공략 | 🔜 다음 |

**현재 정확도 (12-class test)**

| 구성 | test | 비고 |
|---|---:|---|
| 64채널 **연속** log-mel | 0.920 | 네트워크 내부는 이진 — **모델은 문제없음** |
| 16채널 **연속** spice+√+xmax | 0.862 | 연속값 16채널 **천장** |
| **16채널 이진 = 실제 모델** | **0.781** | ← 현재. 목표 0.85 |

16채널 78.1%는 Cerutti의 채널 수 곡선(8ch 76.3% → 64ch 86.0%)이 예측하는 값에서
1.4pp 안쪽이다. **16채널이 부족한 것이지 구현이 틀린 게 아니다.**

## 개발 환경

**로컬(맥)** — 코드 작성, 단위 테스트. GPU 불필요.

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest
```

**Colab Pro** — 실제 학습. [`docs/colab_setup.md`](docs/colab_setup.md)의
GitHub PAT + Colab 시크릿 셋팅을 한 번 끝낸 뒤
[`notebooks/colab_bootstrap.ipynb`](notebooks/colab_bootstrap.ipynb)를 연다.
셀을 위에서부터 실행하면 clone/sync → 의존성 → 테스트 → 데이터셋 다운로드
(~2.3GB, `/content`) → 학습 → (C,T) sweep 순으로 진행된다.

흐름: **로컬에서 수정 → push → Colab에서 부트스트랩 셀 재실행 → 학습.**
Colab 워킹 카피는 매 실행마다 `git reset --hard` 되므로 거기서 소스를 고치지 않는다.

주요 명령 (Colab, `data.root`은 dotted override로 전달):

```bash
# 단일 학습 (base config, C=64 T=128)
python -m train.train --config configs/base.yaml \
    data.root=/content/datasets/speech_commands_v2 --tag sc_v2

# (C, T) sweep — 85% 넘기는 최소 (C,T) 탐색
python -m experiments.sweep --config configs/base.yaml \
    --C 16 32 48 64 --T 40 64 96 128 --epochs 100 \
    data.root=/content/datasets/speech_commands_v2
```

## 디렉터리

**소프트웨어 (PyTorch) — 현재 작업 중**

```
configs/      YAML 설정. C, T, 커널, 정밀도 — 모든 크기의 단일 출처
data/         데이터셋·전처리·AFE 시뮬 + 이진화 파이프라인
models/       BinaryMatchboxNet, 이진 레이어, STE, BN-threshold 융합
train/        config 정의, 학습 루프, 로깅
experiments/  (C, T) sweep, 발화율 측정 등
export/       정수 threshold + bit-packed 가중치 변환 (하드웨어 대비)
tests/        단위 테스트 (165개)
notebooks/    Colab 부트스트랩
```

**아날로그 / 문서 / 참고자료**

```
proposal/     ⭐ 설계 확정안 — 회로도, 수식, 채널 표, 재현 (여기부터)
analog/       AFE 회로 설계·시뮬·튜닝 전부. 지도는 analog/README.md
                ↳ analog/AFE/artifacts/filterbank_matrix.csv 를 학습이 직접 읽는다
                  (아날로그↔ML 유일한 접점. 경로는 train/config.py 한 곳)
docs/         설계 해설과 실험 기록 (아래 표)
papers/       참고 논문 PDF
troubleshooting/  ML·FPGA·아날로그 교차 검토 노트 + 연산량 추정 스크립트
```

```
datasets/     (gitignore) 다운로드된 오디오
runs/         (gitignore) 체크포인트·로그
```

## 문서

| 문서 | 내용 |
|---|---|
| **[`proposal/`](proposal/)** | **⭐ 설계 확정안 — 여기부터 읽으면 된다.** 왜 이 방향인가, 아날로그 회로도, 디지털 모델, 채널별 수치, 재현 |
| [`docs/experiments_log.md`](docs/experiments_log.md) | **실험 기록 — 여기부터 읽으면 된다.** 무엇을 시도했고 무엇이 왜 실패했는가 |
| [`analog/README.md`](analog/README.md) | 아날로그 폴더 지도 — 무엇이 확정이고 무엇이 기각된 탐색인가, 확정 실측값 표 |
| [`docs/xmax_normalization.md`](docs/xmax_normalization.md) | **채널간 상대 임계(`xmax`)와 `floor`** — 음량 불변성을 회로 변경 없이 얻는 방법, 실측, 하드웨어 대응 |
| [`docs/normalization.md`](docs/normalization.md) | global min-max 정규화가 정확히 무엇을 하는가 (그리고 왜 하드웨어로는 불가능한가) |
| [`docs/afe_circuit_explained.md`](docs/afe_circuit_explained.md) | GIC 필터 → 능동 검출기 → 비교기, 기초부터. 모든 수치는 ngspice 실측 |
| [`docs/afe_config.md`](docs/afe_config.md) | AFE 관련 config 필드 레퍼런스 |
| [`docs/conv_structure.md`](docs/conv_structure.md) | 각 층의 커널·정밀도·수용 영역 |
| [`docs/colab_setup.md`](docs/colab_setup.md) / [`docs/remote_gpu_setup.md`](docs/remote_gpu_setup.md) | 학습 환경 |

## 설정 다루기

크기·정밀도는 **코드에 하드코딩하지 않는다**. 전부 `configs/base.yaml`에서 온다.
빌드될 모델을 먼저 확인:

```bash
python3 experiments/inspect_config.py configs/base.yaml
python3 experiments/inspect_config.py configs/base.yaml model.C=16 model.T=96 afe.envelope_win_ms=10.0
```

`train/config.py`의 `validate()`는 CLAUDE.md 2.2의 절대 규칙을 강제한다 —
첫 층(conv1)이나 마지막 층(conv4)을 binary로 바꾼 YAML은 **로드 시점에 실패**한다.

## 논문 근거 수치

[`papers/`](papers/)의 두 PDF에서 확인한 값. 인용 시 여기를 근거로 삼는다.

| 항목 | 값 | 출처 |
|---|---|---|
| 이진화 ablation (Mel 입력) | baseline 93.4 / binary weights 93.2 / BNN 89.9 / BNN+binary input 85.6 | Cerutti Fig.5 |
| 8-filter AFE (50 Hz–8 kHz) | 76.3% | Cerutti Fig.6 |
| 64-ch AFE, 12-class | 86.0% | Cerutti 결론 |
| AFE 정확도 상한 | 이진화 정보 손실로 ~85% 부근 | Cerutti VI-C |
| threshold 초기화 | 학습셋 전체의 채널별 평균, min-max 스케일 | Cerutti IV-A |
| 엔벨로프 | 10 ms 또는 25 ms 윈도우 내 스펙트로그램 최대값 | Cerutti IV-A |
| BN 융합 | `binAct(x) = x·sgn(γ′) ≥ ⌊β′/γ′⌋ ? 0 : 1` | Cerutti 식 (3) |
| MatchboxNet 전처리 | 64 MFCC, 25 ms/10 ms, 128 프레임 zero-pad | MatchboxNet 4.1 |
| MatchboxNet 최적화 | NovoGrad(0.95/0.5), WHD 5/45/50, lr 0.05→0.001, wd 0.001, 200 ep | MatchboxNet 4.1 |
| MatchboxNet-3x2x64 정확도 (v2, 35-class) | 97.21% ± 0.072, 93K params | MatchboxNet Table 3 |

**주의**: 16채널 AFE에서 12-class 85% 목표는 Cerutti의 8ch(76.3%)와
64ch(86.0%) 사이에 있고, 논문 스스로 AFE 방식의 상한을 ~85%로 언급한다.
목표가 낙관적일 수 있다 — sweep 결과가 나오면 재검토한다.

**결정 기록 (2026-07-21)**: conv2는 기본 **separable** (총 96.5K, 논문 93K와
일치). dense(총 324K)는 정확도 미달 시 1순위 ablation — CLAUDE.md 2.2 참고.
