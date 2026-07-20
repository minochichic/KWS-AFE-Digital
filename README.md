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
| 5 | 학습 루프 + 합성 데이터 오버핏 검증 | ⬜ |
| 6 | Speech Commands v2 파이프라인, (C, T) sweep | ⬜ |

## 개발 환경

**로컬(맥)** — 코드 작성, 단위 테스트. GPU 불필요.

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest
```

**Colab Pro** — 실제 학습. [`docs/colab_setup.md`](docs/colab_setup.md)의
GitHub PAT + Colab 시크릿 셋팅을 한 번 끝낸 뒤
[`notebooks/colab_bootstrap.ipynb`](notebooks/colab_bootstrap.ipynb)를 연다.

흐름: **로컬에서 수정 → push → Colab에서 부트스트랩 셀 재실행 → 학습.**
Colab 워킹 카피는 매 실행마다 `git reset --hard` 되므로 거기서 소스를 고치지 않는다.

## 디렉터리

```
configs/      YAML 설정. C, T, 커널, 정밀도 — 모든 크기의 단일 출처
data/         데이터셋·전처리·AFE 이진화 파이프라인 (파이썬 패키지)
models/       BinaryMatchboxNet, 이진 레이어, STE, BN-threshold 융합
train/        config 정의, 학습 루프, 로깅
experiments/  (C, T) sweep 스크립트, 결과 집계
export/       정수 threshold + bit-packed 가중치 변환 (하드웨어 대비)
tests/        단위 테스트
notebooks/    Colab 부트스트랩
datasets/     (gitignore) 다운로드된 오디오
runs/         (gitignore) 체크포인트·로그
```

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

레포지토리 루트의 두 PDF에서 확인한 값. 인용 시 여기를 근거로 삼는다.

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
