# 첫 작업 지시 (Claude Code에 붙여넣을 프롬프트)

아래 내용을 Claude Code 세션에 그대로 전달한다. (CLAUDE.md가 프로젝트 루트에 있어야 한다.)

---

프로젝트 루트의 `CLAUDE.md`를 먼저 읽어라. 거기 적힌 아키텍처 결정과 제약을 반드시 따른다.
이번 세션의 목표는 **BinaryMatchboxNet의 학습 가능한 PyTorch 스캐폴드를 세우고,
아주 작은 데이터로 학습 루프가 끝까지 도는 것을 확인**하는 것이다.
전체를 한 번에 완성하려 하지 말고 아래 단계를 순서대로, 각 단계마다 실행·검증하며 진행하라.

## 진행 순서

### 단계 1 — 환경과 뼈대
- 프로젝트 디렉터리 구조 생성: `data/ models/ train/ experiments/ export/ tests/ configs/`.
- `requirements.txt` 작성(torch, torchaudio, numpy, pyyaml, pytest 등).
- 설정을 담는 `configs/base.yaml` 또는 dataclass 정의:
  `C`(채널 폭, 기본 64), `T`(기본 64), `n_afe_channels`(16), `n_classes`(12),
  윈도우/hop(25ms/10ms), Mel 범위(50–7500Hz), 각 층 커널(11/13/15/17/29/1/1)과
  정밀도 플래그(Conv1=int8, B1~B3=binary, Conv2=binary, Conv3=int8, Conv4=fixed).
- 여기서 멈추고 구조를 보여준 뒤 다음으로.

### 단계 2 — 이진 연산 원자(레이어)와 단위 테스트
- `models/binary_ops.py`:
  - `sign_ste`: forward는 sign, backward는 STE(예: hardtanh 기울기 근사).
  - `BinaryConv1d`: 이진 가중치(latent full-precision 유지, forward에서 sign)로 1D conv.
    내부적으로 XNOR-popcount와 동치인 연산임을 주석으로 명시.
  - depthwise 버전과 pointwise(1×1) 버전을 구분해 구현.
- **먼저 테스트부터**: `tests/test_binary_ops.py`
  - ±1 입력·±1 가중치의 일반 conv 결과와 `2*popcount(XNOR)-N` 공식 결과가 일치하는지.
  - `sign_ste`를 통과해도 gradient가 0이 아니라 흐르는지.
- 테스트를 실행해 통과시킨 뒤 다음으로. (실패하면 고치고 재실행.)

### 단계 3 — AFE 이진화 모듈
- `data/afe.py`:
  - log-Mel/MFCC → 16채널로 재매핑(Mel 도메인 등간격 16 corner freq).
  - 채널별 학습 가능 threshold(16개). min-max 정규화, 평균으로 초기화.
  - threshold 이진화는 STE로 backward 통과.
  - 출력 shape `[16, T]`, 값 {-1,+1}.
- 작은 랜덤 텐서로 forward/backward가 도는지 확인.

### 단계 4 — 모델 조립
- `models/binary_matchboxnet.py`:
  - CLAUDE.md 2.2 표대로 스테이지 조립. Conv1(int8 입력이진), B1~B3 이진 TCS,
    Conv2/3/4, avg pool, 12-class head.
  - 이진 TCS sub-block은 depthwise→pointwise→BN→sign, residual은 정수 누산 단계 합산.
  - **모든 층 크기는 config에서 온다**(C, T, 커널). 하드코딩 금지.
- 더미 입력 `[batch, 16, T]`로 forward가 `[batch, 12]`를 내는지 확인. 파라미터 수 출력.

### 단계 5 — 학습 루프 (오버핏 테스트)
- `train/train.py`: 표준 학습 루프(Adam, CrossEntropy, seed 고정, 로깅).
- **실제 데이터셋을 받기 전에**, 소량의 랜덤/합성 배치 몇 개로
  **일부러 오버핏**시켜 loss가 떨어지고 정확도가 100%로 가는지 확인.
  → 이게 되면 gradient가 STE 포함 전 구간을 제대로 흐른다는 증거다.
- 여기서 멈추고 결과(loss 곡선/정확도)를 보고하라.

### 단계 6 — (다음 세션 예고, 지금은 하지 말 것)
- Google Speech Commands v2 다운로드·전처리 파이프라인.
- 실제 학습, (C, T) sweep, 85% 도달 확인.
- 이건 단계 5 검증이 끝난 뒤 별도 세션에서.

## 규칙
- 각 단계 끝에서 **실행 결과를 보여주고** 다음으로 넘어가라. 통과 못 하면 멈추고 고쳐라.
- 오래 걸리거나 대용량 다운로드가 필요한 작업(단계 6 등)은 지금 하지 말고 예고만.
- CLAUDE.md 규칙과 충돌하는 설계 판단이 필요하면 진행 전에 나에게 물어라.
- 논문 수치를 추측으로 지어내지 말 것.

지금 단계 1부터 시작하라.
