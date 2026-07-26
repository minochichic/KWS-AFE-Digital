# AFE — 아날로그 프론트엔드 SPICE 검증

이 폴더는 **아날로그 AFE(GIC 필터 + 능동 검출기 + 비교기)의 회로 시뮬**만 담는다.
ML 코드(`../data`, `../models`, `../train`, …)와 완전히 분리되어 있으며, 이 폴더
바깥은 건드리지 않는다. 목적: 이상 삼각 mel 필터 대신 **실제 회로의 필터 응답**을
추출해 AFE 설계를 공고히 하고(로드맵 Phase B), 그 응답을 ML 필터뱅크로 넘길 근거를
만든다.

## 구조
```
AFE/
  models/      OPA379.LIB          # 실제 opamp SPICE 모델(TI, 유한 GBW)
  netlists/    gic_channel.cir     # GIC 밴드패스 1채널(파라미터화, .ac)
  scripts/     sweep_filterbank.py # RA 스윕 → 채널별 f_c/gain/Q 특성화
  sim/         (gitignore) ngspice 출력 CSV
```

## 실행
```bash
brew install ngspice          # ngspice-46 사용
cd AFE
ngspice -b netlists/gic_channel.cir          # 1채널 .ac
python scripts/sweep_filterbank.py           # 필터뱅크 스윕(ngspice 반복 호출)
```
> `.venv`의 python(numpy 필요)으로 스크립트 실행. LPV7215(비교기)·BAT54(다이오드)
> 모델은 아직 없으나 **필터 응답 추출(.ac)에는 GIC 단만 필요**해 불필요하다.

## 원본 넷리스트 검토 (친구 KiCad 작성)
- **핀 순서 정상**: OPA379 서브ckt는 `+IN -IN +V -V OUT`, 넷리스트 배선이 일치.
- **경로 수정**: `C:/Users/duyou/...` 절대경로 → 자립형 `models/OPA379.LIB`(CRLF·비ASCII 정리).
- **`.ac`는 GIC 필터에만 유효**: 검출기(D1/D2 정류)·비교기는 비선형이라 `.ac`(소신호
  선형)로는 무의미. 필터 형상 추출엔 정확히 맞고, 실제 엔벨로프/펄스는 `.tran` 필요(후속).
- 원본 1채널 값: `RA=2.32k, C=10n, R1=14.7k, R2=R3=100k`.

## 검증으로 나온 발견 (중요)

1. **실제 f_c ≠ 이상 공식, 고주파일수록 크게 어긋난다.**
   `RA=2.32k, C=10n`에서 공식 `1/(2πRA·C)=6861 Hz`인데 **실측 피크 5390 Hz**.
   스윕 전체에서 `sim/ideal` 비율이 저주파 ~1.00 → 고주파 0.73으로 하락. OPA379 유한
   GBW + GIC 실제 전달함수 때문. → **공식으로 코너를 배치하면 20%까지 빗나간다.**

2. **8 kHz 도달은 가능하나, RA가 아니라 C를 줄여야 한다.**
   `C=10n` 고정 RA 스윕은 최고 ~6.1 kHz에서 막힌다(그 조합의 한계). `RA=2.32k`에서
   **C를 4.7n으로 줄이면 f_c≈9.5 kHz** 도달. 즉 상위 채널은 작은 캡으로 설계해야 하며,
   **f_max=8000은 OPA379로 물리적으로 실현 가능**(ML의 f_max=8000 이득이 헛되지 않음).

3. **Q·이득이 채널마다 제멋대로다(단일 파라미터 스윕 기준).**
   RA만 바꾸면 Q가 0.1~47, 이득이 6~28 dB로 요동. 원본 `RA=2.32k` 채널은 우연히
   고Q(28 dB) 지점. **깨끗한 등Q 필터뱅크가 아니다** → 채널별로 `(RA, C, R1)`를
   **공동 설계**해 목표 f_c에 Q·이득을 맞춰야 한다(다음 단계).

## 다음 단계
- **B-a. 2-파라미터 채널 설계**: 목표 f_c(50–8000 Hz, mel 등간격)마다 `(RA, C, R1)`를
  잡아 f_c·Q·peak-gain을 SPICE로 맞춘다. 저주파=큰 RA, 고주파=작은 C.
- **B-b. 필터뱅크 행렬 export**: 각 채널 `.ac` 응답 `H_k(f)`를 STFT 주파수 그리드에
  보간해 `filterbank_matrix[16, n_freq]` CSV 생성 → ML(`AFEConfig.filterbank_source="spice"`).
- **B-c. transient 교차검증**: 순음·GSC 클립을 PWL로 주입, 비교기 펄스열을 확인
  (BAT54/LPV7215 모델 필요 — 일반 Schottky + 거동 비교기로 대체 가능).
- **B-d. 편차 강건 학습**: `(f_c, Q, gain)` perturbation을 ML 증강으로.
