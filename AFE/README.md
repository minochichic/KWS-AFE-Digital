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

## 16채널 필터뱅크 설계 (mel 매칭) — 완료

**제어 구조(검증됨)**: `RA → f_c`, `R1 → Q`(Q ≈ R1/RA, f_c와 독립), C는 채널별로
RA를 적정 범위(~10 kΩ)에 두도록 선택. 이 분리 덕에 채널별 설계가 빠르고 안정적이다
(`scripts/design_filterbank.py`).

**결과** (`artifacts/`):
- **f_c 오차: 평균 0.9%, 최대 2.0%** — RA 이분법이 mel 센터에 정밀 정렬.
- **Q도 목표에 일치** (R1=Q·RA 제어), 이득 6~7.8 dB로 균일.
- 컴포넌트값 `(RA, C, R1)` × 16채널 → `filterbank_design.csv`.
- `H_k(f)`를 ML STFT 그리드(0–8000 Hz, 257 bin)에 보간 → `filterbank_matrix.csv [16,257]`.

**이상 mel과 비교** (`scripts/compare_mel.py`, `artifacts/filterbank_vs_mel.png`):
- **평균 cosine 유사도 0.831** (ch15는 0.913). **피크는 완벽 정렬**(f_c 일치).
- 차이는 **2차 밴드패스의 넓은 스커트** vs mel 삼각형의 직선변 — 형상 자체의 한계.
  이는 **Cerutti 논문 Fig.3의 "Simulation vs Mel" 특성을 그대로 재현**한 것. 스커트를
  좁히려면 고차 필터(채널당 opamp↑, 전력↑)가 필요 → 2차 GIC는 저전력 선택의 결과.
- **함의**: SPICE 뱅크는 mel보다 채널 간 겹침이 크다(선택도↓). 이 행렬로 ML을 재학습하면
  정확도가 소폭 변할 수 있다(Phase B 현실). ← 다음 단계.

## Transient 풀체인 (Cerutti Fig.1 재현) — 완료

`scripts/run_transient.py` + `netlists/full_chain.cir`. 음성 클립을 PWL로 주입해
`.tran`으로 V_in → V_filt → V_+(엔벨로프) → V_out(펄스) 4단을 재현
(`artifacts/afe_transient.png`).

- **결과**: V_filt는 채널 주파수(~1.36kHz) 진동(진폭변조), V_+는 **톱니 엔벨로프**
  (빠른 충전+느린 방전), V_out은 임계 교차 펄스 — **논문 Fig.1과 정성적으로 동일**.
- **τ를 회로에서 측정**: 톱니 방전 시상수 **τ ≈ R5·C3 = 47k×100n ≈ 4.7ms**.
  논문에 없던 값을 회로로 확정 (CLAUDE.md 2.5의 "추정 수 ms"와 일치).
- **비교기 임계는 엔벨로프 범위에 튜닝 필수**: V_+가 이 신호(입력 4mV)에서 0.915~0.934V
  → 임계 0.924V. 친구 넷리스트의 R7/R8 분압(0.9216V)은 마침 이 범위에 가깝지만, 임계는
  **입력 레벨·채널마다 달라지는** 값(= ML의 학습 threshold, 하드웨어의 분압 튜닝).
- **입력**: `AFE/audio/`에 GSC `.wav`를 넣으면 그걸(가장 큰 40ms 창) 자동 사용,
  없으면 음성 유사 합성신호로 데모.

### 모델 호환성 메모 (ngspice)
- **BAT54**: 벤더 subckt가 `mfg=Diotec` 파라미터로 ngspice에서 실패 → 표준 BAT54
  Schottky를 2핀 `.model DBAT54`로 직접 정의.
- **LPV7215**: PSpice `if()`/switch 구문이라 ngspice 파싱 불가 → 비교기를 거동(tanh)
  모델로 대체. 실제 비교기 비이상성(offset/delay/hysteresis)은 LTspice/PSpice에서 검증 가능.
- **OPA379**는 ngspice 호환 → 필터·검출기에 실모델 그대로 사용.

## 남은 단계
- **ML 연동**: `AFEConfig`에 `filterbank_source: "mel"|"spice"` + `spice_matrix_path`
  추가 → `artifacts/filterbank_matrix.csv`를 필터로 써서 재학습, 이상 mel 대비 정확도 비교.
- **편차 강건 학습**: `(f_c, Q, gain)` perturbation + E12/E24 스냅을 ML 증강으로.
- (선택) 실제 GSC wav로 transient 재실행, LPV7215 실모델을 LTspice로 교차검증.
