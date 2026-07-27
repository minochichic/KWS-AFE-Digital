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
(`artifacts/afe_transient_ch6.png`).

- **결과**: V_filt는 채널 주파수(~1.36kHz) 진동(진폭변조), V_+는 **톱니 엔벨로프**
  (빠른 충전+느린 방전), V_out은 임계 교차 펄스 — **논문 Fig.1과 정성적으로 동일**.
- **τ를 회로에서 측정**: 톱니 방전 시상수 **τ ≈ R5·C3 = 47k×100n ≈ 4.7ms**.
  논문에 없던 값을 회로로 확정 (CLAUDE.md 2.5의 "추정 수 ms"와 일치).
- **비교기 임계는 엔벨로프 범위에 튜닝 필수**: V_+가 이 신호(입력 4mV)에서 0.915~0.934V
  → 임계 0.924V. 친구 넷리스트의 R7/R8 분압(0.9216V)은 마침 이 범위에 가깝지만, 임계는
  **입력 레벨·채널마다 달라지는** 값(= ML의 학습 threshold, 하드웨어의 분압 튜닝).
- **채널 선택**: `python AFE/scripts/run_transient.py --ch 3` 처럼 16채널 중
  아무거나 (RA/C/R1을 `filterbank_design.csv`에서 읽음). 기본 ch6(~1.36kHz).
- **부품값 표**: 16채널 RA/C/R1 + 공통 부품 → `artifacts/component_table.md`.
- **입력**: `AFE/audio/`에 GSC `.wav`를 넣으면 그걸(가장 큰 40ms 창) 자동 사용,
  없으면 음성 유사 합성신호로 데모.

### 모델 호환성 메모 (ngspice)
- **BAT54**: 벤더 subckt가 `mfg=Diotec` 파라미터로 ngspice에서 실패 → 표준 BAT54
  Schottky를 2핀 `.model DBAT54`로 직접 정의.
- **LPV7215**: PSpice `if()`/switch 구문이라 ngspice 파싱 불가 → 비교기를 거동(tanh)
  모델로 대체. 실제 비교기 비이상성(offset/delay/hysteresis)은 LTspice/PSpice에서 검증 가능.
- **OPA379**는 ngspice 호환 → 필터·검출기에 실모델 그대로 사용.

## 회로도 · 16채널 스펙트로그램 (시각화)

- **회로도** `artifacts/afe_schematic.svg` (+ `.png`) — `full_chain.cir` 기반 1채널
  회로도(GIC 밴드패스 → 능동검출기 → 비교기)를 **실제 배선으로 전부 연결**(논문 Fig.1
  스타일). `scripts/draw_schematic.py`로 생성(`pip install schemdraw` 필요). 값은
  **변수명**(RA=RA1=RA2, CVAL=C1=C2, R1v, R2v=R2=R3, VREF)과 지정자(R4/R5/R6/C3)로만
  표기(상수 숫자 없음). 극성: U1 +=np, U2 −=nn, U3 −=da, CMP +=V+. 16채널은 이 회로의
  **채널별 RA/CVAL/R1v 복제**.
- **16채널 스펙트로그램** `artifacts/afe_spectrogram16.png` — 실제 SPICE 풀체인을 GSC
  단어에 **16채널 모두** 돌려 만든 시간-주파수 이미지(`scripts/spectrogram16.py --wav <이름>`).
  상단 = V+ 엔벨로프(연속), 하단 = 이진 AFE 출력(NN 입력, 채널별 학습 threshold로 이진화).
  'six'에서 **/s/ 마찰음(HF) + 모음(LF) 분리**가 뚜렷 → 회로가 유의미한 T-F 특징을 만든다.

## 비교기 임계 R7/R8 (학습 threshold 매핑) — 진행

비교기 기준 V- = 1.8·R8/(R7+R8)를 채널마다 정한다. 두 근거로 표를 만든다.

- **중간지점**(`scripts/threshold_table.py` → `artifacts/threshold_table.md`):
  각 채널 엔벨로프 V+ 범위의 중앙. 캘리브레이션은 `AFE/audio/`의 실제 GSC wav
  여러 개(없으면 광대역 합성)를 집계. 순진한 기본값.
- **학습 threshold**(`scripts/learned_r7r8.py` → `artifacts/threshold_learned_r7r8.md`):
  best.pt(sc_v2_dense, val 0.8715)의 채널별 학습 threshold t_k∈[0,1]를 V+로 매핑.
  실제 GSC 12단어(마찰음+모음)로 전역/채널별 V+ 스윙 측정.

**두 매핑 철학**:
- **GLOBAL** — ML의 전역 min-max에 충실. t_k를 공유 [V_LO,V_HI]에 배치.
- **LOCAL** — 하드웨어(채널별 독립 R7/R8)에 충실. t_k를 채널 자기 스윙에 배치.

### 핵심 발견 (중요)
1. **GLOBAL에선 HF 채널(12~15)이 OFF로 뜬다.** 원인은 전역 vs 로컬이 **아니라**
   **압축 도메인 불일치**: ML은 `log(power)` 위에서 threshold를 학습하는데(afe.py:147),
   아날로그 검출기 V+는 진폭(∝√power)이라 압축이 훨씬 약하다. log가 약한 HF를
   크게 끌어올리지만 √는 못 해서, HF 정규화 위치가 학습 threshold에 못 미친다.
   → **A(전역 유지)로 가도 비교기 앞에 log-amp가 없으면 OFF는 남는다.**
2. **LOCAL은 채널 간 상대 라우드니스(=스펙트럼 형상) 정보를 버린다.** 전역 min-max가
   보존하던 "지금 어느 채널이 센가"가 사라지므로, 전 채널이 활성이 되는 대신 대비가
   뭉개진다. 그래서 LOCAL은 실용적 근사일 뿐 의미상 열등.
3. **threshold는 프론트엔드에 공적응한다.** 이상 mel(현재)이 아니라 SPICE 필터뱅크
   (넓은 스커트·겹침↑)로 재학습하면 threshold 16개가 다른 값으로 수렴한다.

### 결론 / 다음
가장 충실한 길은 하드웨어에 맞춘 프론트엔드로 **재학습**하는 것 → 아래 로드맵.
(논문은 log-mel + global min-max로 소프트웨어 시뮬 → 실제 아날로그 V+와의 압축
간극은 논문에 명시 없음: **확인 필요**.)

## 로드맵: 회로-정합 재학습 (계획, 결정됨)

**목표**: GIC 필터 + 능동검출기 + 비교기 회로에 정합된 프론트엔드로 NN을 재학습하고,
그 위에서 threshold·R7/R8·비교기를 재조정한다. 그러면 log-vs-√ 불일치(→HF OFF)가
학습 단계에서 흡수된다(threshold가 √압축 V+ 도메인에서 공적응).

**결정 (사용자):**
- **global min-max 정규화는 유지한다.** 채널 간 스펙트럼 형상 + 라우드니스 불변성을
  보존하기 위함(버리면 열등). `normalize="none"`은 채택하지 않는다.
- 따라서 이진화 직전의 이 정규화는 하드웨어에서 **비교기 앞 공유 AGC/정규화 단**
  (또는 이벤트 수집 후 디지털 정규화)에 대응하고, **R7/R8은 그 정규화 스케일 기준**으로
  설정한다. 이 단을 아키텍처에 명시할 것.

**구조 (미분가능 대리모델 — ngspice는 학습 루프에 못 넣음, `.tran`이 수 초):**
```
audio
 → SPICE 필터뱅크 (artifacts/filterbank_matrix.csv, 추출 완료)   ← mel 대체
 → 검출기 모델: 정류 + √압축 + EMA(τ≈4.7ms, 회로서 측정)          ← 미분가능 근사
 → global min-max (유지)
 → 채널별 학습 threshold(STE) = 비교기                            ← 학습 대상
 → binary [16,T] → NN  (end-to-end)
후처리: 학습 threshold → R7/R8 환산 → SPICE 풀체인으로 교차검증,
        비교기 behavioral(tanh) → 실제 LPV7215(offset/hysteresis) 재조정
```
SPICE는 **학습 루프가 아니라 특성화(필터행렬·τ·압축곡선)와 사후 검증**에만 쓴다.

**단계:**
1. 검출기 √압축·τ 곡선을 SPICE에서 정량 추출(입력 진폭 sweep → V+ 응답).
2. `data/afe.py`에 `filterbank_source: mel|spice` + 검출기 압축 모델 추가(격리 벗어남 →
   착수 전 승인). global min-max·STE threshold는 그대로.
3. 재학습 → 정확도(이상 mel baseline 대비) + 새 threshold 16개 확보.
4. 새 threshold → R7/R8(정규화 스케일 기준) → SPICE 풀체인 교차검증.
5. 비교기 실모델(LPV7215)·편차(E12/E24, f_c/Q perturb) 강건화.

## 검출기 특성화 (부품값 근거) — 완료 [로드맵 step 1]

검출기 상수(R4/R5/R6/C3)를 주장하는 대신 **그 값이 만드는 응답을 SPICE로 실측**
(`scripts/characterize_detector.py` + `netlists/detector.cir`, 검출기 단독 구동).
결과 `artifacts/detector_characterization.png`, `detector_char.md`.

- **τ 방전 = 4.62 ms ≈ 공칭 R5·C3(4.70 ms)** — C3/R5 값의 근거 **확립**.
  충전 t90 ≈ 0.65 ms → **빠른 충전 / 느린 방전** 톱니 확인(비대칭 ~7배).
- **압축은 선형(R5/R4)이 아니다**: 데드존 무릎 + 초선형(p≈1.3~1.9) + 완만한 포화.
  gain@100mV ≈ 3~4 (공칭 4.7보다 낮음).
- **주파수 의존 데드존**: 250 Hz 5.9 mV → 1 kHz 7.3 mV → 4 kHz 12.1 mV.
  저GBW OPA379의 크로스오버 왜곡이 **저신호·고주파를 못 정류** → 앞서 HF 채널이
  약했던 물리적 원인 중 하나. **R4(이득)·OPA379(GBW)가 재검토 대상**임을 정량 확인.
- **함의**: 재학습용 검출기 모델은 단순 √/선형이 아니라 **(데드존+무릎+포화, 주파수
  의존) 곡선**을 재현해야 한다(위 곡선을 룩업/파라메트릭으로). τ는 EMA로.

## 검출기 이득 재조정 (R4 sweep) — 비교기 마진 확보

`scripts/sweep_detector_gain.py` → `artifacts/detector_gain_sweep.png/.md`.
G=R5/R4를 R4로 조절(R5·C3 고정 → τ 불변). 비교기 Vos≈5mV, 레일여유 900mV 기준.

- **현 R4=10k(G=4.7)는 약신호(10mV 입력)에서 V+ 스윙 ~4mV < Vos** → 마진 없음(확증).
- **R4↓로 약신호 스윙 크게 개선**: R4=1k(G=47) → 55mV(≈11×Vos), R4=2.2k(G=21) → 30mV.
- **데드존(~5–7mV)은 R4로 거의 안 준다** — OPA379 GBW의 크로스오버 한계. HF를 더
  낮추려면 **더 빠른 저전력 opamp**가 별도 레버.
- **상단은 ~635mV 소프트 포화**(900mV 레일 미도달=하드클립 없음). 이득 크면 천장에 일찍
  닿아 큰 신호 다이내믹레인지↓.
- **권장**: **R4≈1kΩ(G≈47)** 1차(마진 최대·무클립), 상단 압축 우려 시 R4=2.2k(G≈21).
  대가: R4↓ = 필터출력 부하·소비전류↑(µW 예산).
- **다음 검증**: 확정 R4로 full_chain·threshold 재캘리브레이션 + **비교기 오프셋/히스테리시스
  모델**을 넣어 채널별 펄스 생존 재평가(behavioral tanh는 이 문제를 가림).

## op-amp GBW sweep — 데드존의 진짜 레버 (SR)

`scripts/sweep_opamp_gbw.py` + `netlists/detector_gbw.cir`(GBW 가변 단극 거동
op-amp). → `artifacts/opamp_gbw_sweep.png/.md`. 1 kHz에서 GBW를 30k→10M sweep.

- **데드존 ∝ 1/GBW 확인**: 저GBW 구간 로그-로그 기울기 ≈ −0.84 (이론 −1).
- **반전 (중요)**: 단극 모델 @90kHz = 0.31 mV인데 **실측 OPA379 = 7.3 mV (~23배)**.
  이론선 $2V_D f/GBW$ 위에 실측이 정확히 놓임 → 실제 데드존은 **SR(슬루레이트) 한계**가
  지배(0 교차에서 출력이 2·Vd 슬루하는 시간). 단극 모델은 SR 무한이라 과소평가.
- **결론**: op-amp를 빠르게 하면 데드존↓(방향 맞음), **단 진짜 병목은 GBW가 아니라 SR**.
  HF 채널을 살리려면 GBW·SR 둘 다 높은 op-amp 필요 → 둘 다 **소비전류↑**(µW 예산과
  트레이드오프). 다음: 거동모델에 SR 제한을 넣어 SR sweep으로 데드존↔전력 정량화.

## 논문 대조: OPA379로 어떻게 동작하나 (데드존 스레드 결론)

PDF 확인: **Cerutti 논문은 실제로 OPA379(GBW 90 kHz, 2.9 µA)와 LPV7215 비교기를
그대로 사용**한다(우리 부품과 동일). 즉 우리가 찾은 데드존/슬루 한계는 **그들 설계에도
존재**하며, 논문 정확도(8ch 76.3%, 64ch 86%)는 이미 그 비이상성을 포함한 실측이다.

그럼에도 동작하는 이유:
- **BNN이 AFE 비선형/희소성을 견디도록 설계됨.** 논문이 공유하는 관점은 AFE에서 "고선형성을
  노리지 않음"(Yang et al. 인용)으로 전력을 줄인다는 것. 데드존에 걸린 약/HF 채널은 "드물게
  발화하는 희소 특징"이 될 뿐 실패가 아니며 분류기가 흡수한다.
- **능동 마이크(16 µA@0.9V)+검출기 이득 G=R5/R4**가 주요(저·중역) 채널을 데드존 위로 올림.
  (참고: 우리가 본 "LNA"는 Cerutti가 아니라 인용문헌 Yang의 것 — Cerutti엔 별도 LNA 없음.)
- **우리의 "마진 붕괴"는 상당 부분 우리가 임의로 고른 낮은 이득(R4=10k)·입력레벨 탓**
  (category B 값). 논문의 실제 R4/R5는 미공개. 데드존은 실물리지만 정확도 저하 여부는 동작점
  문제이고, 논문은 동작점이 존재함을 보인다.

**함의**: 데드존은 "고칠 버그"가 아니라 "모델링할 프론트엔드 특성"이다. → R4 조정도 유효하나
**근본 해법은 실제 비선형 응답(데드존 포함)으로 BNN을 재학습**(= Cerutti가 실물에서 한 방식).

## 비교기 마진 검증 (R4 10k vs 1k + 오프셋) — 완료

`scripts/verify_comparator_margin.py` → `artifacts/comparator_margin.png/.md`.
`full_chain.cir`에 `R4v`·`VOS`(비교기 입력 오프셋) 파라미터 추가(기본 10k/0 = 기존과 동일).
클립 `six.wav`, Vos=5mV.

- **Part A**: R4=10k에선 **16채널 중 다수가 V+ 스윙 ≤ Vos(5mV)** → 마진 없음.
  R4=1k로 바꾸면 대부분 15–53mV로 상승(그림). 단 **최상위 HF(5.7/6.8kHz)는 1k에서도
  4–5mV로 애매** — 데드존/OPA379 GBW 한계라 R4로는 안 풀림(더 빠른 opamp 필요).
- **Part B (오프셋 플립)**: 마진 애매한 ch2(447Hz)에서 임계=엔벨로프 중앙, ±5mV 주입:
  - R4=10k: 출력 듀티 **−5mV→100%, +5mV→0%** = 오프셋이 0/1을 완전히 뒤집음(센싱 붕괴).
  - R4=1k: 듀티 65% vs 41%(펄스 13/14) = **채널이 살아 실제 신호를 감지**(마진 회복).
  - (rising-edge 카운트는 '항상 ON'을 0으로 오독 → **듀티사이클**로 측정.)
- **결론**: R4↓(이득↑)이 비교기 마진 문제를 실제로 완화함을 정량 확인. behavioral tanh
  (Vos=0)로는 안 보이던 문제. **채택 시** `R4v` 기본값을 1k로 바꾸고 component/threshold/
  spectrogram 아티팩트를 재생성해야 함(현재 기본값은 원본 10k 유지).

## 남은 단계
- **ML 연동**: `AFEConfig`에 `filterbank_source: "mel"|"spice"` + `spice_matrix_path`
  추가 → `artifacts/filterbank_matrix.csv`를 필터로 써서 재학습, 이상 mel 대비 정확도 비교.
- **편차 강건 학습**: `(f_c, Q, gain)` perturbation + E12/E24 스냅을 ML 증강으로.
- (선택) 실제 GSC wav로 transient 재실행, LPV7215 실모델을 LTspice로 교차검증.
