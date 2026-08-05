# AFE_tuning — 동료 ngspice_channel_sweeper 회로의 소자값 튜닝 (격리)

동료가 제공한 `../ngspice_channel_sweeper/`(GUI 스윕 툴)의 **회로 토폴로지는 그대로 두고**
소자값만 어떻게 조정할지 SPICE로 검증한 결과.

## 동료 툴 분석 요약
- `netlist_template.cir` = **우리 AFE와 동일 토폴로지** 1채널(GIC + 능동검출기 + LPV7215),
  소자값이 전부 `.param`으로 노출 → 스윕에 이상적.
- `channel_components.csv` = 16채널 소자표. **RA/C/R1/gain_dB는 우리 `filterbank_design.csv`
  값**, R7/R8/Vthr는 우리 **초기(mel+log) 매핑** 값.
- 잘 설계된 규칙: **R6 = R4∥R5 자동 제안**, **Vmargin = Vref − Vdet → R7/R8 역산**(0.01 kΩ
  반올림), WAV를 **10 mVpp**로 정규화(평균 제거), `tmax ≤ 1/(20·f_max)` 근거.
- ⚠️ 넷리스트 기본 입력이 `SIN(0 1 1k)` = **진폭 1 V**(AC 단위입력). 그대로 transient를
  돌리면 V+가 1.48–1.56 V로 **포화**(실측). GUI는 PWL(10 mVpp)로 교체하므로 정상이지만,
  넷리스트만 단독으로 쓸 때는 입력 레벨을 반드시 낮춰야 한다.

## 결과
[`artifacts/tuning_r4.md`](artifacts/tuning_r4.md) — R4 스윕, 이득 예산, 최종 권고,
바꾸지 말 것, 적용 방법.
[`artifacts/channel_components_tuned.csv`](artifacts/channel_components_tuned.csv) —
동료 GUI `버전 불러오기`로 바로 읽히는 형식(경로 B용).

**핵심**: `G=R5/R4`, `τ=R5·C3` → **R4가 유일한 깨끗한 이득 노브**(R5는 τ까지 바꿈).
단 **프리앰프와 R4를 함께 올리면 포화** → 권장은 **프리앰프 G≈10 + R4=10k(기본 유지)**.

## 격리 규칙
이 폴더 안에서만 작업. `../ngspice_channel_sweeper/`(동료 원본)와 `AFE/`, `data/`,
`configs/`는 건드리지 않음. 복귀점 태그 `phase-b-2ndorder`.
