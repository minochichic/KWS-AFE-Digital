# analog/ — 아날로그 프론트엔드 작업 전체

Cerutti식 AFE(GIC 대역통과 → 능동 검출기 → 비교기)의 설계·시뮬·튜닝 기록.
**모든 수치는 ngspice에서 실제 벤더 모델(OPA379, BAT54WT1, LPV7215, TLV904x)로 측정**했다.
회로 자체의 해설은 [`../docs/afe_circuit_explained.md`](../docs/afe_circuit_explained.md).

> ⚠️ 폴더 이름은 **바꾸지 말 것**. 스크립트들이 서로를 상대경로로 참조한다
> (예: `AFE_tuning/scripts/*.py` → `../../ngspice_v15_2608001/lib`).

---

## 지금 살아 있는 것 (ML이 실제로 쓴다)

| 폴더 | 내용 | 상태 |
|---|---|---|
| [`AFE/`](AFE/) | 16채널 GIC 필터뱅크 설계 + SPICE 추출. **`artifacts/filterbank_matrix.csv`를 학습이 매 런 읽는다** (`afe.filterbank_source: spice`) | ✅ **확정 — 삭제/이동 금지** |
| [`AFE_tuning/`](AFE_tuning/) | 동료 회로 실측 튜닝. R7/R8 채널별 확정, 검출기 DC 전달곡선, op-amp GBW 측정 | ✅ 확정 |
| [`AFE_final/`](AFE_final/) | 최종 시스템 회로도(소자값·전력 포함), Stage 2 근거 정리 | ✅ 확정 |

**`AFE/artifacts/filterbank_matrix.csv`가 아날로그와 ML을 잇는 단 하나의 접점이다.**
경로는 `train/config.py`의 `spice_matrix_path` 한 곳에만 있다.

## 기각된 탐색 (기록 보존용)

| 폴더 | 시도 | 왜 기각했나 |
|---|---|---|
| [`AFE_highorder/`](AFE_highorder/) | 4차 필터로 스커트 좁히기 | 정확도 0.79로 **이득 없음**, 전력·면적만 2배 |
| [`AFE_micamp/`](AFE_micamp/) | 마이크 프리앰프로 HF 스윙 확보 | 오프셋 여유는 늘었으나 **ML 정확도 불변** — 오프셋은 병목이 아니었음 ([`../docs/experiments_log.md`](../docs/experiments_log.md)) |

## 동료 제공 회로 (외부 입력, 우리가 수정하지 않음)

| 폴더 | 버전 |
|---|---|
| [`ngspice_channel_sweeper/`](ngspice_channel_sweeper/) | 1차 버전 |
| [`ngspice_v15_2608001/`](ngspice_v15_2608001/) | v15 (최신). `lib/`에 벤더 SPICE 모델, `component_versions/`에 우리가 되돌려준 수정 소자값 |

---

## 확정된 실측값 (자주 참조됨)

| 항목 | 값 | 출처 |
|---|---|---|
| GIC 통과대역 이득 | **2.00** (6 dB, 공식과 일치) | `AFE_tuning/` |
| 검출기 이득 G = R5/R4 | **4.7** (47k/10k) | 회로 |
| 엔벨로프 시상수 τ = R5·C3 | **4.7 ms** | 회로 |
| 검출기 정지점 v_env | **917.8 mV** (동료 917.83과 0.55 mV 이내 일치) | `AFE_tuning/artifacts/r7r8_real.md` |
| DC 전달곡선 기울기 / 바닥 | **−4.70 / 894.4 mV** | ″ |
| 채널 스윙(정지점 위 상승) | **28~65 mV** (~균일) | ″ |
| op-amp GBW | OPA379 92 kHz / TLV9042 328 kHz / TLV9041D 1.72 MHz | `AFE_tuning/artifacts/opamp_gain_per_channel.md` |
| 비교기 LPV7215 | 0.58 µA | 데이터시트 |

> ⚠️ **주의**: 초기 채널별 "스윙"은 **정상상태 리플**을 잰 것이라 틀렸다(86→1.2 mV,
> ch14/15 사용불가 결론). 비교기가 검출하는 것은 **정지점 위로의 상승**이고 그건 거의
> 균일하다. `v15_circuit_review.md` §3-3에 오류 표시를 남겨 두었다.

## 재현

```bash
# 벤더 모델 파싱에 .spiceinit 의 `set ngbehavior=pski` 가 필요하다
cd analog/AFE_tuning && python scripts/swing_real.py
```
