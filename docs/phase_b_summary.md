# Phase B 요약 — 회로-정합 프론트엔드 재학습 (복귀용 문서)

이 문서 하나로 Phase B 전체 맥락을 복원한다. 상세 표는
[`experiments_log.md`](experiments_log.md)의 "Phase B" 절, 아날로그 근거는
[`../analog/AFE/README.md`](../analog/AFE/README.md).

**복귀점(태그)**: `git checkout phase-b-2ndorder` → 이 상태(2차 GIC · 16채널 · Phase B
완료)로 정확히 복귀. 이후 고차 필터 연구는 `../analog/AFE_highorder/`에서 격리 진행(기존 결과 불변).

---

## 무엇을 했나

이상 mel 필터 대신 **SPICE로 추출한 실제 GIC 필터뱅크**로 디지털단(BinaryMatchboxNet)을
재학습. `AFEConfig`에 프론트엔드 스위치 추가(기본값은 mel+log = 기존 baseline 불변):

- `filterbank_source: "mel" | "spice"` — 이상 삼각 vs SPICE GIC 필터.
- `compression: "log" | "sqrt"` — MFCC 관습(log) vs 검출기 충실(√=진폭).
- `spice_gain_restore: bool` — peak-norm 해제, 채널 실이득 복원 (효과 ~0).
- `spice_deadzone: bool` — 학습 검출기 데드존 (효과 음성, 권장 안 함).

global min-max 정규화는 전 과정 유지(채널 간 상대 레벨 보존).

## 결과 (전부 separable, RTX, f_max 8000, seed 1234)

| # | front end | test(best) | train | 한 줄 |
|---|---|---:|---:|---|
| B0 | mel + log (앵커) | 0.831 | 0.856 | 기존 baseline과 동일 프론트엔드 |
| B1 | spice + log | 0.586 | — | log이 넓은 스커트 과증폭 → 참사 |
| B2 | spice + √ | **0.802** | 0.82 | √로 회복 = **회로-정합 최선** |
| B3 | spice + √ + dense | ~0.80 | 0.82 | dense 무효(입력 정보 천장) |
| B4 | mel + √ | **~0.85** (0.846 last) | 0.87 | √가 log보다 나음 |
| B5 | spice + √ + deadzone | ~0.80 (0.78 last) | 0.793 | 데드존 정보 삭제 → 하락 |

## 결론 (다섯 가지)

1. **√ > log, 그리고 spice엔 √ 필수.** mel에서 log→√ 0.831→0.846. log에선 필터 대가가
   −24pp(0.831→0.586) 참사인데 √가 −4.4pp로 낮춤. → **√ 채택 확정.**
2. **실제 2차 GIC 필터의 대가 ≈ −4.4pp** (mel+√ 0.846 vs spice+√ 0.802). 넓은 스커트가
   스펙트럼을 **블러**한 것 = 물리적 한계. Cerutti AFE가 이상 mel보다 낮은 것과 동일 성격.
3. **채널 상관은 정확도의 프록시가 아니다.** √(상관 0.56<mel 0.75)도, 데드존(0.38)도
   상관을 낮췄지만 정확도는 낮았음. 넓은 꼬리는 상관돼도 판별 정보를 담는다.
4. **−4.4pp는 후처리로 복구 불가.** 데드존·탈상관·이득복원 전부 실패. 블러는 되돌릴 수 없음.
5. **16채널이 ~0.85(이상)/~0.80(실제) 천장.** 85%를 견고히 넘기려면 후처리가 아니라
   **프론트엔드 정보량**(채널 수 또는 필터 선택도)을 늘려야 함. Cerutti: 8ch 76% / 64ch 86%.

## 재현 명령 (RTX / Colab)

```python
# B4 mel+√ (이상 필터 최선, ~0.85)
cfg = load_config('configs/base.yaml', {'tag':'sc_v2_mel_sqrt', 'afe.compression':'sqrt'})
# B2 spice+√ (회로-정합 최선, ~0.80)
cfg = load_config('configs/base.yaml',
    {'tag':'sc_v2_spice_sqrt', 'afe.filterbank_source':'spice', 'afe.compression':'sqrt'})
# B0 mel+log 앵커 = 태그 이전 baseline (experiments_log #2/#7)
cfg = load_config('configs/base.yaml', {'tag':'sc_v2'})
cfg.data.root = DATA_ROOT
# 평가는 항상 best.pt로 (train 후):
#   ck=torch.load(trainer.run_dir/'best.pt',map_location=trainer.device,weights_only=True)
#   trainer.model.load_state_dict(ck['model']); trainer.afe.load_state_dict(ck['afe'])
#   trainer.evaluate(test_loader)
```

## 다음 (복귀 후 선택지)

- **①마무리**: 위 −4.4pp를 "아날로그 비용"으로 확정 리포트, export/하드웨어 추정으로.
- **②채널 수 16→24/32**: 진짜 천장 돌파구지만 CLAUDE.md 고정 → 승인 필요.
- **③고차/고Q 필터** ← *현재 진행 중*, `../analog/AFE_highorder/`에서 격리 연구.
