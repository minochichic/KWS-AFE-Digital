# Stage 2 — ML을 하드웨어 실측에 정합

Stage 1(실제 벤더 모델)에서 얻은 수치로 ML의 오프셋 모델을 **추정에서 실측 기반으로** 바꾼다.

## 1. 측정된 하드웨어 상수

| 양 | 값 | 출처 |
|---|---|---|
| Venv_DC (무신호 정지점) | OPA379 917.8 mV / TLV9041D 943.4 mV | 실모델 .op |
| 검출 신호 rise (대역 활성 시 v_env 상승) | 28~65 mV (대표 ~50 mV), 채널 무관 균일 | swing_real.py |
| 검출기 이득 R5/R4 | ×4.7 | 설계 |
| LPV7215 비교기 오프셋 | ~0.3 mV (모델) | lpv_offset.cir |
| op-amp GBW | OPA379 92k / TLV9042 328k / **TLV9041D 1.72M** | gbw_*.cir |

## 2. comparator_vos 앵커 (정규화 단위 = Vos / rise)

ML의 `comparator_vos`는 정규화 임계 도메인의 std다. 실제로는 `Vos / rise`:

| 시나리오 | Vos | **vos_norm** | 대응 ML 정확도 |
|---|---:|---:|---|
| 이상 (오프셋 0) | 0 | 0 | 0.80 (상한) |
| **캘리브레이션 후** (트림 잔차+드리프트) | ~1 mV | **0.015–0.036** | **~0.74** (E1 평탄) |
| 미캘리브레이션 (검출기 offset ×4.7 소자편차) | ~5 mV | 0.08–0.18 | ~0.63 |

우리 기존 offset-aware 런(vos=0.035→0.73, 0.10→0.63)이 이 범위를 **이미 브래킷**한다.
즉 Stage 1 측정이 앞선 ML 실험이 옳은 범위였음을 확증.

**결정적 함의**: 동료 GUI의 `Vmargin→R7/R8` 기능으로 **소자별 1회 캘리브레이션**을 하면
잔차가 ~1 mV → vos_norm ~0.02 → **정확도 ~0.74**. 캘리브레이션 안 하면 ~0.63.
→ **per-device 캘리브레이션이 +10pp 가치**. (콜드 스타트 시 DC .op 1회로 자동화 가능.)

## 3. 남은 결정: 정규화 방식 (정확도 vs 하드웨어)

offset-aware 런은 모두 per-clip minmax 베이스였다. 하지만 per-clip minmax는 **실현
불가**(미래를 보는 이상적 AGC). 실현 가능한 두 베이스로 offset-aware를 돌려야 최종
숫자가 나온다:

| 정규화 | 베이스 정확도 | 실현 | offset-aware(캘리브 vos=0.02) |
|---|---:|---|---|
| per-clip minmax | 0.80 | ✗ 이상적 AGC | 0.744 (기측정) |
| **fixed / 절대임계** | 0.70 | ✓ 고정 R7/R8 (추가 회로 0) | **← Run A (미측정)** |
| **causal AGC** | 0.807 | ✓ VGA 프리앰프 필요 | **← Run B (미측정)** |

Run A vs Run B 격차 = **"AGC/VGA 회로를 넣을 가치가 있나"**의 답.

## 4. 확인 런 (RTX, 코드 변경 0 — config만)

```python
import torch, numpy as np
def run(tag, norm, vos):
    cfg = load_config('configs/base.yaml', {'tag':tag,
        'afe.filterbank_source':'spice', 'afe.compression':'sqrt',
        'afe.normalize':norm, 'afe.comparator_vos':vos,
        **({'afe.ste_clip':4.0} if norm=='none' else {})})
    cfg.data.root = DATA_ROOT; set_seed(cfg.train.seed)
    afe = AFEFrontend(cfg.afe); model = BinaryMatchboxNet(cfg.model)
    tr, va, te = build_dataloaders(cfg.data, cfg.train.batch_size,
                                   cfg.afe.sample_rate, seed=cfg.train.seed)
    w = next(iter(tr))[0]
    afe.init_fixed_scale(w); afe.init_thresholds(w)      # 순서 중요
    t = Trainer(cfg, model, afe=afe); t.fit(tr, va, resume=True)
    ck = torch.load(t.run_dir/'best.pt', map_location=t.device, weights_only=True)
    t.model.load_state_dict(ck['model']); t.afe.load_state_dict(ck['afe'])
    print(tag, 'val', round(ck['best_acc'],4), 'test', round(t.evaluate(te)['acc'],4))

# Run A — 고정 스케일(추가 회로 없음) + 캘리브레이션 오프셋
run('af_fixed_vos02', 'fixed', 0.02)
# Run B — 인과 AGC(VGA 필요) + 캘리브레이션 오프셋
run('af_agc_vos02',   'agc',   0.02)
# (선택) 미캘리브레이션 대조: vos=0.10 으로 각각 한 번 더
```

## 5. 판정 기준

- **Run B − Run A ≥ ~7pp** → AGC/VGA 회로 값어치 있음 → 프리앰프를 가변이득(VGA)으로 설계.
- **격차 작음** → 고정 스케일 + 고정 프리앰프로 단순화(회로 절약).
- 두 경우 모두 **per-device 캘리브레이션은 필수**(무하면 −10pp).

이 두 런이 아날로그↔ML 정합의 마지막 조각이고, 이후 Stage 3(학습 threshold→R7/R8→SPICE
역검증)로 간다.
