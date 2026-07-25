# 실험 로그 (running ledger)

지금까지의 모든 학습 실행과 결과. 새 실행이 끝나면 표에 한 줄 추가.
설계·사양은 [`first_task_result.md`](first_task_result.md), AFE 변수는
[`afe_config.md`](afe_config.md) 참고. 모든 실행은 GSC v2, 12-class, T4/Colab.

## 실행 요약

| # | tag | C·T | f_max | ep | seed | test | best val | train | 비고 |
|---|-----|-----|-------|----|----|------|----------|-------|------|
| 1 | sc_v2 | 64·128 | 7500 | 100 | 1234 | 0.809 (last) / 0.8236 (best ep64) | 0.840 | 0.849 | 첫 baseline |
| 2 | sc_v2 (8000) | 64·128 | **8000** | 100 | **1234** | **0.850** (last) / **0.8513** (best ep79) | 0.862 | 0.872 | 85% 넘김 |
| 3 | sc_v2_seed2024 | 64·128 | 8000 | 100 | **2024** | **0.838** (last) / best TBD | ~0.856 | 0.872 | **85% 미달** |
| 4 | sweep C32_T96 | 32·96 | 7500 | 70 | 1234 | 0.592 | — | — | ⚠️ T=96 broken |
| 5 | sweep C32_T128 | 32·128 | 7500 | 70 | 1234 | 0.761 | — | — | C 축소 효과 |
| 6 | sweep C48_T96 | 48·96 | 7500 | 70 | 1234 | (중단) | — | — | broken, 중단됨 |

> **85%는 경계값 (seed 민감).** seed 1234 = 0.850/0.851 (넘김), seed 2024 = 0.838
> (last, 미달). 두 seed last 평균 ≈ 0.844. "안정적 돌파"가 아니라 목표선 위에서
> 진동. 확정하려면 여러 seed 평균±CI(Cerutti는 5-trial) 또는 더 결정적인 레버 필요.
> #4~6 sweep은 f_max=7500 시절이라 #2/#3과 직접 비교 불가.

### 진행/미결 (Colab 실행 필요)

- [x] **best.pt(8000, seed1234)** = **0.8513** (ep79).
- [~] **seed 2024**: last **0.838** (미달). best.pt 재평가 남음 (val 최고 ~0.856).
- [ ] **더 결정적인 레버**로 85%를 여유있게 넘기기: conv2 dense, 증강(SpecAugment).
- [ ] (선택) **여러 seed 평균±CI**로 현재 설정의 진짜 정확도 확정.
- [ ] **C-sweep @ 8000** — 단, 85%가 경계값이라 "최소 크기 탐색"은 더 견고한
      설정 확보 후가 순서상 맞음.

---

## 핵심 발견

1. **f_max 7500→8000 = +4%p (0.809→0.850, seed 1234).** seed 동일, f_max만 변경.
   16-filter AFE에선 코너 주파수 배치가 정확도를 채널 수만큼 좌우 (Cerutti Fig.6:
   8-filter에서 범위만 바꿔 53.9→76.3%). → **f_min/f_max가 가장 강력한 레버.**
   단, 결과는 **85% 경계값이라 seed 민감**(seed 2024 last 0.838). 견고한 돌파는 아님.
2. **작은 T는 구조적으로 깨진다.** conv2 span 57 > (conv1 후 프레임 수)이면 conv2가
   padding을 훑음. T=96→48프레임→0.592. T=128→64프레임→정상. sweep이 자동 skip.
   ([`08_dilation_span.svg`](diagrams/08_dilation_span.svg))
3. **C 축소는 정확도 하락** (7500 기준 C32·T128=0.761 < C64·T128=0.809). 용량 병목.
4. **best-val 모델이 last보다 낫다** (7500: best 0.8236 > last 0.809). test는 best.pt로.
5. **Mel "등간격"은 Mel 척도 기준** (Hz로는 저주파 촘촘, Fig.3). 구현 정상.
6. **과적합 아님, 약한 과소적합** (train≈val+1.6%). 용량/증강 여지 남음.

---

## 다음 레버 (우선순위)

1. best.pt 재평가 + seed 2024 재현 → 0.850 확정.
2. **C-sweep @ 8000** (T=128, C=16/32/48/64) → 최소 크기 = 하드웨어 크기 확정
   (CLAUDE.md 1 최종 목표).
3. 더 밀기: **conv2 dense**(용량), **증강**(SpecAugment), **f_min** 탐색.

---

## 재현용 Colab 셀

**best.pt(8000) 재평가:**
```python
from data.afe import AFEFrontend
from data.speech_commands import ensure_dataset, build_dataloaders
from models.binary_matchboxnet import BinaryMatchboxNet
from train.config import load_config
from train.train import Trainer, set_seed
import torch

cfg = load_config('configs/base.yaml', {'tag': 'sc_v2'})
cfg.data.root = '/content/datasets/speech_commands_v2'
ensure_dataset(cfg.data.root, cache_tar='/content/drive/MyDrive/kws_data/gsc_v2.tar')
set_seed(cfg.train.seed)
afe = AFEFrontend(cfg.afe); model = BinaryMatchboxNet(cfg.model)
_, _, test_loader = build_dataloaders(cfg.data, cfg.train.batch_size,
                                      cfg.afe.sample_rate, seed=cfg.train.seed)
ck = 'runs/sc_v2/best.pt'
ep = torch.load(ck, map_location='cpu', weights_only=False)['epoch']
trainer = Trainer(cfg, model, afe=afe); trainer.load_checkpoint(ck)
test = trainer.evaluate(test_loader)
print(f"best.pt (f_max={cfg.afe.f_max:.0f}, epoch {ep}) test acc {test['acc']:.4f}")
```

**seed 2024 재현:**
```python
!python -m train.train --config configs/base.yaml --tag sc_v2_seed2024 \
    train.seed=2024 data.root=/content/datasets/speech_commands_v2
```

**C-sweep @ f_max=8000:**
```python
!python -m experiments.sweep --config configs/base.yaml \
    --C 16 32 48 64 --T 128 --epochs 70 \
    data.root=/content/datasets/speech_commands_v2
```
