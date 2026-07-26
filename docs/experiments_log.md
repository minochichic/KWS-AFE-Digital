# 실험 로그 (running ledger)

지금까지의 모든 학습 실행과 결과. 새 실행이 끝나면 표에 한 줄 추가.
설계·사양은 [`first_task_result.md`](first_task_result.md), AFE 변수는
[`afe_config.md`](afe_config.md) 참고. 모든 실행은 GSC v2, 12-class, T4/Colab.

## 실행 요약

| # | tag | C·T | f_max | ep | seed | test | best val | train | 비고 |
|---|-----|-----|-------|----|----|------|----------|-------|------|
| 1 | sc_v2 | 64·128 | 7500 | 100 | 1234 | 0.809 (last) / 0.8236 (best ep64) | 0.840 | 0.849 | 첫 baseline |
| 2 | sc_v2 (8000) | 64·128 | **8000** | 100 | **1234** | **0.850** (last) / **0.8513** (best ep79) | 0.862 | 0.872 | 85% 넘김 |
| 3 | sc_v2_seed2024 | 64·128 | 8000 | 100 | **2024** | 0.838 (last) / **0.8523** (best ep79) | ~0.856 | 0.872 | best는 넘김 |
| 4 | sweep C32_T96 | 32·96 | 7500 | 70 | 1234 | 0.592 | — | — | ⚠️ T=96 broken |
| 5 | sweep C32_T128 | 32·128 | 7500 | 70 | 1234 | 0.761 | — | — | C 축소 효과 |
| 6 | sweep C48_T96 | 48·96 | 7500 | 70 | 1234 | (중단) | — | — | broken, 중단됨 |
| 7 | sc_v2 (RTX) | 64·128 | 8000 | 100 | 1234 | 0.8355 (last) / **0.8314** (best ep73) | 0.847 | 0.856 | RTX 5070Ti, no-aug |
| 8 | sc_v2_aug (RTX) | 64·128 | 8000 | 100 | 1234 | 0.822 (last) / **0.8247** (best ep95) | 0.845 | **0.809↓** | ⚠️ 증강이 해침 |
| 9 | sc_v2_dense (RTX) | 64·128 | 8000 | 100 | 1234 | **0.858** (last) / best.pt TBD | 0.871 | **0.886↑** | conv2 dense, 324K |

> **#9 conv2 dense (separable→dense, 96.5K→324K).** test 0.8355→**0.858**, train
> 0.856→0.886 → **용량이 레버**였다(순수 AFE 정보 천장 아님). 85% 초과. 대가는
> 파라미터 3.4×(FPGA 완전펼침 비용↑). 정확도↔하드웨어 트레이드오프의 상단.

> **#7,#8 (로컬 RTX 5070 Ti, torch 2.11+cu128).** 같은 환경 공정비교: 증강이 best
> (0.8314→0.8247)·last(0.8355→0.822) 모두 하락 → **증강은 과소적합 국면에서 역효과**
> 확정. #7 no-aug ~0.83~0.836은 Colab last-대역(0.838~0.850) 하단과 겹쳐 정상 변동
> 범위(체계적 환경차 아님). 이번 run은 best.pt(0.8314)가 last(0.8355)보다 낮음 —
> val/test split 노이즈(best-val ≠ best-test). **종합: 결과는 ~0.82~0.85에서 진동,
> 85%는 경계값.** train 0.856(≠100%)+증강 역효과 → 과소적합/AFE 정보 천장(~85%).

> **best-model 기준 85% 돌파는 견고, last-model은 seed-noisy.** best.pt:
> seed1234 0.8513 / seed2024 0.8523 (둘 다 넘김, 평균 0.852). last: 0.850/0.838
> (seed 민감). 표준인 best-val 선택을 쓰면 2 seed 모두 85%↑ — 합리적 돌파.
> (완전한 확정은 5-seed CI. 두 seed 다 best@ep79 → 80 epoch면 충분.)
> #4~6 sweep은 f_max=7500 시절이라 #2/#3과 직접 비교 불가.

### 진행/미결 (Colab 실행 필요)

- [x] **best.pt(8000)** seed1234 **0.8513** / seed2024 **0.8523** — 둘 다 85%↑.
- [ ] **증강(augmentation)** 도입으로 마진 확보 (다음 레버; time-shift/noise/SpecAugment).
- [ ] (선택) **epoch 100→80**으로 단축 (best가 ep79).
- [ ] (선택) 5-seed 평균±CI로 확정. / C-sweep @ 8000 (더 견고해진 뒤).

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
6. **과소적합; 용량이 레버(#9로 갱신).** 증강(#8)은 역효과였으나 conv2 dense(#9)는
   0.8355→0.858, train 0.856→0.886으로 상승 → 순수 AFE 정보 천장이 아니라 **용량
   병목**이었다. 단 파라미터 3.4×. 정확도↔하드웨어 트레이드오프. 채널 수↑(64ch=86%)는
   여전히 정보 천장을 더 올리는 별도 레버(헌장 고정 → 상의).

---

## 다음 레버 (우선순위)

증강은 과소적합 국면에서 역효과(#8 확인) → 제외. 남은 순서:

1. **평가는 best.pt로 통일** (last는 seed·하드웨어에 흔들림).
2. **저비용 용량/최적화 1회씩**: **conv2 dense**(용량), **NovoGrad+WHD**(최적화) →
   용량/최적화로 짜낼 여지가 남았는지 확인.
3. **AFE 채널 16→24/32** — 진짜 천장 돌파구지만 헌장 고정이라 **상의 필요**.
4. (선택) 5-seed 평균±CI로 현재 설정 확정. / C-sweep로 최소 크기(85% 유지 시).

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
