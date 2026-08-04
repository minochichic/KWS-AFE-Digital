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

## Phase B: 회로-정합 프론트엔드 (mel→SPICE 필터, log→√ 압축)

이상 mel 대신 **SPICE 추출 GIC 필터뱅크**(`AFE/artifacts/filterbank_matrix.csv`)와, log 대신
**검출기 충실 √ 압축**(V+∝진폭=√power)을 `AFEConfig`로 선택. 전부 RTX, separable, f_max 8000,
seed 1234, global min-max 유지. 앵커(mel+log+sep) = #7 test-best **0.8314** (Colab #2 0.8513).

| front end | filterbank | comp | conv2 | test | best val | train | 비고 |
|---|---|---|---|---|---|---|---|
| **B0** mel+log (=#7) | mel | log | sep | 0.8314 (best) | 0.847 | 0.856 | 앵커 |
| **B1** spice+log | spice | log | sep | **0.586** | — | — | log이 넓은 스커트 과증폭 → 채널 상관 0.89(중복) |
| **B2** spice+√ | spice | sqrt | sep | **0.8016** (best) | 0.8195 | ~0.82 | √로 회복. 채널 상관 0.56 |
| **B3** spice+√ dense | spice | sqrt | dense | 0.776 (last) | ~0.81 | 0.82 | dense 무효 → **입력 정보 천장**(용량 아님) |
| **B4** mel+√ | mel | sqrt | sep | 0.846 (last) | 0.864 | 0.87 | **√ > log** (mel에서 0.831→0.846) |
| **B5** spice+√+deadzone | spice | sqrt+dz | sep | 0.780 (last) | ~0.802 | 0.793 | ❌ 데드존이 정보 삭제 → 하락 |
| **B6** spice **4차**+√ | spice(4차뱅크) | sqrt | sep | 0.770 (last) | ~0.790 | 0.78 | ❌ 4차(블러−49%)도 정확도 하락 |

> **B6 4차 필터 뱅크 = 음성 결과 (블러 가설 최종 반증).** AFE_highorder에서 설계한 대역보존
> 4차 뱅크(인접 겹침 −49%, 커버리지 유지)로 재학습 → best ~0.79 < 2차 0.80, train 0.78 < 0.82.
> 즉 **필터를 날카롭게 해도 정확도는 안 오르고 오히려 하락.** 데드존(B5)에 이어 **두 번째로**
> "채널 겹침/블러는 −4.4pp의 원인이 아니다"를 확증. 넓은 스커트 꼬리는 유용한 정보를 담아
> 잘라내면(데드존)·좁히면(4차) 손해. → **mel과의 −4.4pp는 필터 기하로 복구 불가**(실제 필터
> 응답의 고유 특성). 4차는 전력·면적 2× + 정확도 하락 → **기각**. 회로-정합 최선은 2차 spice+√
> 0.80 유지. (남은 물리적 레버는 필터 날카로움이 아니라 **채널 수↑** — 헌장 고정.)

> **B5 데드존 = 음성 결과 (중요 교훈).** 학습 데드존(init 0)이 채널 상관을 로컬에서
> 0.58→0.38로 낮췄지만(sharpening 프록시), 실제로는 train 천장 0.82→0.793으로 **하락**.
> 즉 **채널 상관은 정확도의 잘못된 프록시**였다 — 넓은 필터의 꼬리는 상관돼 있어도
> 판별 정보를 담고, 자르면 손해. (√도 상관 0.56<mel 0.75인데 정확도는 낮았음 → 같은 함정.)
> → **−4.4pp는 후처리로 복구 불가**: 2차 GIC의 넓은 스커트가 스펙트럼을 **블러**한 것이라
> 데드존/탈상관으로 못 되살림. 진짜 물리적 필터 한계(Cerutti AFE < 이상 mel과 동일 성격).
> 코드: `spice_deadzone` 플래그는 남기되 기본 OFF, 권장 안 함.

> **2×2로 원인 분리 완료.** (a) **√는 범인 아님 — 오히려 낫다**: mel에서 log→√ 0.831→0.846.
> (b) **spice의 하락은 전부 필터뱅크**: 같은 √끼리 mel 0.846 → spice 0.802 = **−4.4pp**(2차 GIC
> 넓은 스커트의 순수 대가). (c) **√는 spice 필터엔 필수**: log에선 필터 대가가 −24pp(0.831→0.586)
> 참사인데 √가 −4.4pp로 낮춤. → **√ 채택 확정.** 남은 −4.4pp = 실제 아날로그 필터의 정직한 비용
> (Cerutti AFE가 이상 mel보다 낮은 것과 동일 성격). 코드: `data/afe.py`(|H|² + compression),
> `train/config.py`(filterbank_source/compression). 진단: 채널 상관 log 0.89 vs √ 0.56.

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

## Stage 2 — 필터 모양 vs 정규화 분해 (2026-08-02)

per-clip min-max가 비인과(클립 전체 max 참조)라 하드웨어 불가임을 확인한 뒤, **필터뱅크와
정규화를 2×2로 교차**해 두 손실을 처음으로 분리 측정. 전부 √ 압축, 16채널, 동일 모델·학습.

| | minmax (실현 불가) | fixed q=0.75 (실현 가능) | 정규화 비용 |
|---|---:|---:|---:|
| **mel** (이상적 삼각) | 0.846 | **0.761** | −8.5pp |
| **spice** (실제 GIC) | 0.802 | **0.726** | −7.6pp |
| **필터 모양 비용** | −4.4pp | −3.5pp | |

### 결론: 두 손실은 독립·가산적이고, 정규화가 2배 크다

```
0.846  --(-4pp 필터 모양)-->  0.80  --(-8pp 정규화)-->  0.726
```

- **필터 모양 ≈ −4pp** (정규화와 무관하게 일정)
- **정규화 ≈ −8pp** (필터와 무관하게 일정)
- → "정확도 하락은 실제 필터 모양 탓"은 **틀렸다**. 모양은 1/3, 정규화가 2/3.

### 부수 실험 (전부 기각)

| 시도 | 결과 | 판정 |
|---|---:|---|
| fixed 스케일 조건수 (q 스윕) | q=1.0 0.693 / **q=0.75 0.726** / q=0.6 0.653 | q=0.75 최적, 레버 **소진** |
| 시간 비트 2배 (5 ms, T=256) | ~0.67 (10 ms 0.726 대비 **−6pp**) | ❌ **기각** |
| 주파수 범위 125–5000 (mel+log) | 0.836 (50–8000의 0.831 대비 +0.5pp) | 미미 → spice 재설계 불필요 |
| 오프셋 vos 스윕 | 0~0.002에서 평평(0.726→0.721) | 오프셋은 병목 아님 |

**5 ms 실패 원인**: 커널이 프레임 단위라 conv2 시간 수용영역이 1140 ms → 570 ms로 반토막.
단어(~500 ms) 전체를 못 봄. 게다가 검출기 τ=4.7 ms가 이미 대역을 제한하므로 5 ms 창의
추가 비트는 새 정보가 아님. **10 ms가 τ와 정합**.

### 남은 레버 (AGC 미채택 전제)

필터 개선은 **최대 +3.5pp**가 상한이고(이상적 필터 도달 시), 4차 필터는 이미 실패(0.79).
시간축도 실패. → **주파수축 비트 증가만 남음**:

| 레버 | 아날로그 추가 | 상태 |
|---|---|---|
| 채널당 비교기 2개 (비트 2배) | +17 µW | 미시도, 최우선 |
| 채널 수 16→32 | +266 µW | Cerutti의 검증된 레버 |
| (AGC + 캘리브레이션) | +5~10 µW → **0.789** | 사용자 판단으로 보류 |

### Cerutti 대조
우리 16ch mel+fixed = **0.761** ≈ Cerutti 8ch **0.763**. 그들이 per-clip 정규화를 썼다면
우리 16ch minmax 0.846이 그들 8ch(0.763)~64ch(0.860) 사이에 자연스럽게 놓인다.
→ **논문의 76/86%도 레벨 정규화를 가정한 값일 가능성**이 높다(고정 분압기 매핑은 미검증).
