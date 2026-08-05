# 환경과 재현

---

## 1. 툴과 버전

| 용도 | 도구 | 비고 |
|---|---|---|
| 아날로그 시뮬 | **ngspice 46** | `brew install ngspice` |
| 아날로그 모델 | OPA379, BAT54WT1, LPV7215, TLV904x | 벤더 제공 실제 모델, [`analog/ngspice_v15_2608001/lib/`](../analog/ngspice_v15_2608001/lib/) |
| 회로 원본 | KiCad (동료) | [`analog/ngspice_v15_2608001/`](../analog/ngspice_v15_2608001/) |
| ML | **PyTorch + torchaudio** | [`requirements.txt`](../requirements.txt) |
| 개발 | macOS, Python 3.9 (로컬 `.venv`) | 코드 작성 + 단위 테스트만, GPU 불필요 |
| 학습 | **RTX (WSL)**, Python 3.12 | GitHub pull로 코드 동기화 |
| 데이터셋 | Google Speech Commands **v2** | 105k utterances, 12-class, 공식 split |

> ⚠️ ngspice가 벤더 PSpice 라이브러리를 파싱하려면 `.spiceinit`에
> `set ngbehavior=pski` 와 `set filetype=ascii` 가 필요하다.

**작업 흐름**: 맥에서 수정 → push → RTX에서 `git pull` → 학습. RTX 워킹 카피에서
소스를 고치지 않는다.

## 2. 아날로그 재현

```bash
# 16채널 필터뱅크 설계 + 주파수 응답 추출 (.ac)
cd analog/AFE && python scripts/sweep_filterbank.py

# 실제 벤더 모델로 스윙/마진 측정 (.tran)
cd analog/AFE_tuning && python scripts/swing_real.py
```

산출물:
- `analog/AFE/artifacts/filterbank_design.csv` — 채널별 f_c/Q/RA/C/R1
- `analog/AFE/artifacts/filterbank_matrix.csv` — **ML이 매 학습마다 읽는 파일**

## 3. 학습 재현

경로는 `train/config.py`의 `spice_matrix_path` 한 곳에만 있다
(`analog/AFE/artifacts/filterbank_matrix.csv`).

```python
import os, sys
os.chdir(os.path.expanduser('~/KWS-AFE-Digital')); sys.path.insert(0, os.getcwd())

import torch, numpy as np
from train.config import load_config
from data.speech_commands import build_dataloaders
from data.afe import AFEFrontend
from models.binary_matchboxnet import BinaryMatchboxNet
from train.train import Trainer, set_seed

DATA_ROOT = os.path.expanduser('~/datasets/speech_commands_v2')

def run(tag, over):
    cfg = load_config('configs/base.yaml', {'tag': tag, **over})
    cfg.data.root = DATA_ROOT
    set_seed(cfg.train.seed)
    afe = AFEFrontend(cfg.afe); model = BinaryMatchboxNet(cfg.model)
    tr, va, te = build_dataloaders(cfg.data, cfg.train.batch_size,
                                   cfg.afe.sample_rate, seed=cfg.train.seed)
    w = next(iter(tr))[0]
    if cfg.afe.normalize in ('fixed', 'agc', 'xmax', 'xmix'):
        afe.init_fixed_scale(w)          # 순서 중요: δ가 먼저
    afe.init_thresholds(w)               # 그 다음 α
    t = Trainer(cfg, model, afe=afe)
    t.fit(tr, va, resume=True)
    ck = torch.load(t.run_dir/'best.pt', map_location=t.device, weights_only=True)
    t.model.load_state_dict(ck['model']); t.afe.load_state_dict(ck['afe'])
    print(f"\n>>> {tag}: val {ck['best_acc']:.4f}   test {t.evaluate(te)['acc']:.4f}")
    return t

# 확정 설정
run('af_xmix_f005', {'afe.filterbank_source': 'spice', 'afe.compression': 'sqrt',
                     'afe.normalize': 'xmix', 'afe.xmax_floor_frac': 0.05})
```

### 진단용 변형

```python
# 비교기만 제거 — 1비트가 얼마를 먹는지
run('nn_cont_afe16', {..., 'afe.binarize': False})

# 아키텍처 검증 — 표준 64 log-mel
run('nn_cont_mel64', {'afe.n_channels': 64, 'model.in_channels': 64,
                      'afe.filterbank_source': 'mel', 'afe.compression': 'log',
                      'afe.normalize': 'minmax', 'afe.binarize': False})
```

## 4. α 추출 — 하드웨어 BOM 만들기

학습이 끝나면 이걸로 **채널별 저항비 설계표**를 뽑는다.

```python
ck = torch.load('runs/af_xmix_f005/best.pt', map_location='cpu', weights_only=True)
alpha = ck['afe']['threshold']
delta = float(ck['afe']['xmax_floor'])
fc = [166,295,447,631,832,1072,1349,1660,2042,2455,2951,3467,4169,4898,5754,6761]

RTOT = 1e6                                    # Ra + Rb 자유도. 클수록 저전력
print(f"δ = {delta:.5f} (ML 단위)   Ra+Rb = {RTOT/1e3:.0f} kΩ\n")
print(f"{'ch':>2} {'f_c[Hz]':>8} {'α':>7} {'Rb[kΩ]':>9} {'Ra[kΩ]':>9}")
for c, (f, a) in enumerate(zip(fc, alpha.tolist())):
    rb = RTOT * a; ra = RTOT - rb
    print(f"{c:>2} {f:>8} {a:>7.4f} {rb/1e3:>9.1f} {ra/1e3:>9.1f}")
ok = alpha.min() >= 0 and alpha.max() <= 1
print(f"\n범위 {alpha.min():.3f} ~ {alpha.max():.3f}  "
      f"{'[0,1] 안 — 제작 가능 ✅' if ok else '범위 이탈 — 제작 불가 ⚠️'}")
```

**반드시 확인할 것**: α가 전부 [0,1] 안에 있어야 한다. 벗어나면 어떤 저항 쌍으로도
만들 수 없다.

## 5. 검증 스크립트

```bash
# 단위 테스트 (172개) — 이진 연산, AFE, config, 증강
python -m pytest

# 채널별 표 재생성 (설계 · 제작 · 실측 대조)
python experiments/channel_table.py <speech_commands_v0.02 경로>

# 무음/음성 발화율 — 이벤트율 = 인터럽트 = 전력
python experiments/xmax_event_rate.py <speech_commands_v0.02 경로>

# 회로도 재생성
python proposal/artifacts/make_schematic.py
```

## 6. 재현성

- seed 고정 (`train.seed = 1234`), 결과·config·체크포인트 경로를 파일로 기록
- 데이터셋 split은 공식 권장(`validation_list.txt` / `testing_list.txt`)
- silence 클래스는 배경잡음 크롭으로 **결정론적 생성** (`split_seed + 777`) —
  Colab 세션마다 달라지면 재현성이 깨지므로 고정했다
- `resume=True`라 같은 태그로 다시 돌리면 이어서 학습한다. **처음부터 하려면 태그를
  바꾸거나 `runs/<tag>/last.pt`를 지운다.**
