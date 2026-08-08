"""Rebuild notebooks/gpu_local.ipynb.

Fixes the three ways pasting a cell used to fail:
  * `%autoreload 2` does NOT pick up new dataclass FIELDS, so after a git pull
    the kernel kept an old AFEConfig and dotted overrides raised "no such key"
    (or worse, silently left xmax_floor at 0 and burned 100 epochs);
  * cells depended on names defined in earlier cells (cfg, DATA_ROOT), so
    running one on its own raised NameError;
  * the training cell never called init_fixed_scale(), which xmix/xmax/xlse
    need before init_thresholds().
"""
import json
import pathlib

OUT = pathlib.Path("/Users/tv_mac/Documents/Minho/KWS-AFE-Digital/notebooks/gpu_local.ipynb")


def md(*lines):
    return {"cell_type": "markdown", "metadata": {},
            "source": "\n".join(lines).splitlines(keepends=True)}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": "\n".join(lines).splitlines(keepends=True)}


cells = [

md("""# BinaryMatchboxNet KWS — Remote GPU (WSL / RTX)

**셀 1 → 2 → 6을 먼저 실행한다.** 그 뒤로는 어느 실험 셀이든 단독 실행 가능하다.

| 확정된 설정 | 값 |
|---|---|
| 필터뱅크 | `spice` (실측 GIC 16채널, 50–8000 Hz) |
| 압축 | `sqrt` (검출기가 진폭에 선형 — 실측 검증) |
| 정규화 | `xmix`, `xmax_floor_frac 0.02` |
| 비교기 | 채널당 2개 (2비트) → `model.in_channels = 32` |
| 현재 정확도 | **test 0.802** |

배경과 근거는 [`proposal/`](../proposal/README.md).

> ⚠️ `%autoreload 2`는 **dataclass에 새로 생긴 필드를 반영하지 못한다.**
> `git pull` 뒤에는 **커널을 재시작**하고 셀 1부터 다시 돌린다.
> 셀 2가 그걸 검사해서 조용히 틀린 값으로 학습하는 사고를 막는다."""),

code("""# --- 1. setup: repo 루트로 이동 + 최신 코드 ------------------------------
%load_ext autoreload
%autoreload 2

import os, sys, subprocess
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
REPO = os.getcwd()
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# --no-pager 필수: VSCode의 !셸은 pty에 붙어 git이 less를 띄우고 멈춘다.
print(subprocess.run(['git', 'pull', '--ff-only'],
                     capture_output=True, text=True).stdout.strip())
print(subprocess.run(['git', '--no-pager', 'log', '-1', '--oneline'],
                     capture_output=True, text=True).stdout.strip())
print('repo   :', REPO)
print('python :', sys.executable)"""),

code('''# --- 2. 프리플라이트 + 공용 import (여기서 멈추면 아래는 돌리지 말 것) ---
# 파일이 아니라 "로드된 모듈"을 검사한다. autoreload가 dataclass 필드 추가를
# 반영하지 못하므로, 파일에는 있는데 모듈에 없으면 커널 재시작이 필요하다.
import os, sys, pathlib
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import torch, numpy as np
from train.config import load_config, AFEConfig
from data.speech_commands import ensure_dataset, build_dataloaders, class_names
from data.afe import AFEFrontend
from models.binary_matchboxnet import BinaryMatchboxNet
from train.train import Trainer, set_seed

_NEED = [                    # (설명, 파일, 파일에서 찾을 문자열, 모듈 검사)
    ("spice 경로 analog/ 이전", 'train/config.py', 'analog/AFE/artifacts',
     lambda: AFEConfig().spice_matrix_path.startswith('analog/')),
    ("normalize='xmix'", 'data/afe.py', 'def _xmix',
     lambda: hasattr(AFEFrontend, '_xmix')),
    ("normalize='xlse'", 'data/afe.py', 'def _xlse',
     lambda: hasattr(AFEFrontend, '_xlse')),
    ("effective_alpha()", 'data/afe.py', 'def effective_alpha',
     lambda: hasattr(AFEFrontend, 'effective_alpha')),
    ("비교기 k개", 'train/config.py', 'comparators_per_channel',
     lambda: 'comparators_per_channel' in AFEConfig.__dataclass_fields__),
    ("binarize 스위치", 'train/config.py', 'binarize',
     lambda: 'binarize' in AFEConfig.__dataclass_fields__),
]
_disk, _mem = [], []
for name, f, needle, check in _NEED:
    on_disk = needle in pathlib.Path(f).read_text()
    in_mem = bool(check())
    print(f"{'OK ' if in_mem else 'X  '} {name:<26} 파일 {'O' if on_disk else 'X'}"
          f"  모듈 {'O' if in_mem else 'X'}")
    (_disk if not on_disk else _mem if not in_mem else []).append(name)

if _disk:
    raise RuntimeError(f"파일에 없음 -> git pull 이 안 됐습니다: {_disk}")
if _mem:
    raise RuntimeError(f"파일엔 있는데 모듈엔 없음 -> **커널을 재시작**하세요: {_mem}")

DATA_ROOT = os.path.expanduser('~/datasets/speech_commands_v2')

# 확정 설정. 아래 실험 셀들은 이걸 복사해 필요한 키만 바꾼다.
CONFIRMED = {'afe.filterbank_source': 'spice',
             'afe.compression': 'sqrt',
             'afe.normalize': 'xmix',
             'afe.xmax_floor_frac': 0.02,
             'afe.comparators_per_channel': 2}

print(f"\\n모듈 정상.  DATA_ROOT={DATA_ROOT}  존재={os.path.isdir(DATA_ROOT)}")
print('torch', torch.__version__, '| cuda',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(CPU only)')'''),

code("""# --- 3. 의존성 (한 번만) --------------------------------------------------
# requirements-colab.txt 는 torch/torchaudio 를 일부러 뺀 파일이다. 우리는
# cu128 빌드를 이미 깔았으니 이걸로 설치해야 torch 가 다운그레이드되지 않는다.
!{sys.executable} -m pip install -q -r requirements-colab.txt
# soundfile 이 libsndfile 을 요구하면:  sudo apt install -y libsndfile1"""),

code("""# --- 4. 단위 테스트 (이진 연산 / AFE / 모델 / config 가드) ----------------
!{sys.executable} -m pytest -q"""),

code("""# --- 5. 데이터셋 (한 번 받으면 디스크에 영속) -----------------------------
import collections
ensure_dataset(DATA_ROOT)

_cfg = load_config('configs/base.yaml'); _cfg.data.root = DATA_ROOT
_tr, _va, _te = build_dataloaders(_cfg.data, _cfg.train.batch_size,
                                  _cfg.afe.sample_rate)
print('classes:', class_names())
for nm, dl in [('train', _tr), ('val', _va), ('test', _te)]:
    print(f'{nm:5s}: {len(dl.dataset):>6} clips, {len(dl)} batches')
_w, _y = next(iter(_tr))
print('batch', tuple(_w.shape), 'label hist',
      dict(sorted(collections.Counter(_y.tolist()).items())))"""),

md("""---
## 실험

아래 `run()` 하나로 모든 실험을 돌린다. **셀 2만 통과했으면 어느 셀이든 단독 실행 가능**하다.

`run()`이 학습 **전에** 막아주는 것:

| 검사 | 막는 사고 |
|---|---|
| `model.in_channels` 자동 계산 | `n_channels × k`와 어긋나 config 에러 |
| `init_fixed_scale` → `init_thresholds` 순서 | δ가 없는 상태로 α 초기화 |
| `d > 0` | `floor_frac`이 반영 안 돼 100 에폭 헛돌기 |
| α 범위 | 저항비로 만들 수 없는 값 |"""),

code('''# --- 6. 공용 러너 ---------------------------------------------------------
import yaml

_NEEDS_SCALE = ('fixed', 'agc', 'xmax', 'xmix', 'xlse')
_NEEDS_FLOOR = ('xmax', 'xmix', 'xlse')

def run(tag, over=None, epochs=None, quiet=False):
    """학습 1회. over 는 dotted override dict.

    model.in_channels 는 afe.n_channels * afe.comparators_per_channel 로
    자동 계산한다 (직접 넘기면 그 값을 존중).
    """
    over = dict(over or {})
    ov = {'tag': tag, **over}
    if epochs is not None:
        ov['train.epochs'] = epochs
    # in_channels 를 먼저 계산해야 한다. load_config 는 validate() 를 거치므로
    # 여기서 부르면 바로 그 불일치로 예외가 난다 -> YAML/기본값만 직접 읽는다.
    _raw = (yaml.safe_load(open('configs/base.yaml')) or {}).get('afe') or {}
    _d = AFEConfig()
    n_ch = over.get('afe.n_channels', _raw.get('n_channels', _d.n_channels))
    k = over.get('afe.comparators_per_channel',
                 _raw.get('comparators_per_channel', _d.comparators_per_channel))
    ov.setdefault('model.in_channels', n_ch * k)

    cfg = load_config('configs/base.yaml', ov)
    cfg.data.root = DATA_ROOT
    set_seed(cfg.train.seed)

    afe = AFEFrontend(cfg.afe)
    model = BinaryMatchboxNet(cfg.model)
    tr, va, te = build_dataloaders(cfg.data, cfg.train.batch_size,
                                   cfg.afe.sample_rate, seed=cfg.train.seed)
    w = next(iter(tr))[0]
    if cfg.afe.normalize in _NEEDS_SCALE:
        afe.init_fixed_scale(w)        # 순서 중요: δ 먼저
    afe.init_thresholds(w)             # 그 다음 α

    info = [f'{n_ch}채널 x {k}비트 = {n_ch*k}행',
            f'{sum(p.numel() for p in model.parameters()):,} params']
    if cfg.afe.normalize in _NEEDS_FLOOR:
        d = float(afe.xmax_floor)
        assert d > 0, (f'd=0 — afe.xmax_floor_frac={cfg.afe.xmax_floor_frac} 이 '
                       f'반영되지 않았습니다 (커널 재시작 필요?)')
        info.append(f'd={d:.5f}')
    if cfg.afe.normalize == 'xlse':
        info.append(f'T={float(afe.lse_temp):.4f}')
    a0 = afe.effective_alpha()
    info.append(f'init α {a0.min():.3f}~{a0.max():.3f}')
    print(f'[{tag}] ' + '  '.join(info))

    t = Trainer(cfg, model, afe=afe)
    t.fit(tr, va, resume=True)         # last.pt 있으면 이어감

    ck = torch.load(t.run_dir / 'best.pt', map_location=t.device,
                    weights_only=True)
    t.model.load_state_dict(ck['model']); t.afe.load_state_dict(ck['afe'])
    acc = t.evaluate(te)['acc']
    a = t.afe.effective_alpha()
    dead = int(((a >= 0.99) | (a <= 0.01)).sum())
    print(f"\\n>>> {tag}   val {ck['best_acc']:.4f}   test {acc:.4f}")
    print(f"    α {a.min():.3f} ~ {a.max():.3f}   죽은 비교기 {dead}/{a.numel()}개")
    if k > 1:
        g = (a.view(n_ch, k)[:, 1:] - a.view(n_ch, k)[:, :-1])
        print(f"    임계 간격 {g.min():.4f} ~ {g.max():.4f}  (0이면 중복 낭비)")
    return t
print('run() 준비 완료.')'''),

code("""# --- 7. 확정 설정 재현 (test 0.802 이 나와야 함) --------------------------
run('af_k2', CONFIRMED)"""),

code('''# --- 8. 학습된 α → 채널별 저항비 (동료에게 넘길 BOM) ----------------------
# xmix/xlse 는 α를 forward에서 straight-through 클램프하므로 파라미터 원값이
# [0,1] 밖으로 떠다닌다. 반드시 effective_alpha() 로 읽는다.
TAG, RTOT = 'af_k2', 1e6                      # Ra + Rb. 크게 잡을수록 저전력

cfg = load_config('configs/base.yaml', {**CONFIRMED, 'tag': TAG})
n_ch = cfg.afe.n_channels
k = cfg.afe.comparators_per_channel
afe = AFEFrontend(cfg.afe)
afe.load_state_dict(torch.load(f'runs/{TAG}/best.pt', map_location='cpu',
                               weights_only=True)['afe'])
alpha = afe.effective_alpha().view(n_ch, k)
fc = [166, 295, 447, 631, 832, 1072, 1349, 1660,
      2042, 2455, 2951, 3467, 4169, 4898, 5754, 6761]

print(f"delta = {float(afe.xmax_floor):.5f}   Ra+Rb = {RTOT/1e3:.0f} kohm\\n")
hdr = f"{'ch':>2} {'f_c[Hz]':>8}" + "".join(
    f"{'a'+str(i):>8}{'Rb[k]':>8}{'Ra[k]':>8}" for i in range(k))
print(hdr)
for c in range(n_ch):
    row = f"{c:>2} {fc[c] if c < len(fc) else 0:>8}"
    for i in range(k):
        x = float(alpha[c, i]); rb = RTOT * x
        row += f"{x:>8.4f}{rb/1e3:>8.1f}{(RTOT-rb)/1e3:>8.1f}"
    st = '  <- 죽음' if bool(((alpha[c] >= 0.99) | (alpha[c] <= 0.01)).any()) else ''
    print(row + st)
ok = bool(alpha.min() >= 0 and alpha.max() <= 1)
print(f"\\n범위 {alpha.min():.3f} ~ {alpha.max():.3f}   "
      f"{'제작 가능' if ok else '범위 이탈 — 제작 불가'}")'''),

md("""### 대기 중인 실험

**다이오드-OR는 max가 아니라 log-sum-exp를 만든다** (지수 I-V 때문에 지는 채널도
계속 흘린다). 오차 상한이 `n·V_T·ln16` = 72–108 mV인데 엔벨로프 스윙은 28–65 mV라,
**조용한 음성에서 분모가 2배 부풀어 오른다**.

증폭기 16개를 더 사는 대신 **회로가 실제로 만드는 값으로 학습**해서 다이오드만으로
되는지 판정한다. 기준은 `xmix`의 **0.778**.

| `lse_temp_frac` | 해당 조건 | 분모 과대평가 |
|---|---|---|
| 0.13 | 스윙 ~200 mV (프리앰프 있음, 보통 음성) | 1.4× |
| 0.52 | 스윙 ~50 mV (조용한 음성) — **최악** | 4.1× |"""),

code("""# --- 9. xlse: 다이오드-OR의 soft-max 로 학습 ------------------------------
LSE = {**CONFIRMED, 'afe.normalize': 'xlse',
       'afe.comparators_per_channel': 1}      # xmix k=1 (0.778) 과 직접 비교

run('af_lse013', {**LSE, 'afe.lse_temp_frac': 0.13})
run('af_lse052', {**LSE, 'afe.lse_temp_frac': 0.52})"""),

code("""# --- 10. 진단: 비교기를 빼면 얼마가 남나 (하드웨어 설정 아님) -------------
# binarize=False 는 비교기만 제거해 연속 엔벨로프를 네트워크에 넣는다.
# "모델이 약한가" 와 "1비트가 손해인가" 를 가른다.
run('nn_cont_afe16', {**CONFIRMED, 'afe.comparators_per_channel': 1,
                      'afe.binarize': False})          # 16채널 연속 -> 0.862
run('nn_cont_mel64', {'afe.n_channels': 64, 'afe.filterbank_source': 'mel',
                      'afe.compression': 'log', 'afe.normalize': 'minmax',
                      'afe.binarize': False})          # 아키텍처 검증 -> 0.920"""),

code("""# --- 11. 결과 모아보기 -----------------------------------------------------
import json, glob
rows = []
for p in sorted(glob.glob('runs/*/history.json')):
    tag = p.split('/')[1]
    h = json.load(open(p))
    best = max(h, key=lambda e: e.get('val_acc', 0)) if isinstance(h, list) else {}
    rows.append((tag, len(h) if isinstance(h, list) else 0,
                 best.get('val_acc', float('nan'))))
print(f"{'run':<22}{'epochs':>8}{'best val':>10}")
for tag, n, v in rows:
    print(f'{tag:<22}{n:>8}{v:>10.4f}')"""),

md("""### 결과 커밋

`runs/` 는 gitignore 이므로 체크포인트는 올라가지 않는다. 정확도와 설정만 기록한다:

```bash
# proposal/EXPERIMENTS.md 와 docs/experiments_log.md 에 결과 표를 추가한 뒤
git add proposal docs && git commit -m "..." && git push
```"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("wrote", OUT, "|", len(cells), "cells")
