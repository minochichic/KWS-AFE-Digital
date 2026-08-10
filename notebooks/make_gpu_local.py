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
| 정확도 | **test 0.802** |

이게 `CONFIRMED` 딕셔너리의 내용이다. **최고 기록은 따로 있다** —
`xlse` frac 0.78 + **비교기 1개**로 **test 0.8445** (셀 9). 부품이 오히려 적은데
더 좋지만, 봉우리가 이웃 frac(1.2 / 2.0 = 0.826)보다 튀어 있어 **시드 재현 전까지
`CONFIRMED`를 바꾸지 않는다.**

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
import json, yaml

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
    save_result(t.run_dir, tag, cfg, float(ck['best_acc']), float(acc), a)
    return t


def save_result(run_dir, tag, cfg, val, test, alpha):
    """runs/<tag>/result.json — test 정확도는 여기 말고 어디에도 안 남는다.

    history.json 은 epoch별 train/val 만 담고 test 는 print 로만 나갔다.
    커널이 죽으면 스크롤백과 함께 사라지므로 반드시 파일로 남긴다.
    """
    rec = {'tag': tag,
           'val': round(val, 6), 'test': round(test, 6),
           'normalize': cfg.afe.normalize,
           'lse_temp_frac': float(getattr(cfg.afe, 'lse_temp_frac', 0.0)),
           'xmax_floor_frac': float(cfg.afe.xmax_floor_frac),
           'n_channels': int(cfg.afe.n_channels),
           'k': int(cfg.afe.comparators_per_channel),
           'binarize': bool(getattr(cfg.afe, 'binarize', True)),
           'seed': int(cfg.train.seed), 'epochs': int(cfg.train.epochs),
           'alpha': [round(float(x), 4) for x in alpha],
           'dead': int(((alpha >= 0.99) | (alpha <= 0.01)).sum())}
    pathlib.Path(run_dir, 'result.json').write_text(
        json.dumps(rec, ensure_ascii=False, indent=2))
    return rec


_TE = None                       # test loader 는 런 사이에 동일 -> 한 번만 만든다

def backfill(tag):
    """result.json 이 생기기 전에 끝난 런을 best.pt 로부터 복원한다.

    δ(xmax_floor)와 LSE 온도는 register_buffer 라 체크포인트에 들어 있으므로
    init_fixed_scale 을 다시 부를 필요가 없다.
    """
    global _TE
    d = pathlib.Path('runs', tag)
    if not (d / 'best.pt').exists():
        print(f'  {tag}: best.pt 없음'); return None

    cfg = load_config(str(d / 'config.yaml'))     # 그 런이 실제로 쓴 설정
    cfg.data.root = DATA_ROOT
    if _TE is None:
        _TE = build_dataloaders(cfg.data, cfg.train.batch_size,
                                cfg.afe.sample_rate, seed=cfg.train.seed)[2]

    t = Trainer(cfg, BinaryMatchboxNet(cfg.model), afe=AFEFrontend(cfg.afe))
    ck = torch.load(d / 'best.pt', map_location=t.device, weights_only=True)
    t.model.load_state_dict(ck['model']); t.afe.load_state_dict(ck['afe'])
    rec = save_result(d, tag, cfg, float(ck['best_acc']),
                      float(t.evaluate(_TE)['acc']), t.afe.effective_alpha())
    print(f"  {tag}: val {rec['val']:.4f}  test {rec['test']:.4f}  "
          f"(k={rec['k']}, frac={rec['lse_temp_frac']})")
    return rec

print('run() / backfill() 준비 완료.')'''),

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

md("""### xlse — 다이오드-OR의 soft-max

**다이오드-OR는 max가 아니라 log-sum-exp를 만든다** (지수 I-V 때문에 지는 채널도
계속 흘린다). 이걸 처음엔 **결함**으로 보고 증폭기 16개로 스윙을 키우려 했는데,
실측 결과 **하드 max보다 낫다**.

`lse_temp_frac` = `n·V_T / (전형 프레임 피크 스윙)`. 즉 **frac 하나가 곧 회로 조건**이고,
뒤집으면 **요구 스윙 = 26 mV / frac** 이 동료에게 줄 설계 목표가 된다.

의미: 큰 소리 프레임에서는 frac과 무관하게 분모가 하드 max이고, **조용해질수록
16채널 평균으로 넘어간다. frac이 그 전환점을 정한다.**

| frac | 요구 스윙 | test | |
|---|---|---|---|
| — | (하드 max `xmix`) | 0.7778 | 기준선 |
| 0.13 | 200 mV | 0.8185 | |
| 0.52 | 50 mV | 0.8257 | |
| **0.78** | **33 mV** | **0.8445** | **최고** — 연속 천장 0.862에 −1.7pp |
| 1.2 | 22 mV | 0.8259 | |
| 2.0 | 13 mV | 0.8263 | |
| 5.5 | 4.7 mV | 0.8059 | **프리앰프 없는 조건** |

**스윙 스펙이 헐렁하다**: 13–50 mV(4배 범위)에서 test가 0.826–0.845로 평평하다.
동료가 정확한 값을 맞출 필요가 없다.

> ⚠️ 0.78이 이웃(1.2 / 2.0 = 둘 다 0.826)보다 1.8pp 튀어 있는데, 봉우리가 아니라
> **스파이크 모양**이다. 게다가 `ReduceLROnPlateau` 발동 시점이 런마다 달라
> (lse078 ep93 / lse120 ep71) 런간 변동이 test SE(±0.57pp)보다 크다.
> **시드 재현 전에는 33 mV를 설계 목표로 확정하지 말 것.**"""),

code("""# --- 9. xlse frac 스윕 ----------------------------------------------------
LSE = {**CONFIRMED, 'afe.normalize': 'xlse',
       'afe.comparators_per_channel': 1}      # xmix k=1 (0.778) 과 직접 비교

# frac = 26mV / 요구 스윙.  이미 돌린 런은 last.pt 때문에 즉시 리턴한다.
for frac in (0.13, 0.52, 0.78, 1.20, 2.00, 5.50):
    run(f'af_lse{int(frac*100):03d}', {**LSE, 'afe.lse_temp_frac': frac})"""),

md("""### 다음 3개 (순서대로)

| # | 실험 | 왜 |
|---|---|---|
| **1** | frac 0.78 **시드 재현** | 스파이크가 진짜인지. 이게 먼저다 — 아니면 33 mV를 틀린 근거로 넘긴다 |
| 2 | frac 0.78 + **k=2** | 하드 max에서 +2.5pp였던 레버가 soft-max와 겹치는지 |
| 3 | **표준 증강** | 12-class 문헌(Hello Edge 등)은 배경잡음 + shift ≤100 ms를 쓴다. 우리만 없다 |"""),

code("""# --- 9-2. 후속 (위 표 순서대로 하나씩) ------------------------------------
BEST = {**LSE, 'afe.lse_temp_frac': 0.78}

run('af_lse078_s2', {**BEST, 'train.seed': 1235})            # 1. 재현
# run('af_lse078_k2', {**BEST, 'afe.comparators_per_channel': 2})   # 2
# run('af_lse078_aug', {**BEST, 'data.aug_time_shift_ms': 100.0,    # 3
#                       'data.aug_noise_prob': 0.8})"""),

code("""# --- 10. 진단: 비교기를 빼면 얼마가 남나 (하드웨어 설정 아님) -------------
# binarize=False 는 비교기만 제거해 연속 엔벨로프를 네트워크에 넣는다.
# "모델이 약한가" 와 "1비트가 손해인가" 를 가른다.
run('nn_cont_afe16', {**CONFIRMED, 'afe.comparators_per_channel': 1,
                      'afe.binarize': False})          # 16채널 연속 -> 0.862
run('nn_cont_mel64', {'afe.n_channels': 64, 'afe.filterbank_source': 'mel',
                      'afe.compression': 'log', 'afe.normalize': 'minmax',
                      'afe.binarize': False})          # 아키텍처 검증 -> 0.920"""),

code("""# --- 11. result.json 이 없는 옛 런 백필 (한 번만) -------------------------
# result.json 도입 이전에 끝난 런은 test 가 파일에 없다. best.pt 로 복원한다.
# 런당 test 1회 통과(4,888클립)라 GPU에서 수십 초.
import pathlib
for p in sorted(pathlib.Path('runs').glob('*/best.pt')):
    if not (p.parent / 'result.json').exists():
        backfill(p.parent.name)"""),

code("""# --- 12. 결과 모아보기 -----------------------------------------------------
import json, pathlib
rows = [json.loads(p.read_text())
        for p in sorted(pathlib.Path('runs').glob('*/result.json'))]
rows.sort(key=lambda r: -r['test'])

print(f"{'run':<20}{'norm':>6}{'frac':>6}{'k':>3}{'val':>9}{'test':>9}{'죽은':>7}")
for r in rows:
    frac = f"{r['lse_temp_frac']:.2f}" if r['normalize'] == 'xlse' else '—'
    print(f"{r['tag']:<20}{r['normalize']:>6}{frac:>6}{r['k']:>3}"
          f"{r['val']:>9.4f}{r['test']:>9.4f}{r['dead']:>4}/{len(r['alpha'])}")

# frac 곡선만 따로: 요구 스윙 = n*V_T / frac  (n=1, V_T=26mV)
lse = sorted((r for r in rows if r['normalize'] == 'xlse'),
             key=lambda r: r['lse_temp_frac'])
if lse:
    print(f"\\n{'frac':>6}{'요구 스윙':>10}{'test':>9}")
    for r in lse:
        f = r['lse_temp_frac']
        print(f"{f:>6.2f}{26.0/f:>8.0f}mV{r['test']:>9.4f}")"""),

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
