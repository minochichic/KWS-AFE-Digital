"""Generate notebooks/lab.ipynb -- gpu_local's runner plus the analysis cells.

Two notebooks grew in parallel without knowing about each other: gpu_local.ipynb
has the run history, `run()`, result.json and the aggregation; workbench.ipynb
has the window-offset, level-calibration and frac-sweep measurements. Running
both is a hazard -- they write the same runs/ directory with resume=True but
disagree about the baseline (CONFIRMED is k=2, BASE was k=1), so the same tag
could be continued under a different setting.

lab.ipynb merges them. Neither original is touched, so any run started from
them still reproduces.

The runner is gpu_local's verbatim, deliberately: the whole recorded history
(af_k2, af_lse013..550) was produced by `run(tag, over_dict)` with
`model.in_channels` derived from n_channels x k, and result.json is the only
place test accuracy is ever written down. Changing that signature would orphan
the results.

Written as a generator so the cells stay reviewable in git and the preflight
cannot drift from what the code requires.
"""
from __future__ import annotations

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "lab.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


CELLS = [

md("""# BinaryMatchboxNet KWS — 실험 노트북 (lab)

`gpu_local`의 러너 + `workbench`의 측정 셀을 하나로 합친 것. 두 원본은 **그대로 둔다**
(거기서 시작한 런이 재현돼야 한다). 앞으로는 이 노트북만 쓴다.

**규칙**: 셀 3(프리플라이트)만 통과했으면 **어느 셀이든 단독 실행 가능**하다.
분석 셀은 전부 체크포인트에서 시작하고, 셀 사이로 변수를 넘기지 않는다.

### 순차 실행해도 안전하다

**학습을 시작하는 셀은 하나도 없다.** §7의 `run(...)`은 전부 주석이고 돌릴 줄 하나만
직접 풀어야 한다 — 100 에폭짜리가 실수로 시작되면 안 되기 때문이다.

| 셀 | 하는 일 | 시간 |
|---|---|---|
| §1–2 | pull + 프리플라이트 | 초 |
| §3 | 단위 테스트 · 데이터셋 | 1분 |
| §5 | **결과 표** (백필 포함) | 초~1분 |
| §6a | 창 위치 곡선 | 수 분 |
| §6c | 레벨 보정 | 1~2분 |
| §6d | **frac 스윕** | 수 분 |

> **커널을 재시작했으면 반드시 셀 1부터.** `git pull`은 프로젝트 모듈을 import 하기
> *전에* 일어나야 하고, 이미 import 된 모듈은 pull 해도 안 바뀐다."""),

md("## 1. 셋업"),
code("""# --- repo 루트로 이동 + 최신 코드 ---------------------------------------
import os, sys, subprocess, pathlib
os.chdir(os.path.expanduser('~/KWS-AFE-Digital')); sys.path.insert(0, os.getcwd())
print(subprocess.run(['git','pull'], capture_output=True, text=True).stdout.strip())
print(subprocess.run(['git','log','--oneline','-1'], capture_output=True,
                     text=True).stdout.strip())"""),

md("""## 2. 프리플라이트 — 파일이 아니라 **로드된 모듈**을 검사한다

파일만 보면 stale 커널을 못 잡는다. 그것 때문에 100 에폭을 두 번 버렸다."""),
code('''# --- 프리플라이트 + 공용 import (여기서 멈추면 아래는 돌리지 말 것) ------
import torch, numpy as np
from train.config import load_config, AFEConfig
from data.speech_commands import build_dataloaders, ensure_dataset, class_names
from data.afe import AFEFrontend, collect_init_batch, load_afe_state
from models.binary_matchboxnet import BinaryMatchboxNet
from train.train import Trainer, set_seed
import data.afe as _A
import experiments.window_offset as _W
import experiments.level_calibration as _L

_NEED = [
    ('spice 경로 analog/ 이전', 'train/config.py', 'analog/AFE/artifacts',
     lambda: AFEConfig().spice_matrix_path.startswith('analog/')),
    ("normalize='xmix'", 'data/afe.py', 'def _xmix',
     lambda: hasattr(_A.AFEFrontend, '_xmix')),
    ("normalize='xlse'", 'data/afe.py', 'def _xlse',
     lambda: hasattr(_A.AFEFrontend, '_xlse')),
    ('effective_alpha()', 'data/afe.py', 'def effective_alpha',
     lambda: hasattr(_A.AFEFrontend, 'effective_alpha')),
    ('비교기 k개', 'train/config.py', 'comparators_per_channel',
     lambda: 'comparators_per_channel' in AFEConfig.__dataclass_fields__),
    ('창 위치 곡선', 'experiments/window_offset.py', 'def offset_curve',
     lambda: hasattr(_W, 'offset_curve')),
    ('레벨 보정', 'experiments/level_calibration.py', 'def level_stats',
     lambda: hasattr(_L, 'level_stats')),
    ('δ 안정화', 'data/afe.py', 'def collect_init_batch',
     lambda: hasattr(_A, 'collect_init_batch')),
]
_disk, _mem = [], []
for name, f, needle, check in _NEED:
    on_disk = needle in pathlib.Path(f).read_text()
    in_mem = bool(check())
    print(f"{'OK ' if in_mem else 'X  '} {name:<22} 파일 {'O' if on_disk else 'X'}"
          f"  모듈 {'O' if in_mem else 'X'}")
    (_disk if not on_disk else _mem if not in_mem else []).append(name)
if _disk:
    raise RuntimeError(f"파일에 없음 -> git pull 이 안 됐습니다: {_disk}")
if _mem:
    raise RuntimeError(f"파일엔 있는데 모듈엔 없음 -> **커널을 재시작**하세요: {_mem}")

DATA_ROOT = os.path.expanduser('~/datasets/speech_commands_v2')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SR  = AFEConfig().sample_rate

# 확정 설정 = configs/base.yaml 그 자체다. 이 딕셔너리는 이제 비어 있고,
# 남겨둔 건 실험 셀이 {**CONFIRMED, ...} 형태를 계속 쓸 수 있게 하기 위해서다.
#
# base.yaml 이 spice + sqrt + xlse + k=1 로 확정됐다 (2026-08-11). 다이오드-OR 은
# max 가 아니라 log-sum-exp 를 만들고, 그 soft max 로 학습한 쪽이 하드 max 보다
# +6.7pp 높다 (0.8445 vs 0.7778) -- 부품은 하나도 안 늘었다.
#
# ⚠️ 옛 xmix 런을 이 딕셔너리로 재현하려 하지 말 것. af_k2 는 xmix/k=2 라 32행이고
#    resume 이 형상 불일치로 막는다. 옛 런은 runs/<tag>/config.yaml 로 평가한다.
CONFIRMED = {}
BEST = {'afe.lse_temp_frac': 0.78}     # frac 은 아직 확정 아님 (평탄대 13~50 mV)
XMIX = {'afe.normalize': 'xmix'}       # 하드 max 대조군이 필요할 때만

# ── 트랙 2: 다이오드-OR 없는 고정 임계 ────────────────────────────────────
# 정규화 자체를 빼고, 채널마다 자기 R7/R8 분압으로 절대 전압과 비교한다.
# 다이오드 16개 + 버퍼 + 공유 V_ref 가 통째로 사라진다 (대신 분압이 16벌).
# 동료 회로가 다이오드-OR 을 못 해줄 때의 대비이자, "정규화가 정말 필요한가"의 대조군.
#
# ⚠️ 기록된 fixed 0.726 은 **낡았다** (2026-08-05, Stage 3). 그 뒤 spice 필터뱅크,
#    sqrt 압축, f_max 8000, 게인 증강이 전부 들어왔고 그 넷은 트랙 2 에도 적용된다.
#    그래서 0.726 을 트랙 2 의 예상치로 쓰지 말 것 -- 다시 재야 한다.
#
# ⚠️ `fixed_scale_quantile` 을 **반드시** 같이 준다. 기본값 1.0 = 전역 max 인데,
#    그건 이상치 클립 하나라 전형적 클립이 [0,1] 의 얇은 조각으로 눌린다.
#    threshold 가 0.002~0.010 에 앉고 Adam 한 스텝이 그걸 15% 씩 움직여서
#    학습이 요동친다 (data/afe.py init_fixed_scale 주석에 이미 적혀 있다).
#    실제로 이걸 빼고 돌린 fx_g12 가 0.5763 + 죽은 비교기 4개로 끝났다.
#    과거 q 스윕: q=1.0 → 0.693 / **q=0.75 → 0.726** / q=0.6 → 0.653.
FIXED = {'afe.normalize': 'fixed',     # lse_temp_frac / xmax_floor_frac 은 무관해진다
         'afe.fixed_scale_quantile': 0.75}

# δ 추정에 쓸 클립 수. 배치 하나(128)로는 2% 분위수가 시드에 2.4배 흔들렸다.
N_INIT_CLIPS = 2048

print(f"\\n모듈 정상.  DATA_ROOT={DATA_ROOT}  존재={os.path.isdir(DATA_ROOT)}")
print('torch', torch.__version__, '| cuda',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(CPU only)')'''),

md("""### 확정 설정 — `configs/base.yaml` 이 곧 baseline 이다

전에는 노트북의 `CONFIRMED` 딕셔너리가 진짜 설정을 들고 있었고 `base.yaml` 은
`minmax` 였다. 이제 반대다: **`base.yaml` 이 확정 설정이고**, 실험 셀은 거기서
필요한 키만 바꾼다. 두 곳에 따로 적혀 있으면 언젠가 어긋난다.

| | 값 | 왜 |
|---|---|---|
| `filterbank_source` | `spice` | 실제 GIC 응답 (mel 삼각형 아님) |
| `compression` | `sqrt` | 검출기가 진폭에 선형 |
| **`normalize`** | **`xlse`** | **다이오드-OR 은 max 가 아니라 soft max 다. +6.7pp** |
| `comparators_per_channel` | 1 | k=2 는 업그레이드 경로 |
| `f_min` / `f_max` | 50 / 8000 Hz | 125–5000 은 −1.6pp |
| **`xmax_floor_frac`** | **0 (δ=0)** | **§6e 로 확정.** LSE 자체 바닥 + δ가 offset 아래 |

**`xlse` 가 baseline 인 이유**: 하드 max(`xmix`)로 학습한 0.7778 은 **평범한
다이오드-OR 로는 만들 수 없는 프론트엔드**의 값이다. 회로가 실제로 만드는 soft max 로
학습하니 **0.8445** 였다 — 부품은 하나도 안 늘었다. 자세히:
[`docs/EXPERIMENT_MAP.md`](../docs/EXPERIMENT_MAP.md) §A-1.

> **남은 미확정은 `lse_temp_frac` 하나뿐이다** (평탄대 13–50 mV 중 어디).
> δ는 §6e 로 끝났다 — `V_ref` 를 detector 정지점에 두면 된다."""),

md("## 3. 환경 점검 (처음 한 번)"),
code("""# 의존성 — torch 를 다운그레이드하지 않도록 requirements-colab.txt 를 쓴다
# !{sys.executable} -m pip install -q -r requirements-colab.txt

!{sys.executable} -m pytest -q"""),
code("""import collections
ensure_dataset(DATA_ROOT)
_cfg = load_config('configs/base.yaml'); _cfg.data.root = DATA_ROOT
_tr, _va, _te = build_dataloaders(_cfg.data, _cfg.train.batch_size,
                                  _cfg.afe.sample_rate)
print('classes:', class_names())
for nm, dl in [('train', _tr), ('val', _va), ('test', _te)]:
    print(f'{nm:5s}: {len(dl.dataset):>6} clips, {len(dl)} batches')"""),

md("""## 4. 러너 — `run(tag, over)`

`gpu_local`에서 그대로 가져왔다. 기록된 런(`af_k2`, `af_lse013`~`550`)이 전부 이
서명으로 만들어졌고 `result.json`이 **test 정확도가 남는 유일한 곳**이라, 여기를
바꾸면 기존 결과가 고아가 된다.

학습 **전에** 막아주는 것:

| 검사 | 막는 사고 |
|---|---|
| `model.in_channels` 자동 계산 | `n_channels × k`와 어긋나 config 에러 |
| `init_fixed_scale` → `init_thresholds` 순서 | δ 없이 α 초기화 |
| `d > 0` (요구했을 때만) | `floor_frac` override 미반영으로 100 에폭 헛돌기 |
| α 범위 | 저항비로 만들 수 없는 값 |

**δ 안정화**: `init_fixed_scale`에 배치 하나가 아니라 **2048클립**을 넘긴다.
δ는 2% 분위수라 배치 하나로는 시드에 **2.4배** 흔들렸고(0.00100 ↔ 0.00244), 그 값은
동료가 만들 **V_ref 오프셋**이다. 학습 시작 전에 **서로 겹치지 않는 두 절반**으로
δ를 각각 재서 아직도 흔들리면 바로 보여준다."""),
code('''# --- 공용 러너 -----------------------------------------------------------
import json, yaml

_NEEDS_SCALE = ('fixed', 'agc', 'xmax', 'xmix', 'xlse')
_NEEDS_FLOOR = ('xmax', 'xmix', 'xlse')

def build_cfg(tag, over=None, epochs=None):
    """run() 과 floor_probe 가 같은 config 를 쓰도록 한 곳에 모은다."""
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
    return cfg, n_ch, k


def run(tag, over=None, epochs=None):
    """학습 1회. over 는 dotted override dict."""
    cfg, n_ch, k = build_cfg(tag, over, epochs)
    set_seed(cfg.train.seed)

    afe = AFEFrontend(cfg.afe)
    model = BinaryMatchboxNet(cfg.model)
    tr, va, te = build_dataloaders(cfg.data, cfg.train.batch_size,
                                   cfg.afe.sample_rate, seed=cfg.train.seed)
    # δ 는 2% 분위수라 배치 하나(128클립)로는 시드에 2.4배 흔들린다 (0.00100 ↔
    # 0.00244). 그 값은 동료가 만들 V_ref 오프셋이므로 클립을 더 모아 추정한다.
    # α(채널 평균)와 T(중앙값)는 한 배치로도 충분하지만 같은 배치를 쓴다.
    w = collect_init_batch(tr, N_INIT_CLIPS)
    if cfg.afe.normalize in _NEEDS_SCALE:
        afe.init_fixed_scale(w)        # 순서 중요: δ 먼저
    afe.init_thresholds(w)             # 그 다음 α

    # 안정성을 눈으로 확인한다: 서로 겹치지 않는 두 절반이 같은 δ 를 주는가.
    if (cfg.afe.normalize in _NEEDS_FLOOR and w.shape[0] >= 4
            and cfg.afe.xmax_floor_frac > 0):   # δ=0 은 잴 게 없다
        _h = w.shape[0] // 2
        _probe = AFEFrontend(cfg.afe)
        _d2 = []
        for _half in (w[:_h], w[_h:]):
            _probe.init_fixed_scale(_half)
            _d2.append(float(_probe.xmax_floor))
        _rat = max(_d2) / max(min(_d2), 1e-12)
        print(f'   δ 반쪽 검사: {_d2[0]:.5f} / {_d2[1]:.5f}  = {_rat:.2f}x'
              + ('' if _rat < 1.15 else '   ⚠️ 아직 흔들린다 — N_INIT_CLIPS 를 늘릴 것'))

    info = [f'{n_ch}채널 x {k}비트 = {n_ch*k}행',
            f'{sum(p.numel() for p in model.parameters()):,} params']
    if cfg.afe.normalize in _NEEDS_FLOOR:
        d = float(afe.xmax_floor)
        # d=0 은 이제 정당한 확정값이다 (floor_frac=0). 잡아야 할 건 "0 을 요구하지
        # 않았는데 0 이 나온" 경우 -- override 가 안 먹었거나 커널이 stale 한 것이다.
        assert d > 0 or cfg.afe.xmax_floor_frac == 0, (
            f'd=0 인데 afe.xmax_floor_frac={cfg.afe.xmax_floor_frac} 이다 — '
            f'override 가 반영되지 않았다 (커널 재시작 필요?)')
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
           'aug_gain_db': list(getattr(cfg.data, 'aug_gain_db', []) or []),
           'aug_time_shift_ms': float(getattr(cfg.data, 'aug_time_shift_ms', 0.0)),
           'alpha': [round(float(x), 4) for x in alpha],
           'dead': int(((alpha >= 0.99) | (alpha <= 0.01)).sum())}
    pathlib.Path(run_dir, 'result.json').write_text(
        json.dumps(rec, ensure_ascii=False, indent=2))
    return rec

print('build_cfg / run(tag, over) / save_result 준비됨')'''),

md("""## 5. 분석 공통 — 체크포인트에서만 시작

> **설정은 `runs/<tag>/config.yaml`에서 읽는다** — `CONFIRMED`가 아니라.
> `xmix` 런과 `xlse` 런이 섞여 있고, 틀린 정규화로 평가해도 **에러 없이
> 그럴듯한 숫자**가 나온다."""),
code('''def load_run(tag, **over):
    """그 런이 실제로 학습한 config로 복원한다."""
    saved = f'runs/{tag}/config.yaml'
    if not os.path.isfile(saved):
        raise FileNotFoundError(f'{saved} 없음 — 그 태그로 학습한 적이 있는지 확인')
    cfg = load_config(saved, dict(over))
    cfg.data.root = DATA_ROOT
    afe = AFEFrontend(cfg.afe).to(DEV).eval()
    model = BinaryMatchboxNet(cfg.model).to(DEV).eval()
    ck = torch.load(f'runs/{tag}/best.pt', map_location=DEV, weights_only=True)
    model.load_state_dict(ck['model'])
    load_afe_state(afe, ck['afe'])      # 옛 런의 deadzone 키를 안전하게 흘린다
    return cfg, afe, model, ck


def test_loader(cfg):
    return build_dataloaders(cfg.data, cfg.train.batch_size,
                             cfg.afe.sample_rate, seed=cfg.train.seed)[2]


@torch.no_grad()
def accuracy(afe, model, loader, T, shift_ms=0.0, gain_db=0.0):
    """shift_ms > 0 = 소리가 늦게 들어옴 (밖으로 나간 부분은 버려진다)."""
    k = int(round(shift_ms * SR / 1000)); ok = n = 0
    g = 10.0 ** (gain_db / 20.0)
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        if k:
            x = torch.roll(x, k, dims=-1)
            if k > 0: x[..., :k] = 0.0
            else:     x[..., k:] = 0.0
        if gain_db: x = x * g
        ok += (model(afe(x, target_T=T)).argmax(1) == y).sum().item()
        n += y.numel()
    return ok / n

print('load_run / test_loader / accuracy 준비됨')'''),

md("""## 6. 결과 표 — `result.json` 백필 + 집계

`result.json`이 도입되기 전에 끝난 런은 test가 파일에 없다. `best.pt`로 복원한다.
δ와 LSE 온도는 `register_buffer`라 체크포인트에 들어 있어 `init_fixed_scale`을
다시 부를 필요가 없다."""),
code('''# test 세트는 런 사이에 동일하지 "않다". unknown 서브샘플과 합성 silence 가
# 둘 다 seed 로 뽑히므로(data/speech_commands.py:162,166) 테스트 클립의 약 20%가
# seed 에 따라 바뀐다. 하나를 만들어 돌려쓰면 다른 seed 의 런을 남의 테스트
# 세트로 채점하게 된다 -- af_lse078_s2 가 0.8351 대신 0.8331 로 나온 이유다.
_TE_CACHE = {}

def _te_key(cfg):
    d = cfg.data
    return (cfg.train.seed, d.root, d.split, d.silence_fraction,
            d.unknown_fraction, cfg.train.batch_size, cfg.afe.sample_rate)

def backfill(tag):
    d = pathlib.Path('runs', tag)
    if not (d / 'best.pt').exists():
        print(f'  {tag}: best.pt 없음'); return None
    cfg = load_config(str(d / 'config.yaml'))      # 그 런이 실제로 쓴 설정
    cfg.data.root = DATA_ROOT
    key = _te_key(cfg)
    if key not in _TE_CACHE:
        _TE_CACHE[key] = test_loader(cfg)
    _TE = _TE_CACHE[key]
    t = Trainer(cfg, BinaryMatchboxNet(cfg.model), afe=AFEFrontend(cfg.afe))
    ck = torch.load(d / 'best.pt', map_location=t.device, weights_only=True)
    t.model.load_state_dict(ck['model'])
    load_afe_state(t.afe, ck['afe'])
    rec = save_result(d, tag, cfg, float(ck['best_acc']),
                      float(t.evaluate(_TE)['acc']), t.afe.effective_alpha())
    print(f"  {tag}: test {rec['test']:.4f} 기록")
    return rec


def results(backfill_missing=True):
    failed = []
    if backfill_missing:
        # 옛 런들은 지금 코드와 안 맞을 수 있다. 하나가 실패해도 나머지 30개의
        # 결과까지 날리면 안 되므로 런 단위로 잡고 끝에 모아 보고한다.
        # 경로 경고는 억누른다 -- analog/ 이전에 쓰인 config 라서 나오는 것이고,
        # 폴백이 맞는 파일을 찾은 것까지 확인된 상태다. 여기선 읽기만 한다.
        import warnings
        for p in sorted(pathlib.Path('runs').glob('*/config.yaml')):
            if (p.parent / 'result.json').exists():
                continue
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*spice_matrix_path.*')
                    backfill(p.parent.name)
            except Exception as e:
                failed.append((p.parent.name, f'{type(e).__name__}: {e}'.split(chr(10))[0]))
    rows = [json.loads(p.read_text())
            for p in sorted(pathlib.Path('runs').glob('*/result.json'))]
    if not rows:
        print('runs/ 가 비어 있다'); return []
    rows.sort(key=lambda r: -r['test'])
    print(f"{'run':<20}{'norm':>6}{'frac':>6}{'k':>3}{'seed':>6}"
          f"{'val':>9}{'test':>9}{'죽은':>8}")
    for r in rows:
        frac = f"{r['lse_temp_frac']:.2f}" if r['normalize'] == 'xlse' else '—'
        print(f"{r['tag']:<20}{r['normalize']:>6}{frac:>6}{r['k']:>3}"
              f"{r.get('seed','?'):>6}{r['val']:>9.4f}{r['test']:>9.4f}"
              f"{r['dead']:>5}/{len(r['alpha'])}")
    # frac 곡선: 요구 스윙 = n*V_T / frac  (n=1, V_T=26 mV)
    lse = sorted((r for r in rows if r['normalize'] == 'xlse'),
                 key=lambda r: r['lse_temp_frac'])
    if lse:
        print(f"\\n{'frac':>6}{'요구 스윙':>11}{'test':>9}   "
              f"(각 점이 서로 다른 모델임에 주의 — §6d 참조)")
        for r in lse:
            print(f"{r['lse_temp_frac']:>6.2f}{26.0/r['lse_temp_frac']:>9.0f}mV"
                  f"{r['test']:>9.4f}   {r['tag']}")
    if failed:
        print(f"\\n⚠️ 복원 실패 {len(failed)}개 (표에서 빠졌다):")
        for tag, msg in failed:
            print(f"   {tag:<24} {msg[:88]}")
    return rows

results()'''),

md("""## 6a. 창 위치 곡선 — 슬라이딩의 **진짜** 비용

FPGA는 100 ms마다 판정하는 슬라이딩 창을 쓴다. 한 단어가 창 8개에 걸리고 창마다
위치가 다르므로 그 변화를 견뎌야 한다.

단어를 **자르지 않고** 창 안에서만 옮기고 나머지는 그 클립 자신의 노이즈 플로어로
채운다. 위치는 정규화 p (0 = 왼쪽 밀착, 1 = 오른쪽 밀착).

채움은 세 가지고, **어느 걸 쓰느냐가 위치보다 더 크게 작용했다**:

| 채움 | 무엇 | 쓰임 |
|---|---|---|
| **`room`** | 실제 `_background_noise_` 크롭, 클립 자기 플로어로 스케일 | **배치 질문에 답하는 것** |
| `zero` | 디지털 무음 | `xmix`/`xlse`는 d를 **빼므로** 안전 (`xmax`는 아님) |
| `white` | 같은 RMS의 백색잡음 | ⚠️ 방의 모델로는 **틀렸다** — 아래 참조 |

**낙폭(`room` 기준) 읽는 법:**

| 낙폭 | 판단 |
|---|---|
| ≤ 5pp | 슬라이딩이 사실상 공짜. 최대 확신도 선택만 붙이면 끝 |
| 5–15pp | 최대 확신도 선택 필요 |
| ≥ 15pp | 오정렬이 진짜 문제 — **긴 캔버스** 증강을 만들 가치가 있음 |

> ⚠️ **`white`는 방이 아니다.** 첫 버전이 백색잡음을 썼고 `zero` 대비 **−15pp**가 나와
> 위치 효과(1.8pp)를 완전히 덮어버렸다. 상대 임계는 채널간 max로 나누는데 평평한
> 스펙트럼은 그 분모를 **16채널 전부에서 동시에** 들어올린다 — 음성 에너지가 적은
> 고역에서 특히. 실제 방 잡음은 저역에 몰려 있다. 버그였지만 **결과 자체는 진짜**다:
> **광대역 잡음은 상대 임계에 심각한 위협**이라는 뜻이고, 그래서 배경잡음 증강(§7-4)의
> 우선순위가 올라갔다."""),
code("""from experiments.window_offset import offset_curve, print_offset_curve

def offset_report(tag, steps=9, fills=('room','zero'), **over):
    cfg, afe, model, _ = load_run(tag, **over)
    res = offset_curve(afe, model, test_loader(cfg), cfg.model.T, steps=steps,
                       fills=fills, device=DEV, data_root=cfg.data.root)
    print_offset_curve(res, tag)
    return res

TAG = 'af_lse078'          # ← 위 결과 표에서 고른다
offset_report(TAG)

# 백색잡음까지 같이 보려면:
# offset_report(TAG, fills=('room','white','zero'))"""),

md("""## 6b. 시간 이동 곡선 — ⚠️ **잘림**을 잰다

1초 클립을 그냥 밀기 때문에 단어가 클립 밖으로 나가 **사라진다**. 이 곡선의 −11pp는
오정렬이 아니라 대부분 잘림이고, 증강으로 못 고친다(±300 증강 실측 **+0.33pp**).

**슬라이딩 판단에는 §6a를 쓴다.** 이 셀은 비교·기록용. 그림: `docs/diagrams/13_sliding_window.svg`."""),
code("""def shift_curve(tag, shifts=(-400,-300,-200,-100,0,100,200,300,400), **over):
    cfg, afe, model, _ = load_run(tag, **over)
    te = test_loader(cfg)
    base = accuracy(afe, model, te, cfg.model.T, 0.0)
    print(f'{tag}\\n{"이동(ms)":>9}{"test acc":>10}{"기준 대비":>11}')
    out = {}
    for s in shifts:
        a = accuracy(afe, model, te, cfg.model.T, s)
        out[s] = a
        print(f'{s:>9}{a:>10.4f}{(a-base)*100:>+10.1f}pp')
    return out

print('shift_curve(tag) 준비됨 — 판단은 §6a로')"""),

md("""## 6c. 레벨 보정 — GSC가 배치 음량 범위를 덮는가

`lse_temp_frac`은 고르는 값이 아니라 **1/음량**이다 (frac = n·V_T / 스윙, 분자는
물리가 26–39 mV로 고정). 프리앰프는 이미 ×10 상한이라 범위를 좁힐 수단이 없다:
**60~85 dB SPL = 25 dB = frac 0.55~0.03.**

GSC는 자원자마다 다른 폰·거리로 녹음됐고 로더가 레벨을 정규화하지 않으므로
**우연히** 그 범위를 이미 줄 수도 있다.

**§0의 앵커 점검도 같이 본다** — `init_fixed_scale`이 무음 프레임까지 섞어 중앙값을
내므로 앵커가 아래로 끌려갔을 수 있고, 같은 함수의 2% 분위수(δ)는 배치 하나(128클립)로
추정돼 시드에 **2.4배** 흔들린 전력이 있다."""),
code("""from experiments.level_calibration import level_stats, print_level_report

def level_report(tag, split='train', spl=74.0, **over):
    cfg, afe, _, _ = load_run(tag, **over)
    tr, _, te = build_dataloaders(cfg.data, cfg.train.batch_size,
                                  cfg.afe.sample_rate, seed=cfg.train.seed)
    st = level_stats(afe, tr if split == 'train' else te,
                     cfg.afe.compression, DEV)
    print_level_report(st, float(cfg.afe.lse_temp_frac), spl, f'{tag} [{split}]')
    return st

TAG = 'af_lse078'
level_report(TAG)"""),

md("""## 6d. frac 스윕 — ⭐ **하나의 모델**이 음량 범위를 견디나

§6의 frac 표는 평평하지만, 그 평평함은 **각 점이 서로 다른 모델**이라는 뜻이다.
`af_lse078`은 frac 0.78에서 학습된 모델이고 `af_lse200`은 2.00에서 학습된 다른
모델이다. **배치에는 모델이 하나뿐**이고 그 하나가 화자 음량에 따라 frac 전체를
만난다 — 그러니 §6 표는 "어느 frac으로 학습해도 좋다"는 뜻이지
**"한 모델이 모든 frac을 견딘다"가 아니다.**

`lse_temp`는 **버퍼**라 테스트 시점에 바꿀 수 있다. 즉 재학습 없이 그 질문에 답한다.
frac ∝ 1/음량이므로 배율 m은 곧 **−20·log₁₀(m) dB SPL**이다.

| 낙폭 | 뜻 |
|---|---|
| < 5pp | 앵커 무관. 기록된 정확도가 그대로 유효하고, 동료에게 줄 스윙 스펙도 헐렁하다 |
| ≥ 5pp | 그 정확도는 **한 가지 음량에서만** 성립 → 게인 증강 필요 |"""),
code('''import math

@torch.no_grad()
def frac_sweep(tag, mults=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0), **over):
    cfg, afe, model, _ = load_run(tag, **over)
    if cfg.afe.normalize != 'xlse':
        print(f'{tag}: normalize={cfg.afe.normalize} — lse_temp 없음. '
              f'xlse 런에만 쓴다.')
        return None
    te = test_loader(cfg)
    t0, f0 = float(afe.lse_temp), float(cfg.afe.lse_temp_frac)
    # 학습점(m=1)을 먼저 잰다 — 루프 안에서 잡으면 m<1 행에 기준이 없다.
    base = accuracy(afe, model, te, cfg.model.T)
    print(f'{tag}   학습 frac {f0}   T={t0:.5f}   학습점 {base:.4f}\\n')
    print(f"{'배율':>6}{'frac':>8}{'요구 스윙':>11}{'≈음량':>8}{'test':>9}{'대비':>11}")
    out = {}
    for m in mults:
        afe.lse_temp.fill_(t0 * m)
        a = accuracy(afe, model, te, cfg.model.T)
        out[m] = a
        print(f'{m:>6.2f}{f0*m:>8.3f}{26.0/(f0*m):>9.0f}mV'
              f'{-20*math.log10(m):>+7.0f}dB{a:>9.4f}{(a-base)*100:>+10.1f}pp')
    afe.lse_temp.fill_(t0)
    v = list(out.values())
    drop = max(v) - min(v)
    print(f'\\n낙폭 {drop*100:.1f}pp  → '
          + ('앵커 무관, 기록된 정확도 유효 ✅' if drop < 0.05
             else '앵커에 민감 — 게인 증강 필요 ⚠️'))
    return out

TAG = 'af_lse078'
frac_sweep(TAG)'''),

md("""## 6e. floor_frac — δ 를 어디에 앉힐 것인가 (학습 없음, 몇 초)

**안정화만으로는 안 끝난다.** 클립을 2048개로 늘린 건 δ 가 **덜 흔들리게** 만든 것이고,
지금 δ 는 여전히 **잘못된 것을 재고 있다**:

| | |
|---|---|
| 무음 프레임 | **2.5%** |
| `xmax_floor_frac` | **2.0%** |

2% quantile 을 요구하는데 무음이 2.5%다 → quantile 이 **atom(= `sqrt` guard) 안**에
들어간다. 표본을 늘리면 δ 가 **가드 값으로 안정적으로 수렴**할 뿐이다 —
재현은 되지만 **zero padding 을 잰 값**이라 회로 스펙으로 못 쓴다.
그림: `docs/diagrams/15_delta.svg`, `16_quiescent.svg`.

**그래서 sweep 이 필요하다.** 다만 대부분은 공짜다 — δ 가 atom 을 벗어나는지, α 가
죽는지는 **init 만 해보면** 알 수 있다. 100 에폭을 태워서 "0.05 는 채널 4개를 죽인다"를
알아낼 이유가 없다.

| 열 | 보는 법 |
|---|---|
| `guard 대비` | **1.0x 면 atom 안** = δ 가 padding 을 잰 것 |
| `죽은 α` | init 에서 이미 0/1 에 붙은 채널. xmix 에서 0.05 는 4개를 죽였다 |
| `δ (mV)` | `lse_temp ≡ n·V_T = 25.85 mV` 로 환산 (xlse 만) |"""),
code('''import warnings

@torch.no_grad()
def floor_probe(fracs=(0.0, 0.02, 0.03, 0.05, 0.10), over=None, n_clips=None):
    """floor_frac 을 바꾸면 δ 와 α 가 어디 앉는지. 같은 클립으로만 비교한다."""
    over = dict(over if over is not None else BEST)
    cfg0, _, _ = build_cfg('probe', over)
    tr = build_dataloaders(cfg0.data, cfg0.train.batch_size,
                           cfg0.afe.sample_rate, seed=cfg0.train.seed)[0]
    w = collect_init_batch(tr, n_clips or N_INIT_CLIPS)
    guard = 1e-3 if cfg0.afe.compression == 'sqrt' else 0.0
    print(f'{cfg0.afe.normalize}  클립 {w.shape[0]}개  guard={guard:.5f}\\n')
    print(f"{'floor_frac':>11}{'δ (code)':>11}{'δ (mV)':>9}{'guard 대비':>11}"
          f"{'α 범위':>20}{'죽은 α':>9}")
    for f in fracs:
        cfg, _, _ = build_cfg('probe', {**over, 'afe.xmax_floor_frac': f})
        fe = AFEFrontend(cfg.afe)
        with warnings.catch_warnings():        # 가드 경고는 아래 열이 대신 말한다
            warnings.simplefilter('ignore')
            fe.init_fixed_scale(w); fe.init_thresholds(w)
        d = float(fe.xmax_floor)
        mv = (f'{d / float(fe.lse_temp) * 25.852:>8.3f}'
              if cfg.afe.normalize == 'xlse' else f'{"—":>8}')
        rat = (f'{d / guard:>9.2f}x' if guard > 0 and d > 0 else f'{"—":>10}')
        a = fe.effective_alpha()
        dead = int(((a >= 0.99) | (a <= 0.01)).sum())
        flag = '  ← atom 안' if guard > 0 and 0 < d <= 1.05 * guard else ''
        print(f'{f:>11.2f}{d:>11.5f}{mv}{rat}'
              f'{f"{a.min():.3f}~{a.max():.3f}":>20}{dead:>6}/{a.numel()}{flag}')

floor_probe()'''),

md("""## 6f. 게인 스윕 — **두 트랙의 공통 축** (학습 없음)

§6d 의 `frac_sweep` 은 `xlse` 전용이다 (`fixed` 에는 frac 이 없다). 두 트랙을 나란히
놓으려면 **둘 다에 물리적으로 의미가 같은** 축이 필요한데, 그게 입력 게인이다 —
말하는 사람이 마이크에서 멀어지거나 가까워지는 것.

`xlse` 에서는 게인 스윕과 frac 스윕이 δ=0 일 때 **정확히 같은 것**임을 이미 확인했다
(수치로 1e-7 이내). 그러니 게인 축 하나로 두 트랙을 같은 그림에 올릴 수 있다.

| 트랙 | 음량 방어 | 기대 |
|---|---|---|
| `xl_g12` (xlse) | **구조적** 부분 불변 + 증강 | 완만한 낙폭 |
| `fx_q75` (fixed) | **증강뿐** (분모가 없다) | 가파를 것 — 얼마나인지가 질문 |

> `fixed` 는 `_DIVIDER_FORM` 이 아니라 **threshold 클램프가 없다.** `xmix` 에서
> 채널이 죽은 적이 있어서 클램프를 넣었는데(alpha 1.344), `fixed` 에는 그 보호가
> 없다. 학습 뒤 `dead_channels()` 로 관측 범위 밖으로 걸어나간 채널이 있는지 본다."""),
code('''def gain_sweep(tag, gains=(-18,-12,-6,-3,0,3,6,12,18), **over):
    """정규화 방식과 무관한 공통 축: 입력 게인 [dB]."""
    cfg, afe, model, _ = load_run(tag, **over)
    T, ld = cfg.model.T, test_loader(cfg)
    print(f'=== {tag} ({cfg.afe.normalize}) ===')
    print(f'{"gain[dB]":>9}{"test":>9}{"Δ vs 0":>9}')
    base = None
    rows = []
    for g in gains:
        a = accuracy(afe, model, ld, T, gain_db=float(g))
        if g == 0:
            base = a
        rows.append((g, a))
    for g, a in rows:
        d = '' if base is None else f'{(a - base) * 100:+8.1f}'
        print(f'{g:>9}{a:>9.4f}{d:>9}')
    accs = [a for _, a in rows]
    print(f'\\n평균 {sum(accs) / len(accs):.4f}   최저 {min(accs):.4f}   '
          f'낙폭 {(max(accs) - min(accs)) * 100:.1f}pp')
    return rows


@torch.no_grad()
def dead_channels(tag, n_clips=512, **over):
    """threshold 가 그 채널의 관측 엔벨로프 범위 밖으로 나갔는가.

    밖이면 그 채널 비교기는 항상 0 이거나 항상 1 이다 -- 정확도에는 조용히
    반영되고 하드웨어에서는 납땜해봐야 아무 일도 안 한다.
    """
    cfg, afe, model, _ = load_run(tag, **over)
    env = afe._envelopes_chunked(
        collect_init_batch(test_loader(cfg), n_clips=n_clips).to(DEV), raw=False)
    lo = env.amin(dim=(0, 2))           # [C] 채널별 관측 최소
    hi = env.amax(dim=(0, 2))
    thr = afe.threshold.detach().view(-1)
    print(f'=== {tag} ({cfg.afe.normalize}) — {n_clips} 클립 ===')
    print(f'{"ch":>3}{"thr":>9}{"env min":>10}{"env max":>10}   판정')
    n_dead = 0
    for c in range(thr.numel()):
        t, l, h = thr[c].item(), lo[c].item(), hi[c].item()
        if t <= l:
            v, n_dead = '항상 1 (죽음)', n_dead + 1
        elif t >= h:
            v, n_dead = '항상 0 (죽음)', n_dead + 1
        else:
            frac_above = (env[:, c] > t).float().mean().item()
            v = f'살아있음 (발화 {frac_above * 100:.1f}%)'
        print(f'{c:>3}{t:>9.4f}{l:>10.4f}{h:>10.4f}   {v}')
    print(f'\\n죽은 채널 {n_dead}/{thr.numel()}')
    return n_dead

print('gain_sweep / dead_channels 준비됨')'''),

md("""## 7. 학습 — ⚠️ 전부 주석

돌릴 줄 **하나만** 풀고, 끝나면 다시 주석 처리한다. 각 100 에폭.

**P0 가 두 개를 숫자로 요구했다:**

| 실험 | 근거 | 측정값 |
|---|---|---|
| **게인 증강** | §6d frac 스윕 낙폭 | **27.4pp** (−12 dB 에서 −10.3pp) |
| **배경잡음 증강** | §6a `room` 채움 | **−9.9pp** (처음 보는 배경 종류) |

### 순서가 중요하다 — ① 기준선부터

`af_lse078` 은 **`floor_frac=0.02` 로 학습**됐는데 지금 baseline 은 **δ=0** 이다.
그 차이는 작지만(정규화값 <0.6pp), 기준선으로 쓰면 증강 효과와 δ 변경이 **섞인다.**
→ `xl_d0` 를 한 번 돌려 **깨끗한 비교점**을 만든다. 이후 전부 여기에 붙는다.

### 확인하는 법

증강이 먹었는지는 **학습이 끝난 뒤 §6d / §6a 를 그 런에 다시 돌려서** 본다:

| 돌린 것 | 확인 | 성공 기준 |
|---|---|---|
| `xl_g12` | `frac_sweep('xl_g12')` | 낙폭이 **27.4pp 보다 작아짐** |
| `xl_nz` | `offset_report('xl_nz')` | `room` 이 `zero` 에 **가까워짐** |

> ⚠️ **잡음 증강의 한계**: GSC 의 `_background_noise_` 는 파일이 **6개뿐**이고,
> 증강도 §6a 의 `room` 채움도 **같은 6개**에서 뽑는다. 그래서 학습 뒤 `room` 이
> 좋아지는 건 일부가 "그 6개를 외운" 것이다. 문헌 표준이긴 하지만, **진짜 처음 보는
> 배경**에 대한 값은 아니다 — 보드에서 실측할 때 다시 봐야 한다.

### 나중에

| # | 실험 | 조건 |
|---|---|---|
| k=2 + `xlse` | 하드 max 의 +2.5pp 가 soft max 와 **겹치는지 미검증** | 전력 여유 있으면 |
| `spice_gain_restore` | soft max 는 채널간 절대 스케일을 섞는다 | 항상 값어치 |
| `lse_temp_frac` 재확정 | 평탄대 13–50 mV 중 어디 | **게인 증강 뒤에** — 곡선이 평평해지면 이 스펙도 헐렁해진다 |"""),
code("""# ⚠️ 전부 주석이다. 순차 실행이 100 에폭을 시작하면 안 된다.
#    돌릴 줄 하나만 풀고, 끝나면 다시 주석 처리한다.
#
# base.yaml 이 이미 확정본이다 (spice + sqrt + xlse + k=1 + δ=0).
# BEST 는 lse_temp_frac 하나만 들고 있다.

# ── ① 기준선 (δ=0). 아래 전부가 여기에 붙는다 ─────────────────────────────
# run('xl_d0', BEST)

# ── ② 게인 증강 — §6d 가 ±12 dB 를 요구했다 ───────────────────────────────
# run('xl_g12', {**BEST, 'data.aug_gain_db': [-12.0, 12.0]})
# run('xl_g6',  {**BEST, 'data.aug_gain_db': [-6.0, 6.0]})    # ±12 가 중심을 해치면

# ── ③ 배경잡음 증강 — §6a 가 −9.9pp 를 보였다 ─────────────────────────────
# run('xl_nz', {**BEST, 'data.aug_noise_prob': 0.8})
#     # SNR 은 aug_noise_snr_db 기본값 [5, 30] dB 를 쓴다

# ── ④ 둘 다 (②③ 이 각각 도움이 됐을 때만) ────────────────────────────────
# run('xl_g12_nz', {**BEST, 'data.aug_gain_db': [-12.0, 12.0],
#                           'data.aug_noise_prob': 0.8})

# ── ⑤ τ ablation — CLAUDE.md 3.2 의 2 단계 ────────────────────────────────
#     EMA 는 dt = stft_hop_ms = 10 ms 위에서 돈다: alpha = 1-exp(-10/tau).
#     그래서 tau <= 1 ms 는 alpha ~ 1, 즉 **baseline 과 구분 불가**다 -- 동료가
#     0.5~1 ms 로 맞추면 우리 tau=0 모델이 이미 그 하드웨어의 모델이다.
#     잴 가치가 있는 것은 원래 보드의 3.16 ms 가 얼마나 손해였나 하나뿐이다.
# run('xl_tau3',  {**BEST, 'afe.envelope_tau_ms': 3.16})   # 구 하드웨어 (측정값)
# run('xl_tau10', {**BEST, 'afe.envelope_tau_ms': 10.0})   # 곡선 모양 보려고

# ── ⑥ 주파수 범위 — ⚠️ spice 경로에서는 f_min/f_max 가 안 읽힌다 ──────────
#     filterbank_source: spice 면 필터뱅크가 SPICE 행렬에서 오고, 그 행렬은
#     design_filterbank.py 의 F_MIN/F_MAX = 50/8000 에 박혀 있다. 125-5000 을
#     지금 baseline 에서 보려면 필터뱅크를 다시 설계해야 한다.
#     그래서 **mel 경로에서 먼저 가른다** -- 순수 config 변경이고, 주파수 범위
#     질문을 필터뱅크 모양 질문에서 분리한다. mel 에서도 지면 재설계 불필요.
# run('xl_mel50',  {**BEST, 'afe.filterbank_source': 'mel'})   # 대조군 (필수)
# run('xl_mel125', {**BEST, 'afe.filterbank_source': 'mel',
#                           'afe.f_min': 125.0, 'afe.f_max': 5000.0})
#     # 예전 -1.6pp 는 mel+sqrt+xmix 에서 잰 값이다. xlse 에서는 미측정.

# ── 나중에 ────────────────────────────────────────────────────────────────
# run('xl_k2', {**BEST, 'afe.comparators_per_channel': 2})
# run('xl_gr', {**BEST, 'afe.spice_gain_restore': True})

# ══ 트랙 2 — 다이오드-OR 없는 고정 임계 ═══════════════════════════════════
#   트랙 1(xlse)과 **같은 프론트엔드·같은 증강**으로 돌려야 비교가 성립한다.
#   fixed 는 정규화가 없어 음량에 절대적으로 민감하므로, 게인 증강이
#   유일한 방어다 -> ②를 주력으로 본다. ①은 증강의 기여를 재는 대조군.
#   ⚠️ fx_g12 (q=1.0, 태그 재사용 금지) 는 0.5763 + 죽은 비교기 4개로 폐기됐다.
#      원인은 모델이 아니라 스케일 조건수 -- FIXED 주석 참조. 새 태그로 간다.
# run('fx_d0',   FIXED)                                          # 증강 없음 (대조군)
# run('fx_q75',  {**FIXED, 'data.aug_gain_db': [-12.0, 12.0]})   # ★ 트랙 2 본선
#   폐기된 런은 지운다 (§9): shutil.rmtree('runs/fx_g12')

# ── 끝나면: 표 + 증강이 실제로 먹었는지 ───────────────────────────────────
# results()
# frac_sweep('xl_g12')        # 낙폭이 27.4pp 보다 작아졌나 (xlse 전용)
# offset_report('xl_nz')      # room 이 zero 에 가까워졌나
# gain_sweep('xl_g12'); gain_sweep('fx_q75')   # ★ 두 트랙 공통 축 (§6f)
# dead_channels('fx_q75')     # fixed 는 threshold 클램프가 없다 (§6f)"""),

md("""## 8. 하드웨어 내보내기 — α → 저항비

**반드시 `effective_alpha()`로 읽는다.** `xmix`/`xlse`는 α를 forward에서
straight-through로 클램프하므로 파라미터 원값은 [0,1] 밖으로 떠다닌다."""),
code('''def alpha_table(tag, RTOT=1e6, **over):
    cfg, afe, _, _ = load_run(tag, **over)
    a = afe.effective_alpha()
    fc = [166,295,447,631,832,1072,1349,1660,
          2042,2455,2951,3467,4169,4898,5754,6761]
    k = cfg.afe.comparators_per_channel
    extra = (f'  T={float(afe.lse_temp):.5f} (frac {cfg.afe.lse_temp_frac})'
             if cfg.afe.normalize == 'xlse' else '')
    print(f'{tag} [{cfg.afe.normalize}]  δ={float(afe.xmax_floor):.5f}{extra}  '
          f'Ra+Rb={RTOT/1e3:.0f} kΩ\\n')
    print(f"{'ch':>2}{'f_c[Hz]':>9}{'i':>3}{'α':>8}{'Rb[kΩ]':>9}{'Ra[kΩ]':>9}{'상태':>7}")
    for c in range(cfg.afe.n_channels):
        for i in range(k):
            x = float(a[c*k + i]); rb = RTOT*x
            st = '죽음' if x >= 0.99 or x <= 0.01 else '정상'
            print(f'{c:>2}{fc[c]:>9}{i:>3}{x:>8.4f}{rb/1e3:>9.1f}'
                  f'{(RTOT-rb)/1e3:>9.1f}{st:>7}')
    ok = bool(a.min() >= 0 and a.max() <= 1)
    print(f"\\n범위 {a.min():.3f}~{a.max():.3f}  "
          f"{'제작 가능 ✅' if ok else '제작 불가 ⚠️'}")

TAG = 'af_lse078'
alpha_table(TAG)'''),

code('''from export.afe_constants import dump_afe

def export_afe(tag, **over):
    """비교기 **왼쪽** 상수를 JSON 으로 — 동료·SPICE 스크립트용.

    FPGA 매니페스트와 혼동하지 말 것. 이건 아날로그 쪽 숫자고, ICD 대로
    경계를 넘지 않는다.
    """
    cfg, afe, _, _ = load_run(tag, **over)
    return dump_afe(afe, cfg, f'runs/{tag}/afe.json')

print('export_afe(tag) 준비됨')

# export_afe('xl_g12')      # -> runs/xl_g12/afe.json
# 그 다음 (numpy + ngspice 있는 곳에서):
#   python analog/AFE/scripts/spectrogram16.py --wav six.wav \\
#       --afe-json runs/xl_g12/afe.json'''),

md("""## 8a. 아날로그 SPICE 스윕 — 값 바꿔가며 회로 돌리기

`ngspice`가 있는 머신에서만 돈다 (Mac). **학습과 무관**하고 GPU 박스에서는 건너뛴다.

설계 의도는 두 개뿐이고 나머지는 따라온다:

```
gain = R5 / R4        검출기가 신호를 얼마나 키우나
tau  = R5 * C3        엔벨로프가 얼마나 빨리 잊나
```

`R5`가 둘 다 움직이므로, `tau_ms`를 주면 `C3`가 역산된다.

> **`V_ref`는 파라미터가 아니다.** 정지점이 `da + (R5/R4)(vf − da)`라
> **R4·R5를 바꾸면 같이 움직인다** — `BUILD_TABLE.md`의 918.4 mV는 원래
> 10k/47k에서만 맞다. `sim()`은 매번 정지점을 먼저 재고 거기서 `V_ref`를
> 잡으므로, δ가 학습에서 말하는 그 δ로 유지된다.

`tran=False`면 ~2초(동작점·마진만), `True`면 ~60초(rise·tau·ripple까지)."""),
code('''import sys; sys.path.insert(0, 'analog/AFE/scripts')
from sim_afe import sim, table

# ---- 여기 숫자만 바꾸면 된다 --------------------------------------------
runs = [
    sim("orig",  r4=10e3, r5=47e3,  tau_ms=4.7),          # 원래 보드
    sim("now",   r4=30e3, r5=350e3, tau_ms=0.7),          # 현재 값
    sim("pre5",  r4=30e3, r5=350e3, tau_ms=0.7, preamp=5.0),
]
table(runs)'''),

md("""### 쓸 수 있는 인자

| 인자 | 기본 | 뜻 |
|---|---|---|
| `r4` | 30e3 | 검출기 입력 저항. `gain = R5/R4` |
| `r5` | 350e3 | 피드백 저항. **gain 과 tau 둘 다** 움직인다 |
| `tau_ms` | 0.7 | 원하는 τ. `C3 = tau/R5` 로 역산 |
| `c3` | — | 직접 줄 수도 있다 (주면 `tau_ms` 무시) |
| `preamp` | 10.0 | 마이크 프리앰프 배율. `Rf = (G−1)·10k` |
| `freq` | 1349.0 | 시험 톤 [Hz]. 채널 중심들: 166·295·447·631·832·1072·1349·1660·2042·2455·2951·3467·4169·4898·5754·6761 |
| `delta_mv` | 0.0 | `V_ref − 정지점`. 학습의 `xmax_floor_frac` 에 대응 |
| `hardmax` | False | 다이오드-OR 대신 이상적 max (= `xmax`) |
| `tran` | True | False 면 동작점만, 60초 → 2초 |

**읽는 법**: `floor`는 LSE 바닥 `T·ln16`으로 **이득과 무관하게 고정**이다.
`minMrg = α_min × floor`가 최악 채널 무음 마진, `scatter`는 채널별 정지점
산포 `(R5/R4)·Vos + 비교기 오프셋`. **`head = minMrg/scatter`가 1배 아래로
내려가면 조용한 채널들이 무음에서 발화한다.** 이득을 올리면 scatter는 같이
커지는데 floor는 안 커지므로, 이득에는 이 대가가 붙는다."""),
code('''# 주파수: 채널 중심 몇 개에 톤을 넣어본다 (각 ~60초)
# fcs = [166, 631, 1349, 2951, 6761]
# table([sim(f"f{f}", r4=30e3, r5=350e3, tau_ms=0.7, freq=f) for f in fcs])

# τ 범위: 동료가 말한 0.5~1 ms 구간
# table([sim(f"tau{t}", r4=30e3, r5=350e3, tau_ms=t) for t in (0.5, 0.7, 1.0)])

# R5 범위: 300~400k
# table([sim(f"r5_{r}", r4=30e3, r5=r*1e3, tau_ms=0.7) for r in (300, 350, 400)])

# 프리앰프: 5배 vs 10배 vs 20배
# table([sim(f"pre{g}", r4=30e3, r5=350e3, tau_ms=0.7, preamp=g) for g in (5, 10, 20)])'''),

md("""## 8b. FPGA 비트폭 — RTL 을 쓰기 **전에** 잰다 (학습 없음, 몇 분)

이진 누산기는 `n = 2*popcount(XNOR) - N` 이라 이론상 `±N` (N = K·C_in/groups) 이다.
그 값으로 데이터패스를 잡으면 **맞지만 낭비**다 — 모든 입력이 모든 가중치와 일치해야
도달하는 값이고, 학습된 망은 거기 근처도 안 간다. residual 은 더 심하다:
`acc + res` 는 경계가 더해지므로 최악값이 두 배가 되는데, 실제로는 두 항이 동시에
극단으로 가지 않는다.

그래서 **잰다.** 실제 test 세트를 통과시켜 지점별 진짜 min/max 를 기록하고,
거기에 guard 비트를 얹어 폭을 정한다.

| 지점 | 무엇인가 |
|---|---|
| `binary` | 이진 conv 의 **alpha 적용 전 정수** 누산기 (`fuse.py` 의 threshold 가 이 값 기준) |
| `residual` | 블록 마지막 `post_bn` 의 입력 = `pw_acc + skip_acc`. 두 항 다 unscaled 라 정확히 정수 |
| `real` | int8 / 고정소수점 단 (conv1·conv3·conv4). 폭은 양자화 방식에서 나오므로 범위만 보고한다 |

> ⚠️ 잰 범위는 **본 데이터만큼만** 믿을 수 있다. `guard_bits=1` 이 기본이고,
> RTL 에는 saturate 를 같이 넣는다. 분포 밖 입력이 누산기를 wrap 시키면 안 된다."""),
code('''from export.ranges import measure_ranges, print_range_report, to_json

def bitwidths(tag, max_batches=None, guard_bits=1, save=True):
    """그 런의 체크포인트로 전 지점 정수 범위를 잰다."""
    cfg, afe, model, _ = load_run(tag)
    T = cfg.model.T
    loader = test_loader(cfg)

    @torch.no_grad()
    def feed():                      # 로더는 파형을 준다 -> AFE 를 통과시킨다
        for x, _ in loader:
            yield (afe(x.to(DEV), target_T=T),)

    sites = measure_ranges(model, feed(), device=DEV, max_batches=max_batches)
    print(f'=== {tag} ===')
    print_range_report(sites, guard_bits=guard_bits)
    if save:
        to_json(sites, f'runs/{tag}/ranges.json', guard_bits=guard_bits)
    return sites

print('bitwidths(tag) 준비됨')

# 확정 모델로 재기 (전체 test 세트, 몇 분)
# sites = bitwidths('xl_g12')

# 빠른 확인만 (2배치)
# sites = bitwidths('xl_g12', max_batches=2, save=False)'''),

md("""## 9. 잘못된 런 지우기

`resume=True`라 같은 태그로 다시 돌리면 **이어서** 학습한다. 설정을 바꿨으면 반드시
지우고 시작해야 한다."""),
code("""import shutil
for tag in []:                      # 예: ['af_lse078_g6']
    p = f'runs/{tag}'
    shutil.rmtree(p, ignore_errors=True); print('삭제:', p)"""),
]


def main() -> None:
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
