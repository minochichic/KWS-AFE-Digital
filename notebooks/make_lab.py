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
from data.afe import AFEFrontend
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

# 확정 설정. 실험 셀들은 이걸 복사해 필요한 키만 바꾼다.
CONFIRMED = {'afe.filterbank_source': 'spice',
             'afe.compression': 'sqrt',
             'afe.normalize': 'xmix',
             'afe.xmax_floor_frac': 0.02,
             'afe.comparators_per_channel': 2}

# soft-max 계열. k=1 로 되돌려 xmix k=1 (0.7778) 과 직접 비교한다.
LSE  = {**CONFIRMED, 'afe.normalize': 'xlse', 'afe.comparators_per_channel': 1}
BEST = {**LSE, 'afe.lse_temp_frac': 0.78}

print(f"\\n모듈 정상.  DATA_ROOT={DATA_ROOT}  존재={os.path.isdir(DATA_ROOT)}")
print('torch', torch.__version__, '| cuda',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(CPU only)')'''),

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
| `d > 0` | `floor_frac` 미반영으로 100 에폭 헛돌기 |
| α 범위 | 저항비로 만들 수 없는 값 |"""),
code('''# --- 공용 러너 -----------------------------------------------------------
import json, yaml

_NEEDS_SCALE = ('fixed', 'agc', 'xmax', 'xmix', 'xlse')
_NEEDS_FLOOR = ('xmax', 'xmix', 'xlse')

def run(tag, over=None, epochs=None):
    """학습 1회. over 는 dotted override dict."""
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
           'aug_gain_db': list(getattr(cfg.data, 'aug_gain_db', []) or []),
           'aug_time_shift_ms': float(getattr(cfg.data, 'aug_time_shift_ms', 0.0)),
           'alpha': [round(float(x), 4) for x in alpha],
           'dead': int(((alpha >= 0.99) | (alpha <= 0.01)).sum())}
    pathlib.Path(run_dir, 'result.json').write_text(
        json.dumps(rec, ensure_ascii=False, indent=2))
    return rec

print('run(tag, over) / save_result 준비됨')'''),

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
    model.load_state_dict(ck['model']); afe.load_state_dict(ck['afe'])
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
code('''_TE = None                  # test loader 는 런 사이 동일 -> 한 번만 만든다

def backfill(tag):
    global _TE
    d = pathlib.Path('runs', tag)
    if not (d / 'best.pt').exists():
        print(f'  {tag}: best.pt 없음'); return None
    cfg = load_config(str(d / 'config.yaml'))      # 그 런이 실제로 쓴 설정
    cfg.data.root = DATA_ROOT
    if _TE is None:
        _TE = test_loader(cfg)
    t = Trainer(cfg, BinaryMatchboxNet(cfg.model), afe=AFEFrontend(cfg.afe))
    ck = torch.load(d / 'best.pt', map_location=t.device, weights_only=True)
    t.model.load_state_dict(ck['model']); t.afe.load_state_dict(ck['afe'])
    rec = save_result(d, tag, cfg, float(ck['best_acc']),
                      float(t.evaluate(_TE)['acc']), t.afe.effective_alpha())
    print(f"  {tag}: test {rec['test']:.4f} 복원")
    return rec


def results(backfill_missing=True):
    if backfill_missing:
        for p in sorted(pathlib.Path('runs').glob('*/config.yaml')):
            if not (p.parent / 'result.json').exists():
                backfill(p.parent.name)
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
    return rows

results()'''),

md("""## 6a. 창 위치 곡선 — 슬라이딩의 **진짜** 비용

FPGA는 100 ms마다 판정하는 슬라이딩 창을 쓴다. 한 단어가 창 8개에 걸리고 창마다
위치가 다르므로 그 변화를 견뎌야 한다.

단어를 **자르지 않고** 창 안에서만 옮기고 나머지는 그 클립 자신의 노이즈 플로어로
채운다. 위치는 정규화 p (0 = 왼쪽 밀착, 1 = 오른쪽 밀착).

| `noise` 낙폭 | 판단 |
|---|---|
| ≤ 5pp | 슬라이딩이 사실상 공짜. 최대 확신도 선택만 붙이면 끝 |
| 5–15pp | 최대 확신도 선택 필요 |
| ≥ 15pp | 오정렬이 진짜 문제 — **긴 캔버스** 증강을 만들 가치가 있음 |

`zero` 열이 크게 낮으면 인공 무음에서 상대 임계가 퇴화한 것(EXPERIMENTS.md §4-4)."""),
code("""from experiments.window_offset import offset_curve, print_offset_curve

def offset_report(tag, steps=9, fills=('noise','zero'), **over):
    cfg, afe, model, _ = load_run(tag, **over)
    res = offset_curve(afe, model, test_loader(cfg), cfg.model.T,
                       steps=steps, fills=fills, device=DEV)
    print_offset_curve(res, tag)
    return res

TAG = 'af_lse078'          # ← 위 결과 표에서 고른다
offset_report(TAG)"""),

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

md("""## 7. 학습 — ⚠️ 전부 주석

돌릴 줄 **하나만** 풀고, 끝나면 다시 주석 처리한다.

| # | 실험 | 왜 | 상태 |
|---|---|---|---|
| 1 | frac 0.78 시드 재현 | 스파이크가 진짜인지 | ✅ `af_lse078_s2` 완료 — 0.8351, **스파이크 미재현** |
| 2 | frac 0.78 + k=2 | 하드 max의 +2.5pp 레버가 soft-max와 겹치나 | ⬜ |
| 3 | **게인 증강** | §6d가 민감하다고 하면 | ⬜ |
| 4 | `spice_gain_restore` | 소프트 max는 채널간 절대 스케일을 섞는다 | ⬜ |

**3이 왜 살아났나**: 게인 증강은 예전에 −9pp로 기각됐지만 그건 `normalize=fixed`
얘기였다(고정 임계 아래서 게인은 이미지를 지운다). `xlse`에서 클립 게인을 바꾸는 것은
**`lse_temp_frac`을 바꾸는 것과 같다** — 교란이 아니라 강건해지고 싶은 그 물리량이다.

**4가 왜 필요한가**: `spice_gain_restore=False`의 근거는 "채널별 threshold가 1.8 dB
편차를 흡수한다"였다. **진짜 max에서는 맞지만 소프트 max에서는 틀리다** — LSE 분모는
16채널을 한 숫자로 섞으므로 한 채널의 오차가 **모두의** 분모를 민다."""),
code("""# 2. k=2 와 soft-max 가 겹치는지
# run('af_lse078_k2', {**BEST, 'afe.comparators_per_channel': 2})

# 3. 게인 증강 — 폭은 §6c 가 계산해준 값을 쓴다
# run('af_lse078_g6',  {**BEST, 'data.aug_gain_db': [-6.0, 6.0]})
# run('af_lse078_g12', {**BEST, 'data.aug_gain_db': [-12.0, 12.0]})

# 4. 채널간 스케일 복원 (소프트 max 에서만 의미가 생긴다)
# run('af_lse078_gr', {**BEST, 'afe.spice_gain_restore': True})

# 끝나면
# results(); frac_sweep('af_lse078_g12')"""),

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
