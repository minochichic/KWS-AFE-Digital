"""Generate notebooks/workbench.ipynb -- the working notebook for the RTX box.

Written as a generator rather than by hand so the cells stay reviewable in git
(a .ipynb diff is unreadable) and so the preflight guard can never drift from
what the code actually requires.

The one rule the notebook enforces: cell 1 sets up EVERYTHING (pull, staleness
check, imports, DATA_ROOT), and every later cell starts from a checkpoint. No
cell may depend on a variable another cell happened to leave behind -- that is
what cost two 100-epoch runs and two NameErrors.
"""
from __future__ import annotations

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "workbench.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {},
            "source": "\n".join(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": "\n".join(lines)}


CELLS = [
md("# BinaryMatchboxNet KWS — 작업 노트북",
   "",
   "**규칙 하나**: 셀 1을 돌린 뒤에는 **어느 셀이든 단독으로** 실행할 수 있다.",
   "셀 사이에 변수를 넘기지 않는다 — 분석 셀은 전부 체크포인트에서 시작한다.",
   "",
   "이 규칙이 없어서 겪은 일: 커널 stale로 100 에폭 ×2 낭비, `set_seed` 미정의,",
   "`t` 미정의. 전부 셀 간 암묵적 의존 때문이었다.",
   "",
   "> **커널을 재시작했으면 반드시 셀 1부터.** `git pull`은 프로젝트 모듈을",
   "> import 하기 *전에* 일어나야 하고, 이미 import 된 모듈은 pull 해도 안 바뀐다."),

md("## 1. 프리플라이트 — 커널 재시작 후 항상 여기부터"),
code(
 "import os, sys, subprocess, pathlib",
 "os.chdir(os.path.expanduser('~/KWS-AFE-Digital')); sys.path.insert(0, os.getcwd())",
 "",
 "print(subprocess.run(['git','pull'], capture_output=True, text=True).stdout.strip())",
 "print(subprocess.run(['git','log','--oneline','-1'], capture_output=True,",
 "                     text=True).stdout.strip(), '\\n')",
 "",
 "# 파일이 아니라 '로드된 모듈'을 검사한다. 파일만 보면 stale 커널을 못 잡는다.",
 "import torch, numpy as np",
 "from train.config import load_config, AFEConfig",
 "from data.speech_commands import build_dataloaders",
 "from data.afe import AFEFrontend",
 "from models.binary_matchboxnet import BinaryMatchboxNet",
 "from train.train import Trainer, set_seed",
 "import data.afe as _A",
 "",
 "NEED = [('spice 경로 analog/ 이전', ('train/config.py', 'analog/AFE/artifacts'),",
 "         lambda: AFEConfig().spice_matrix_path.startswith('analog/')),",
 "        (\"normalize='xmix'\", ('data/afe.py', 'def _xmix'),",
 "         lambda: hasattr(_A.AFEFrontend, '_xmix')),",
 "        ('effective_alpha()', ('data/afe.py', 'def effective_alpha'),",
 "         lambda: hasattr(_A.AFEFrontend, 'effective_alpha')),",
 "        ('비교기 k개', ('train/config.py', 'comparators_per_channel'),",
 "         lambda: 'comparators_per_channel' in AFEConfig.__dataclass_fields__)]",
 "bad_disk, bad_mem = [], []",
 "for name, (f, needle), check in NEED:",
 "    on_disk = needle in pathlib.Path(f).read_text()",
 "    in_mem = bool(check())",
 "    print(f\"{'✅' if in_mem else '❌'} {name:<28} 파일 {'O' if on_disk else 'X'}\"",
 "          f\"  모듈 {'O' if in_mem else 'X'}\")",
 "    (bad_disk if not on_disk else bad_mem if not in_mem else []).append(name)",
 "if bad_disk:",
 "    raise RuntimeError(f'파일에 없음 → git pull 실패: {bad_disk}')",
 "if bad_mem:",
 "    raise RuntimeError(f'파일엔 있는데 모듈엔 없음 → **커널 재시작**: {bad_mem}')",
 "",
 "DATA_ROOT = os.path.expanduser('~/datasets/speech_commands_v2')",
 "DEV = 'cuda' if torch.cuda.is_available() else 'cpu'",
 "SR = 16000",
 "assert os.path.isdir(DATA_ROOT), DATA_ROOT",
 "print(f'\\n✅ 준비 완료   {DEV}   {DATA_ROOT}')"),

md("## 2. 확정 설정",
   "",
   "여기 한 곳만 고치면 아래 전부가 따라간다. 확정 근거는 `proposal/`.",
   "",
   "| | 값 | 근거 |",
   "|---|---|---|",
   "| 정규화 | `xmix`, floor 0.02 | 저항 분압으로 구현 가능, minmax 대비 −0.35pp |",
   "| 비교기 | 채널당 1개 | 2개면 +2.5pp (업그레이드 경로) |",
   "| 필터뱅크 | `spice` 16채널 | 동료 v15 보드와 16/16 일치 |",
   "| 대역 | 50–8000 Hz | 125–5000은 −1.6pp |"),
code(
 "BASE = {'afe.filterbank_source': 'spice',",
 "        'afe.compression': 'sqrt',",
 "        'afe.normalize': 'xmix',",
 "        'afe.xmax_floor_frac': 0.02,",
 "        'afe.comparators_per_channel': 1,",
 "        'model.in_channels': 16}",
 "",
 "def make_cfg(tag, **over):",
 "    \"\"\"확정 설정 + 이번 실험만의 override.\"\"\"",
 "    cfg = load_config('configs/base.yaml', {'tag': tag, **BASE, **over})",
 "    cfg.data.root = DATA_ROOT",
 "    return cfg",
 "",
 "print(make_cfg('probe').afe)"),

md("## 3. 학습",
   "",
   "`d > 0`과 α 범위를 **학습 시작 전에** 검사한다. `d = 0`이면 floor가 반영되지",
   "않은 것이고, 그대로 두면 100 에폭을 버린 뒤에야 알게 된다."),
code(
 "def train(tag, **over):",
 "    cfg = make_cfg(tag, **over)",
 "    set_seed(cfg.train.seed)",
 "    afe = AFEFrontend(cfg.afe); model = BinaryMatchboxNet(cfg.model)",
 "    tr, va, te = build_dataloaders(cfg.data, cfg.train.batch_size, SR,",
 "                                   seed=cfg.train.seed)",
 "    w = next(iter(tr))[0]",
 "    afe.init_fixed_scale(w)      # δ 먼저",
 "    afe.init_thresholds(w)       # 그 다음 α",
 "",
 "    k = cfg.afe.comparators_per_channel",
 "    d, a = float(afe.xmax_floor), afe.effective_alpha()",
 "    assert d > 0, f'd=0 — floor_frac({cfg.afe.xmax_floor_frac})이 반영 안 됨'",
 "    n_par = sum(p.numel() for p in model.parameters())",
 "    print(f'[{tag}] d={d:.5f}  입력 {cfg.afe.n_channels*k}행  '",
 "          f'파라미터 {n_par:,}  α init {a.min():.3f}~{a.max():.3f}')",
 "",
 "    t = Trainer(cfg, model, afe=afe)",
 "    t.fit(tr, va, resume=True)",
 "    return report(tag)",
 "",
 "print('train(tag, **override) 준비됨')"),

md("## 4. 평가 — 체크포인트에서만 시작한다",
   "",
   "학습 셀을 안 돌렸어도, 커널을 재시작했어도 동작한다."),
code(
 "def load_run(tag, **over):",
 "    \"\"\"저장된 config로 afe/model을 복원한다. 이전 셀에 의존하지 않는다.\"\"\"",
 "    cfg = make_cfg(tag, **over)",
 "    afe = AFEFrontend(cfg.afe).to(DEV).eval()",
 "    model = BinaryMatchboxNet(cfg.model).to(DEV).eval()",
 "    ck = torch.load(f'runs/{tag}/best.pt', map_location=DEV, weights_only=True)",
 "    model.load_state_dict(ck['model']); afe.load_state_dict(ck['afe'])",
 "    return cfg, afe, model, ck",
 "",
 "@torch.no_grad()",
 "def accuracy(afe, model, loader, T, shift_ms=0.0):",
 "    \"\"\"shift_ms > 0 = 소리가 늦게 들어옴. 감싸 들어온 부분은 무음으로 채운다.\"\"\"",
 "    k = int(round(shift_ms * SR / 1000)); ok = n = 0",
 "    for x, y in loader:",
 "        x, y = x.to(DEV), y.to(DEV)",
 "        if k:",
 "            x = torch.roll(x, k, dims=-1)",
 "            if k > 0: x[..., :k] = 0.0",
 "            else:     x[..., k:] = 0.0",
 "        ok += (model(afe(x, target_T=T)).argmax(1) == y).sum().item()",
 "        n += y.numel()",
 "    return ok / n",
 "",
 "def report(tag, **over):",
 "    cfg, afe, model, ck = load_run(tag, **over)",
 "    _, _, te = build_dataloaders(cfg.data, cfg.train.batch_size, SR,",
 "                                 seed=cfg.train.seed)",
 "    a = afe.effective_alpha()",
 "    dead = int(((a >= 0.99) | (a <= 0.01)).sum())",
 "    acc = accuracy(afe, model, te, cfg.model.T)",
 "    print(f'>>> {tag}  val {ck[\"best_acc\"]:.4f}   test {acc:.4f}')",
 "    print(f'    d={float(afe.xmax_floor):.5f}  α {a.min():.3f}~{a.max():.3f}  '",
 "          f'죽은 비교기 {dead}/{a.numel()}')",
 "    return acc",
 "",
 "print('load_run / accuracy / report 준비됨')"),

md("## 5. 시간 이동 곡선 — 슬라이딩 설계의 검증",
   "",
   "FPGA는 100 ms마다 판정하는 슬라이딩 창을 쓴다. 단어가 창 안에서 **±365 ms**",
   "움직이므로, 그 범위에서 정확도가 유지돼야 한다.",
   "",
   "증강 없이 측정한 결과: ±100 ms는 −2pp지만 **±300 ms에서 −11~17pp**로 무너진다."),
code(
 "def shift_curve(tag, shifts=(-400,-300,-200,-100,0,100,200,300,400), **over):",
 "    cfg, afe, model, _ = load_run(tag, **over)",
 "    _, _, te = build_dataloaders(cfg.data, cfg.train.batch_size, SR,",
 "                                 seed=cfg.train.seed)",
 "    base = accuracy(afe, model, te, cfg.model.T, 0.0)",
 "    print(f'{tag}\\n{\"이동(ms)\":>9}{\"test acc\":>10}{\"기준 대비\":>11}')",
 "    out = {}",
 "    for s in shifts:",
 "        a = accuracy(afe, model, te, cfg.model.T, s)",
 "        out[s] = a",
 "        print(f'{s:>9}{a:>10.4f}{(a-base)*100:>+10.1f}pp')",
 "    return out",
 "",
 "print('shift_curve(tag) 준비됨')"),

md("## 6. 학습 방법 탐색 — time-shift 증강",
   "",
   "슬라이딩이 요구하는 ±365 ms를 학습에 넣으면 곡선이 평평해질 것으로 기대한다.",
   "MatchboxNet 논문은 ±5 ms를 썼는데 우리 배치에는 턱없이 작다.",
   "",
   "**보는 것**: 중심(0 ms) 정확도가 얼마나 떨어지는가 대 ±300 ms가 얼마나 오르는가.",
   "슬라이딩에서 실제로 중요한 건 **곡선 아래 넓이**이지 峰 높이가 아니다."),
code(
 "# 각 런은 100 에폭. 하나씩 돌리고 결과를 보고 다음을 정하는 편이 낫다.",
 "for ms in (150, 300):",
 "    train(f'af_shift{ms}', **{'data.aug_time_shift_ms': float(ms)})",
 "",
 "# 끝나면 곡선 비교",
 "# for tag in ('af_k1_ref', 'af_shift150', 'af_shift300'):",
 "#     shift_curve(tag); print()"),

md("## 7. 하드웨어 내보내기 — α → 저항비",
   "",
   "**반드시 `effective_alpha()`로 읽는다.** `xmix`는 α를 forward에서",
   "straight-through로 클램프하므로 파라미터 원값은 [0,1] 밖으로 떠다닌다."),
code(
 "def alpha_table(tag='af_k1_ref', RTOT=1e6, **over):",
 "    cfg, afe, _, _ = load_run(tag, **over)",
 "    a = afe.effective_alpha()",
 "    fc = [166,295,447,631,832,1072,1349,1660,",
 "          2042,2455,2951,3467,4169,4898,5754,6761]",
 "    k = cfg.afe.comparators_per_channel",
 "    print(f'{tag}   δ={float(afe.xmax_floor):.5f}   Ra+Rb={RTOT/1e3:.0f} kΩ\\n')",
 "    print(f\"{'ch':>2}{'f_c[Hz]':>9}{'i':>3}{'α':>8}{'Rb[kΩ]':>9}{'Ra[kΩ]':>9}{'상태':>7}\")",
 "    for c in range(cfg.afe.n_channels):",
 "        for i in range(k):",
 "            x = float(a[c*k + i]); rb = RTOT*x",
 "            st = '죽음' if x >= 0.99 or x <= 0.01 else '정상'",
 "            print(f'{c:>2}{fc[c]:>9}{i:>3}{x:>8.4f}{rb/1e3:>9.1f}'",
 "                  f'{(RTOT-rb)/1e3:>9.1f}{st:>7}')",
 "    ok = bool(a.min() >= 0 and a.max() <= 1)",
 "    print(f\"\\n범위 {a.min():.3f}~{a.max():.3f}  \"",
 "          f\"{'제작 가능 ✅' if ok else '제작 불가 ⚠️'}\")",
 "",
 "alpha_table()"),

md("## 8. 잘못된 런 지우기",
   "",
   "`resume=True`라 같은 태그로 다시 돌리면 **이어서** 학습한다. 설정을 바꿨으면",
   "반드시 지우고 시작해야 한다."),
code(
 "import shutil",
 "for tag in []:                      # 예: ['af_shift150']",
 "    p = f'runs/{tag}'",
 "    shutil.rmtree(p, ignore_errors=True); print('삭제:', p)"),
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
