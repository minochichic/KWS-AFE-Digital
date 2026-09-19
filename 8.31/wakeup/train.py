"""학습 진입점.

  python -m wakeup.train --target on --epochs 30 --tag on_8ch          (8.31/ 에서)
  python -m wakeup.train --target on --channels 16 --l1 0.05 --tag on_16ch

순서가 중요하다.
  1. 스케일/문턱/형판을 데이터에서 초기화
  2. START 조건과 tau 를 탐색      <- 이걸 건너뛰면 정렬이 어긋나 무너진다
  3. 형판과 문턱을 경사하강으로 학습, k 는 주기적으로 다시 탐색
  4. 회로 상수로 내보내기
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from .config import Config
from .model import WakeupModel
from .search import search_timing, offset_scores
from .model import detect_start, gather_states


# --------------------------------------------------------------------- 평가
@torch.no_grad()
def evaluate(model: WakeupModel, loader, device: str, limit: int = 0) -> Dict[str, float]:
    model.eval()
    tp = fp = n_pos = n_neg = n_found_pos = 0
    for i, (w, y) in enumerate(loader):
        if limit and i * loader.batch_size >= limit:
            break
        w, y = w.to(device), y.to(device)
        out = model.hard(w)
        wake, found = out["wake"], out["found"]
        p, q = y > 0.5, y <= 0.5
        tp += wake[p].sum().item(); n_pos += p.sum().item()
        fp += wake[q].sum().item(); n_neg += q.sum().item()
        n_found_pos += found[p].sum().item()
    return {
        "tpr": tp / max(n_pos, 1),
        "fpr": fp / max(n_neg, 1),
        "start_recall": n_found_pos / max(n_pos, 1),
        "n_pos": n_pos, "n_neg": n_neg,
    }


def _gather_init_batch(loader, n: int, device: str):
    """초기화와 탐색에 쓸 한 덩어리. 양성이 충분히 들어가야 한다."""
    ws, ys, got = [], [], 0
    for w, y in loader:
        ws.append(w); ys.append(y); got += w.shape[0]
        if got >= n:
            break
    return torch.cat(ws)[:n].to(device), torch.cat(ys)[:n].to(device)


# --------------------------------------------------------------------- 진단
@torch.no_grad()
def diagnose(model: WakeupModel, wave: torch.Tensor, y: torch.Tensor) -> None:
    """왜 안 되는지 재서 보여 준다. 30에폭 돌려 놓고 추측하지 않기 위한 것."""
    cfg = model.cfg
    model.eval()
    env = model.frontend.envelopes(wave)
    x = model.features(wave)
    pos = y > 0.5
    C, T = x.shape[1], x.shape[2]

    print("\n" + "=" * 62)
    print("  진단")
    print("=" * 62)

    print(f"\n[1] 정규화 포락선 분포 (문턱이 여기 어디쯤 있어야 한다)")
    q = torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99])
    v = torch.quantile(env.flatten().float(), q.to(env.device))
    print("    분위수 " + "  ".join(f"{int(a*100)}%:{b:.3f}" for a, b in zip(q, v)))
    th = model.frontend.threshold
    print(f"    문턱 theta  최소 {th.min():.3f}  중앙 {th.median():.3f}  최대 {th.max():.3f}")

    print(f"\n[2] 채널 켜짐률 — 너무 높으면 형판이 무의미해진다")
    on = x.mean(dim=(0, 2))
    print("    " + "  ".join(f"ch{i}:{v*100:.0f}%" for i, v in enumerate(on)))
    print(f"    전체 {x.mean()*100:.1f}%,  프레임당 켜진 채널 {x.sum(1).mean():.2f}/{C}")

    print(f"\n[3] START 가 어디서 뜨는가 (양성만)")
    ks = sorted({2, max(2, C // 4), max(3, C // 2), max(4, 3 * C // 4)})
    for mf in (1, 2, 3):
        for k in ks:
            st, fo = detect_start(x, k, mf)
            sp = st[pos & fo].float()
            if sp.numel() < 2:
                continue
            print(f"    k={k:2d} x{mf}프레임: 검출 {fo[pos].float().mean()*100:5.1f}%  "
                  f"중앙 {sp.median():5.1f}  10%~90% [{sp.quantile(.1):.0f}, "
                  f"{sp.quantile(.9):.0f}]  산포 σ {sp.std():.1f}")
    print("    * 중앙이 0 근처이고 산포가 작으면 잡음에 걸려 즉시 뜨는 것이다.")
    print("      그러면 tau 가 상대 시각이 아니라 절대 시각이 된다.")

    print(f"\n[4] START 이후 오프셋별 분리도 (AUC, 1.0 이 완전 분리)")
    st, fo = detect_start(x, cfg.start.k, cfg.start.min_frames)
    keep = fo | (~pos)
    auc, _ = offset_scores(x[keep], st[keep], y[keep], min(T - 1, 60))
    top = torch.topk(auc, 8)
    print("    상위: " + "  ".join(f"d={int(i)}:{float(v):.3f}"
                                  for v, i in zip(*top)))
    print(f"    최대 {auc.max():.3f}  중앙 {auc.median():.3f}")
    print("    * 최대가 0.6 아래면 이 특징으로는 어느 시각을 봐도 못 가른다.")

    print(f"\n[5] 현재 tau 에서의 일치 개수 분포")
    xs = gather_states(x[keep], st[keep], model.tau)
    cnt = model.head(xs)["count"]
    yk = y[keep]
    for s_ in range(cnt.shape[1]):
        cp, cn = cnt[yk > 0.5, s_], cnt[yk <= 0.5, s_]
        print(f"    s{s_+1} (tau={int(model.tau[s_])}): 양성 {cp.mean():5.2f}±{cp.std():.2f}"
              f"   음성 {cn.mean():5.2f}±{cn.std():.2f}"
              f"   차이 {abs(cp.mean()-cn.mean()):.2f}")
    print("    * 차이가 1 미만이면 형판이 양성과 음성을 구분하지 못한다.")
    print("=" * 62 + "\n")


# ----------------------------------------------------------------- 스윕
@torch.no_grad()
def sweep_frontend(cfg: Config, wave: torch.Tensor, y: torch.Tensor,
                   on_rates=(0.05, 0.10, 0.15, 0.20, 0.30),
                   min_recall: float = 0.95) -> None:
    """켜짐률을 바꿔가며 도달 가능한 분리도를 잰다. 학습 없이 답이 나온다.

    켜짐률마다 START 조건 (k, min_frames) 도 함께 훑는다. 둘은 독립이 아니다 --
    특징이 성기어지면 START 가 잘 안 뜨고, 그러면 그 발화는 판정 자체가
    불가능해져 TPR 상한이 된다.

    재는 것은 단일 오프셋의 최대 AUC 가 아니라 **간격을 둔 네 시각의 평균**이다.
    우리에게 필요한 것은 상태 4개이지 제일 좋은 한 순간이 아니다.

    START 가 안 뜬 클립은 양쪽 모두 AUC 에서 뺀다. 음성만 남기면 "START 가
    떴는가"를 재게 되어 오프셋 0 이 늘 이겨 버린다(실측: 최적 d 가 0).
    """
    from .model import WakeupModel
    from .search import pick_offsets
    m = WakeupModel(copy.deepcopy(cfg)).to(wave.device)
    m.frontend.init_fixed_scale(wave)
    env = m.frontend.envelopes(wave)
    rows = _sweep_rates(cfg, env, y, on_rates, min_recall)
    _print_sweep(cfg, rows)


@torch.no_grad()
def _sweep_rates(cfg: Config, env: torch.Tensor, y: torch.Tensor,
                 on_rates, min_recall: float) -> list:
    """정규화된 포락선 위에서 켜짐률과 START 조건을 훑는다.

    포락선은 대상 단어와 무관하므로 한 번만 계산해 여러 단어에 돌려 쓴다.
    """
    from .search import pick_offsets
    from .ste import step_ste
    pos = y > 0.5
    C = env.shape[1]
    flat = env.transpose(0, 1).reshape(C, -1)
    out = []
    for r in on_rates:
        th = torch.quantile(flat, 1.0 - r, dim=1).view(1, -1, 1)
        x = (env >= th).float()
        rate = x.mean().item()
        row = None
        for mf in (1, 2, 3, 4):
            for k in range(1, min(C, 9) + 1):
                st, fo = detect_start(x, k, mf)
                rec = fo[pos].float().mean().item()
                if rec < min_recall:
                    continue
                keep = fo
                if keep.sum() < 32 or (y[keep] > 0.5).sum() < 8:
                    continue
                auc, _ = offset_scores(x[keep], st[keep], y[keep],
                                       min(x.shape[2] - 1, 60))
                sel = pick_offsets(auc, cfg.head.n_states, min_gap=3, min_tau=1)
                ma = float(auc[sel].mean())
                if row is None or ma > row[0]:
                    free = 1.0 - fo[~pos].float().mean().item()
                    row = (ma, float(auc.max()), k, mf, rec, free, sel, rate, r)
        out.append(row)
    return out


def _print_sweep(cfg: Config, rows: list) -> None:
    C = cfg.frontend.n_channels
    print("\n" + "=" * 78)
    print("  프론트엔드 스윕 — 어느 켜짐률에서 네 시각이 갈리는가")
    print("=" * 78)
    print(f"  {'켜짐률':>6} {'START조건':>11} {'검출률':>7} {'음성무시':>8} "
          f"{'평균AUC':>8} {'최고AUC':>8} {'선택된 tau':>20}")
    best = None
    for row in rows:
        if row is None:
            print(f"  {'—':>6} {'—':>11}   검출률 미달로 후보 없음")
            continue
        ma, mx, k, mf, rec, free, sel, rate, r = row
        print(f"  {rate*100:5.1f}% {f'k{k} x{mf}프레임':>11} {rec*100:6.1f}% "
              f"{free*100:7.1f}% {ma:8.3f} {mx:8.3f} {str(sel):>20}")
        if best is None or ma > best[0]:
            best = (ma, r, k, mf, rec, sel)
    print("-" * 78)
    if best:
        print(f"  최고: 켜짐률 {best[1]*100:.0f}%, START k={best[2]} x{best[3]}프레임, "
              f"검출률 {best[4]*100:.1f}%")
        print(f"        tau {best[5]}, 네 시각 평균 AUC {best[0]:.3f}")
        print(f"\n  python -m wakeup.train --target {cfg.train.target_word} "
              f"--channels {C} --on-rate {best[1]} --epochs 30 "
              f"--tag {cfg.train.target_word}_{C}ch_r{int(best[1]*100)}")
    print("\n  * 평균 AUC 0.7 이상이면 쓸 만하다. 0.6 아래면 채널 수·대역·대상 단어를 다시 본다.")
    print("  * 검출률은 TPR 상한이다. START 를 놓친 발화는 되살릴 수 없다.")
    print("  * 음성무시 = START 가 안 뜬 음성 비율. 회로에서 공짜 정답이므로 높을수록 좋다.")
    print("=" * 78 + "\n")


# ------------------------------------------------------------- 단어 비교
# 시간 구조가 뚜렷한 쪽이 이 방식(네 시각의 스냅숏 비교)에 유리하다.
# marvin/sheila 는 Speech Commands 에 웨이크워드 용도로 들어간 다음절 단어다.
# 실제 제품의 웨이크워드가 길고 음소가 특이한 것도 같은 이유다.
WORD_CANDIDATES = ("marvin", "sheila", "backward", "forward", "follow",
                   "visual", "learn", "stop", "six", "seven", "happy",
                   "house", "on", "go")


@torch.no_grad()
def sweep_words(cfg: Config, words: Sequence[str], *, n_clips: int = 12000,
                on_rates=(0.15, 0.20, 0.30), min_recall: float = 0.95) -> None:
    """여러 대상 단어를 같은 클립 묶음으로 비교한다.

    포락선은 단어와 무관하므로 한 번만 계산하고 라벨만 바꿔 단다. 단어마다
    파일을 다시 읽지 않아 빠르고, 같은 데이터를 쓰므로 비교가 공정하다.
    """
    from . import data as D
    from .model import WakeupModel

    dev = cfg.train.device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    t0 = time.time()
    wave, labels = D.raw_batch(cfg.data_root, "train", n_clips,
                               seed=cfg.train.seed)
    print(f"클립 {len(labels)}개 읽음, {time.time()-t0:.0f}s")

    m = WakeupModel(copy.deepcopy(cfg)).to(dev)
    env_parts = []
    for i in range(0, wave.shape[0], 1024):
        w = wave[i:i + 1024].to(dev)
        if i == 0:
            m.frontend.init_fixed_scale(w)
        env_parts.append(m.frontend.envelopes(w).cpu())
    env = torch.cat(env_parts)
    del wave, env_parts
    print(f"포락선 계산 완료 {tuple(env.shape)}, {time.time()-t0:.0f}s\n")

    print("=" * 84)
    print(f"  대상 단어 비교 — {cfg.frontend.n_channels}채널, "
          f"상태 {cfg.head.n_states}개, 클립 {len(labels)}개")
    print("=" * 84)
    print(f"  {'단어':>9} {'양성':>6} {'켜짐률':>6} {'START':>11} {'검출률':>7} "
          f"{'음성무시':>8} {'평균AUC':>8} {'tau 범위':>10} {'선택된 tau':>20}")
    results = []
    for w in words:
        y = torch.tensor([1.0 if l == w else 0.0 for l in labels])
        n_pos = int(y.sum())
        if n_pos < 40:
            print(f"  {w:>9} {n_pos:>6}   양성이 너무 적어 건너뜀")
            continue
        rows = [r for r in _sweep_rates(cfg, env, y, on_rates, min_recall)
                if r is not None]
        if not rows:
            print(f"  {w:>9} {n_pos:>6}   어느 켜짐률에서도 검출률 미달")
            continue
        b = max(rows, key=lambda r: r[0])
        ma, mx, k, mf, rec, free, sel, rate, r = b
        span = sel[-1] - sel[0]
        print(f"  {w:>9} {n_pos:>6} {rate*100:5.1f}% "
              f"{f'k{k} x{mf}':>11} {rec*100:6.1f}% {free*100:7.1f}% "
              f"{ma:8.3f} {span:9d}f {str(sel):>20}")
        results.append((ma, w, r, k, mf, rec, sel, span))

    print("-" * 84)
    if results:
        results.sort(reverse=True)
        print("  좋은 순:")
        for ma, w, r, k, mf, rec, sel, span in results[:5]:
            print(f"    {w:>9}  평균 AUC {ma:.3f}  검출률 {rec*100:.1f}%  "
                  f"tau 범위 {span}프레임 ({span*10} ms)")
        ma, w, r, k, mf, rec, sel, span = results[0]
        print(f"\n  python -m wakeup.train --target {w} "
              f"--channels {cfg.frontend.n_channels} --on-rate {r} "
              f"--epochs 30 --tag {w}_{cfg.frontend.n_channels}ch")
    print("\n  * tau 범위가 좁으면 네 상태가 사실상 같은 구간을 보는 것이다.")
    print("    START 에서 멀어질수록 정렬이 흐트러진다는 뜻이기도 하다.")
    print("=" * 84 + "\n")


# ------------------------------------------------------- 구조(상태 수) 스윕
@torch.no_grad()
def sweep_structure(cfg: Config, *, n_clips: int = 12000,
                    states=(2, 3, 4, 6), gaps=(3, 6, 10),
                    min_recall: float = 0.95) -> None:
    """상태 수와 시각 간격을 훑는다. 이건 곧 보드 크기다.

    상태 하나가 비교기 1개 + 플립플롭 1개 + 저항 한 벌이다. 네 시각이 모두
    START 직후에 몰린다면 상태들이 사실상 같은 구간을 보는 것이므로, 더 적은
    상태로 같은 성능이 나올 수 있다.

    평균 AUC 만 보면 안 된다. WAKE 는 AND 라 가장 약한 상태가 결과를 정하므로
    **최소 AUC** 가 더 정직한 예측이다.
    """
    from . import data as D
    from .model import WakeupModel
    from .search import pick_offsets
    from .ste import step_ste

    dev = cfg.train.device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    t0 = time.time()
    wave, labels = D.raw_batch(cfg.data_root, "train", n_clips,
                               seed=cfg.train.seed)
    m = WakeupModel(copy.deepcopy(cfg)).to(dev)
    parts = []
    for i in range(0, wave.shape[0], 1024):
        w = wave[i:i + 1024].to(dev)
        if i == 0:
            m.frontend.init_fixed_scale(w)
        parts.append(m.frontend.envelopes(w).cpu())
    env = torch.cat(parts)
    del wave, parts
    y = torch.tensor([1.0 if l == cfg.train.target_word else 0.0 for l in labels])
    print(f"클립 {len(labels)}개, 양성 {int(y.sum())}개, {time.time()-t0:.0f}s\n")

    # 켜짐률과 START 는 cfg 에 정해진 것으로 고정하고 구조만 본다
    C = env.shape[1]
    flat = env.transpose(0, 1).reshape(C, -1)
    r = cfg.frontend.init_on_rate or 0.15
    th = torch.quantile(flat, 1.0 - r, dim=1).view(1, -1, 1)
    x = (env >= th).float()

    best_start = None
    for mf in (1, 2, 3, 4):
        for k in range(1, min(C, 9) + 1):
            st, fo = detect_start(x, k, mf)
            rec = fo[y > 0.5].float().mean().item()
            if rec >= min_recall and (best_start is None or rec > best_start[2]):
                best_start = (k, mf, rec, st, fo)
    if best_start is None:
        print("검출률을 맞추는 START 조건이 없다."); return
    k, mf, rec, st, fo = best_start
    auc, _ = offset_scores(x[fo], st[fo], y[fo], min(x.shape[2] - 1, 60))

    print("=" * 76)
    print(f"  구조 스윕 — '{cfg.train.target_word}', {C}채널, 켜짐률 {r*100:.0f}%, "
          f"START k{k} x{mf} (검출률 {rec*100:.1f}%)")
    print("=" * 76)
    print(f"  {'상태':>4} {'간격':>4} {'평균AUC':>8} {'최소AUC':>8} {'범위':>6} "
          f"{'비교기':>6} {'F/F':>4} {'선택된 tau':>26}")
    for ns in states:
        for g in gaps:
            sel = pick_offsets(auc, ns, min_gap=g, min_tau=1)
            if len(sel) < ns:
                continue
            a = auc[sel]
            print(f"  {ns:>4} {g:>4} {float(a.mean()):8.3f} {float(a.min()):8.3f} "
                  f"{sel[-1]-sel[0]:5d}f {ns:>6} {ns:>4} {str(sel):>26}")
    print("-" * 76)
    print("  * WAKE 는 AND 다. 가장 약한 상태가 결과를 정하므로 최소 AUC 를 보라.")
    print("  * 상태 하나 = 비교기 1 + 플립플롭 1 + 저항 한 벌. 이게 보드 크기다.")
    print("=" * 76 + "\n")


# --------------------------------------------------------------------- 학습
def train(cfg: Config, *, init_n: int = 4096, log_every: int = 50,
          diag: bool = False, allow_infeasible: bool = False,
          sweep: bool = False) -> Dict:
    from . import data as D

    torch.manual_seed(cfg.train.seed)
    dev = cfg.train.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("! CUDA 가 없다. CPU 로 돌린다 (느리다).")
        dev = "cpu"

    ld = D.loaders(cfg.data_root, cfg.train.target_word, cfg.train.batch_size,
                   seed=cfg.train.seed, num_workers=cfg.train.num_workers)
    model = WakeupModel(cfg).to(dev)

    # 1) 데이터에서 초기화
    t0 = time.time()
    w0, y0 = _gather_init_batch(ld["train"], init_n, dev)
    print(f"초기화 배치 {w0.shape[0]}개 (양성 {int(y0.sum())}개), "
          f"{time.time()-t0:.1f}s")
    if sweep:
        sweep_frontend(cfg, w0, y0)
        return {"sweep_only": True}

    model.init_from_data(w0, y0)

    # 2) 타이밍 탐색 — 미분되지 않는 값들
    print("타이밍 탐색:")
    r = search_timing(model, w0, y0, min_frames_cands=(1, 2, 3), verbose=True)
    print(f"  -> start_k {r['k']} x{r['min_frames']}프레임, tau {r['tau']}, "
          f"timeout {cfg.head.timeout}")

    if diag:
        diagnose(model, w0, y0)

    # 탐색이 실행 가능한 점을 못 찾았다 = 어떤 tau 로도 FPR 상한을 못 지킨다.
    # 그대로 두면 학습이 "전부 통과"로 무너진다(실측: TPR 0.978 / FPR 0.981).
    # 30에폭을 태우기 전에 여기서 멈춘다.
    if not r.get("feasible", True) and not allow_infeasible:
        print(f"\n! 타이밍 탐색이 FPR {cfg.head.k_max_fpr:.0%} 이하인 점을 못 찾았다 "
              f"(최고 점수 {r['score']:.3f}, 음수 = 실행 불가).")
        print("  이대로 학습하면 '전부 통과'로 무너진다. --diagnose 로 원인을 먼저 보라.")
        print("  그래도 돌리려면 --allow-infeasible.")
        if not diag:
            diagnose(model, w0, y0)
        return {"infeasible": True, "search": r}

    if diag:
        print("--diagnose: 학습은 건너뛴다.")
        return {"diagnose_only": True, "search": r}

    # 3) 경사하강
    pw = cfg.train.pos_weight or max(1.0, float((y0 <= 0.5).sum() / max(int(y0.sum()), 1)))
    print(f"양성 가중 {pw:.2f}")
    opt = torch.optim.Adam([
        {"params": model.head.parameters(), "lr": cfg.train.lr},
        {"params": model.frontend.parameters(), "lr": cfg.train.lr_threshold},
    ], weight_decay=cfg.train.weight_decay)

    best, hist, step = -1.0, [], 0
    out_dir = Path(cfg.out_dir) / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(cfg.train.epochs):
        model.train()
        agg = {"loss": 0.0, "bce": 0.0, "n": 0}
        for w, y in ld["train"]:
            w, y = w.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            o = model.loss(w, y, pos_weight=pw)
            opt.zero_grad(); o["loss"].backward(); opt.step()
            agg["loss"] += o["loss"].item(); agg["bce"] += o["bce"].item()
            agg["n"] += 1
            step += 1
            if cfg.head.refit_k_every and step % cfg.head.refit_k_every == 0:
                model.refit_k(w0, y0)

        model.refit_k(w0, y0)                      # 에폭 끝에 한 번 더
        va = evaluate(model, ld["val"], dev)
        # FPR 상한을 지킨 것 중 TPR 최대
        sel = (1.0 + va["tpr"]) if va["fpr"] <= cfg.head.k_max_fpr else -va["fpr"]
        e = model.export()
        line = (f"ep {ep+1:3d}  loss {agg['loss']/max(agg['n'],1):.4f}  "
                f"val TPR {va['tpr']:.3f} FPR {va['fpr']:.3f}  "
                f"START {va['start_recall']:.3f}  "
                f"저항 {e['n_resistors']:3d}  k {e['k'].tolist()}")
        hist.append({"epoch": ep + 1, **va, "n_resistors": e["n_resistors"]})
        if sel > best:
            best = sel
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict()},
                       out_dir / "best.pt")
            line += "  *"
        print(line)

    # 4) 시험 분할
    ck = torch.load(out_dir / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    te = evaluate(model, ld["test"], dev)
    print(f"\n시험 분할: TPR {te['tpr']:.3f}  FPR {te['fpr']:.3f}  "
          f"START 재현율 {te['start_recall']:.3f}  "
          f"(양성 {te['n_pos']}, 음성 {te['n_neg']})")

    from .export import write_report
    rep = write_report(model, out_dir, test=te, history=hist)
    print(f"\n회로 상수 -> {out_dir}")
    return {"test": te, "history": hist, "export": rep}


# ----------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="고정 회로 웨이크업 학습")
    p.add_argument("--target", default="on", help="대상 단어")
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--states", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--lr-threshold", type=float, default=1e-3)
    p.add_argument("--l1", type=float, default=0.02, help="X 를 늘리는 압력")
    p.add_argument("--min-channels", type=int, default=3)
    p.add_argument("--max-fpr", type=float, default=0.05)
    p.add_argument("--ste-clip", type=float, default=0.03)
    p.add_argument("--filterbank", default="mel", choices=("mel", "spice"))
    p.add_argument("--spice-matrix", default="")
    p.add_argument("--root", default="~/datasets/speech_commands_v2")
    p.add_argument("--out", default="runs")
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sweep-structure", action="store_true",
                   help="상태 수와 시각 간격을 훑는다 (= 보드 크기)")
    p.add_argument("--sweep-words", default="",
                   help="대상 단어 비교. 쉼표로 나열하거나 auto")
    p.add_argument("--sweep-frontend", action="store_true",
                   help="켜짐률·압축을 바꿔가며 분리도만 재고 끝낸다")
    p.add_argument("--on-rate", type=float, default=0.0,
                   help="문턱 초기화의 목표 켜짐률 (0 = 채널 평균)")
    p.add_argument("--compression", default="log", choices=("log", "sqrt"))
    p.add_argument("--diagnose", action="store_true",
                   help="초기화와 타이밍 탐색까지만 하고 진단을 찍는다")
    p.add_argument("--allow-infeasible", action="store_true",
                   help="탐색이 실행 가능한 점을 못 찾아도 학습을 강행한다")
    a = p.parse_args(argv)

    cfg = Config()
    cfg.frontend.n_channels = a.channels
    cfg.frontend.ste_clip = a.ste_clip
    cfg.frontend.init_on_rate = a.on_rate
    cfg.frontend.compression = a.compression
    cfg.frontend.filterbank = a.filterbank
    cfg.frontend.spice_matrix_path = a.spice_matrix
    cfg.head.n_states = a.states
    # tau 는 어차피 탐색으로 다시 정해진다. 여기서는 검증만 통과하면 된다.
    cfg.head.tau = [8 + 8 * i for i in range(a.states)]
    cfg.head.timeout = min(63, cfg.head.tau[-1] + 12)
    cfg.head.l1_gate = a.l1
    cfg.head.min_channels = a.min_channels
    cfg.head.k_max_fpr = a.max_fpr
    cfg.train.target_word = a.target
    cfg.train.epochs = a.epochs
    cfg.train.batch_size = a.batch_size
    cfg.train.lr = a.lr
    cfg.train.lr_threshold = a.lr_threshold
    cfg.train.seed = a.seed
    cfg.train.device = a.device
    cfg.train.num_workers = a.workers
    cfg.data_root = a.root
    cfg.out_dir = a.out
    cfg.tag = a.tag or f"{a.target}_{a.channels}ch_s{a.states}"
    cfg.validate()

    if a.sweep_structure:
        sweep_structure(cfg)
        return
    if a.sweep_words:
        words = (WORD_CANDIDATES if a.sweep_words == "auto"
                 else tuple(w.strip() for w in a.sweep_words.split(",")))
        sweep_words(cfg, words)
        return
    print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2)[:600] + " ...\n")
    train(cfg, diag=a.diagnose, allow_infeasible=a.allow_infeasible,
          sweep=a.sweep_frontend)


if __name__ == "__main__":
    main()
