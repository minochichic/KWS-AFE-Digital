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
from typing import Dict, Optional

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
                   on_rates=(0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
                   compressions=("log",)) -> None:
    """켜짐률과 압축을 바꿔가며 분리도를 잰다. 학습 없이 답이 나온다.

    최대 AUC 가 이 특징으로 도달 가능한 상한이다. 그게 낮으면 tau 나 START 를
    아무리 잘 골라도 소용없다 -- 고쳐야 할 곳이 프론트엔드라는 뜻이다.

    압축은 기본으로 log 하나만 훑는다. 문턱을 분위수로 잡으면 단조 변환이
    이진 출력을 바꾸지 못하므로 log 와 sqrt 의 초기 AUC 가 정확히 같다.
    둘의 차이는 학습 중에만 나타난다(ste_clip 이 도는 스케일이 달라진다).
    """
    from .model import WakeupModel
    pos = y > 0.5
    print("\n" + "=" * 74)
    print("  프론트엔드 스윕 — 어떤 켜짐률에서 특징이 갈리는가")
    print("=" * 74)
    print(f"  {'압축':>5} {'목표':>5} {'실제켜짐':>8} {'최대AUC':>8} {'최적d':>6} "
          f"{'START중앙':>9} {'START σ':>8} {'검출률':>7}")
    best = None
    for comp in compressions:
        for r in on_rates:
            c = copy.deepcopy(cfg)
            c.frontend.compression = comp
            c.frontend.init_on_rate = r
            m = WakeupModel(c).to(wave.device)
            m.frontend.init_fixed_scale(wave)
            m.frontend.init_thresholds(wave, on_rate=r)
            x = m.features(wave)
            rate = x.mean().item()

            # START 는 켜짐률에 맞춰 채널의 절반 정도를 요구하는 지점으로 본다
            kk = max(2, int(round(c.frontend.n_channels * min(0.6, 2 * r))))
            st, fo = detect_start(x, kk, 2)
            sp = st[pos & fo].float()
            keep = fo | (~pos)
            auc, _ = offset_scores(x[keep], st[keep], y[keep],
                                   min(x.shape[2] - 1, 60))
            amax, dbest = float(auc.max()), int(auc.argmax())
            print(f"  {comp:>5} {r*100:4.0f}% {rate*100:7.1f}% {amax:8.3f} "
                  f"{dbest:6d} {sp.median():9.1f} {sp.std():8.1f} "
                  f"{fo[pos].float().mean()*100:6.1f}%")
            if best is None or amax > best[0]:
                best = (amax, comp, r, rate, dbest)
    print("-" * 74)
    print(f"  최고: {best[1]} 압축, 목표 켜짐률 {best[2]*100:.0f}% "
          f"(실제 {best[3]*100:.1f}%) -> AUC {best[0]:.3f} at d={best[4]}")
    print("  * AUC 0.7 이상이면 쓸 만하다. 0.6 아래면 채널 수·대역을 다시 봐야 한다.")
    print("  * 문턱을 분위수로 잡으므로 log/sqrt 는 초기 이진 출력이 동일하다.")
    print("=" * 74 + "\n")


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

    print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2)[:600] + " ...\n")
    train(cfg, diag=a.diagnose, allow_infeasible=a.allow_infeasible,
          sweep=a.sweep_frontend)


if __name__ == "__main__":
    main()
