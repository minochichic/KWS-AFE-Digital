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
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch

from .config import Config
from .model import WakeupModel
from .search import search_timing


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


# --------------------------------------------------------------------- 학습
def train(cfg: Config, *, init_n: int = 4096, log_every: int = 50) -> Dict:
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
    model.init_from_data(w0, y0)

    # 2) 타이밍 탐색 — 미분되지 않는 값들
    print("타이밍 탐색:")
    r = search_timing(model, w0, y0, verbose=True)
    print(f"  -> start_k {r['k']}, tau {r['tau']}, timeout {cfg.head.timeout}")

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
    a = p.parse_args(argv)

    cfg = Config()
    cfg.frontend.n_channels = a.channels
    cfg.frontend.ste_clip = a.ste_clip
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
    train(cfg)


if __name__ == "__main__":
    main()
