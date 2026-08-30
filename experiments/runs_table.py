"""모든 런을 한 표로. 스크롤백을 뒤지지 않기 위해.

학습이 끝나면 `runs/<tag>/` 에 남는 것:

  test.json     best.pt 의 test 정확도 (2026-08-30 부터)
  history.json  에폭별 loss/acc/val_acc/lr
  config.yaml   실제로 쓰인 설정 전체
  best.pt       가장 좋은 val 의 체크포인트

test 정확도는 원래 화면에만 찍고 끝났다. 런이 여러 개면 긴 학습 로그 뒤에서
찾을 수가 없어서 파일로도 남기게 했다. 그 전에 돈 런은 `--rescore` 로 다시 잰다.

    python -m experiments.runs_table
    python -m experiments.runs_table --filter bd_
    python -m experiments.runs_table --rescore        # test.json 없는 런 채우기
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# config 에서 이 표에 띄울 값. (라벨, 점 경로)
COLS = [
    ("thr_min", "afe.threshold_min"),
    ("ste_clip", "afe.ste_clip"),
    ("swing", "afe.spice_swing_path"),
    ("vos", "afe.comparator_vos"),
    ("seed", "train.seed"),
]


def dig(d, dotted):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def short(v):
    if v is None:
        return "-"
    if isinstance(v, str):
        return Path(v).name if "/" in v else (v or "-")
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def rescore(tag: str, runs: Path) -> None:
    """test.json 이 없는 런을 best.pt 로 다시 점수 낸다."""
    import torch
    from train.config import load_config
    from data.afe import AFEFrontend, load_afe_state
    from data.speech_commands import build_dataloaders
    from models.binary_matchboxnet import BinaryMatchboxNet

    run = runs / tag
    cfg = load_config(str(run / "config.yaml"))
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    afe = AFEFrontend(cfg.afe).eval()
    model = BinaryMatchboxNet(cfg.model).eval()
    model.load_state_dict(ck["model"])
    load_afe_state(afe, ck["afe"])
    te = build_dataloaders(cfg.data, cfg.train.batch_size, cfg.afe.sample_rate,
                           seed=cfg.train.seed)[2]
    ok = n = 0
    with torch.no_grad():
        for wav, lab in te:
            ok += (model(afe(wav, target_T=cfg.model.T)).argmax(1) == lab).sum().item()
            n += lab.numel()
    (run / "test.json").write_text(json.dumps(
        {"tag": tag, "best": {"acc": ok / n, "epoch": ck.get("epoch")},
         "rescored": True}, indent=2))
    print(f"  {tag}: {ok/n:.4f} ({n} 클립) -> test.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--filter", default="", help="태그에 이 문자열이 든 것만")
    ap.add_argument("--rescore", action="store_true",
                    help="test.json 이 없는 런을 best.pt 로 다시 점수 낸다")
    args = ap.parse_args()

    runs = Path(args.runs)
    if not runs.is_dir():
        raise SystemExit(f"{runs} 가 없다")
    tags = sorted(d.name for d in runs.iterdir()
                  if d.is_dir() and (d / "config.yaml").is_file()
                  and args.filter in d.name)
    if not tags:
        raise SystemExit(f"런이 없다 (filter={args.filter!r})")

    if args.rescore:
        todo = [t for t in tags if not (runs / t / "test.json").is_file()
                and (runs / t / "best.pt").is_file()]
        print(f"test.json 없는 런 {len(todo)}개 다시 점수:")
        for t in todo:
            rescore(t, runs)
        print()

    rows = []
    for t in tags:
        run = runs / t
        cfg = json.loads(json.dumps(_yaml(run / "config.yaml")))
        r = {"tag": t}
        tj = run / "test.json"
        if tj.is_file():
            j = json.loads(tj.read_text())
            b = j.get("best") or j.get("final_epoch") or {}
            r["test"] = b.get("acc")
            r["ep"] = b.get("epoch")
        hj = run / "history.json"
        if hj.is_file():
            h = json.loads(hj.read_text()).get("history", [])
            r["val"] = max((e.get("val_acc", -1) for e in h), default=None)
            r["eps"] = len(h)
        for lab, path in COLS:
            r[lab] = dig(cfg, path)
        rows.append(r)

    w = max(len(r["tag"]) for r in rows) + 2
    hdr = f"{'tag':<{w}}{'test':>8}{'val':>8}{'ep':>5}{'에폭':>6}"
    hdr += "".join(f"{lab:>10}" for lab, _ in COLS)
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -(x.get("test") or -1)):
        line = f"{r['tag']:<{w}}"
        line += f"{r['test']:>8.4f}" if r.get("test") is not None else f"{'-':>8}"
        line += f"{r['val']:>8.4f}" if r.get("val") is not None else f"{'-':>8}"
        line += f"{r.get('ep') or '-':>5}{r.get('eps') or '-':>6}"
        line += "".join(f"{short(r[lab]):>10}" for lab, _ in COLS)
        print(line)
    missing = [r["tag"] for r in rows if r.get("test") is None]
    if missing:
        print(f"\ntest 값 없음 ({len(missing)}개): {', '.join(missing)}")
        print("  --rescore 로 best.pt 에서 다시 잰다 (런당 몇 분).")


def _yaml(p: Path):
    import yaml
    with p.open() as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    main()
