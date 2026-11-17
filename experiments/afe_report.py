"""One report from one or more viewer sessions (out/afe/sessions/<stamp>/, experiments.afe_web).

Reads every record folder's captures.csv / slots.csv / threshold.csv and writes, into --out:

    report.md              per-class table, totals, where the files came from
    per_class.csv          records, scored captures, FPGA / best.pt / software AFE accuracy per class
    confusion_fpga.*       label x FPGA answer, every scored board capture        (csv, png)
    confusion_best_pt.*    label x best.pt on the same board 100 frames            (csv, png)
    confusion_sim.*        label x best.pt on the software AFE's capture, per clip (csv, png)
    confusion_all.png      the three side by side
    threshold.csv          the per-channel table pooled over all records (captures-weighted)

The matrices are drawn like the Cerutti et al. figure: 12 classes in alphabetical order, counts
in Blues. A timeout (no FPGA answer) is not a class; it is counted in per_class.csv and in the
title, not in the FPGA matrix.

    python -m experiments.afe_report out/afe/sessions/20261031_* --out out/afe/report_1031
    python -m experiments.afe_report out/afe/sessions/20261031_* --only 2610301846_merged_yes   # some ARBs
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def rows(path: Path) -> List[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_cm(cm: Dict[str, Counter], order: List[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label \\ predicted"] + order + ["total", "correct", "acc"])
        for l in order:
            n = sum(cm[l][c] for c in order)
            w.writerow([l] + [cm[l][c] for c in order] + [n, cm[l][l], round(cm[l][l] / n, 4) if n else ""])


def draw_cm(ax, cm: Dict[str, Counter], order: List[str], title: str) -> None:
    M = np.array([[cm[l][c] for c in order] for l in order], dtype=float)
    frac = M / np.maximum(M.sum(1, keepdims=True), 1)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if frac[i, j] > 0.6 else "#333")
    ax.set_xticks(range(len(order)), order, rotation=90)
    ax.set_yticks(range(len(order)), order)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    tot = M.sum()
    ax.set_title(f"{title}\n정확도 {np.trace(M) / tot:.1%} ({int(np.trace(M))}/{int(tot)})" if tot else title,
                 fontsize=11)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="session folders (or record folders)")
    ap.add_argument("--only", nargs="*", default=None, help="keep records of these ARBs only")
    ap.add_argument("--out", default=None, help="default out/afe/report_<now>")
    ap.add_argument("--order", choices=["alpha", "model"], default="alpha",
                    help="class order in the matrices: alpha (like the paper figure) or the model's")
    args = ap.parse_args()

    recs = []
    for s in map(Path, args.sessions):
        recs += [s] if (s / "captures.csv").exists() else sorted(p.parent for p in s.glob("*/captures.csv"))
    if args.only:
        recs = [r for r in recs if r.name.split("_", 1)[1] in args.only]
    if not recs:
        raise SystemExit("no record folders (…/<nn>_<arb>/captures.csv) found")
    out = Path(args.out or f"out/afe/report_{time.strftime('%Y%m%d_%H%M%S')}")
    print(report(recs, out, args.order))
    print(f"wrote {out}")


def report(recs: List[Path], out: Path, order_mode: str = "alpha") -> str:
    """Record folders -> the files listed in the module docstring, in `out`. Returns a one-line summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from data.speech_commands import KEYWORDS

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    names = KEYWORDS + ["silence", "unknown"]
    order = sorted(names) if order_mode == "alpha" else names
    out.mkdir(parents=True, exist_ok=True)
    cm = {k: defaultdict(Counter) for k in ("fpga", "best_pt", "sim")}
    cls: Dict[str, dict] = defaultdict(lambda: {"records": 0, "scored": 0, "fpga_ok": 0, "answered": 0,
                                               "best_ok": 0, "agree": 0, "sim_clip_ok": 0, "sim_cap_ok": 0,
                                               "clips": 0, "sim_missed": 0, "timeouts": 0})
    on = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])          # ch -> board frames, target frames, quiet on, quiet s
    fc: Dict[int, str] = {}
    seen_clips = set()                                      # (arb, slot): a clip counts once in the sim matrix
    for r in recs:
        arb = r.name.split("_", 1)[1]
        caps = [c for c in rows(r / "captures.csv") if c["scored"] == "1"]
        for l in {c["label"] for c in caps} | {s["label"] for s in rows(r / "slots.csv")}:
            cls[l]["records"] += 1
        for c in caps:
            k = cls[c["label"]]
            k["scored"] += 1
            k["best_ok"] += c["best_pt"] == c["label"]
            cm["best_pt"][c["label"]][c["best_pt"]] += 1
            if c["fpga"] in names:
                k["answered"] += 1
                k["fpga_ok"] += c["fpga"] == c["label"]
                k["agree"] += c["fpga"] == c["best_pt"]
                cm["fpga"][c["label"]][c["fpga"]] += 1
            else:
                k["timeouts"] += 1
        for s in rows(r / "slots.csv"):
            if (arb, s["slot"]) in seen_clips:
                continue
            seen_clips.add((arb, s["slot"]))
            k = cls[s["label"]]
            k["clips"] += 1
            k["sim_clip_ok"] += s["sim_clip_pred"] == s["label"]
            if s["sim_capture_pred"] in names:
                k["sim_cap_ok"] += s["sim_capture_pred"] == s["label"]
                cm["sim"][s["label"]][s["sim_capture_pred"]] += 1
            else:
                k["sim_missed"] += 1
        for t in rows(r / "threshold.csv"):
            n, qs, ch = float(t["captures"] or 0), float(t["quiet_s"] or 0), int(t["ch"])
            fc[ch] = t["f_c_hz"]
            if n and t["board"] != "":
                on[ch][0] += float(t["board"]) * n
                on[ch][1] += float(t["target"]) * n
            if qs and t["quiet_rate"] != "":
                on[ch][2] += float(t["quiet_rate"]) * qs
                on[ch][3] += qs

    titles = {"fpga": "보드 → FPGA", "best_pt": "보드 → best.pt (같은 100칸)", "sim": "소프트웨어 AFE → best.pt (클립당 1회)"}
    for k in cm:
        write_cm(cm[k], order, out / f"confusion_{k}.csv")
        fig, ax = plt.subplots(figsize=(7.4, 6.6))
        draw_cm(ax, cm[k], order, titles[k])
        ax.set_xlabel("예측")
        ax.set_ylabel("정답")
        fig.tight_layout()
        fig.savefig(out / f"confusion_{k}.png", dpi=130)
        plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    for ax, k in zip(axs, ("sim", "best_pt", "fpga")):
        draw_cm(ax, cm[k], order, titles[k])
    fig.tight_layout()
    fig.savefig(out / "confusion_all.png", dpi=110)
    plt.close(fig)

    corder = [l for l in order if l in cls]
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "—"
    with (out / "per_class.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class", "records", "clips", "scored_captures", "fpga_answered", "fpga_correct", "fpga_acc",
                    "best_pt_acc", "fpga_eq_best_pt", "sim_clip_acc", "sim_capture_acc", "sim_no_capture", "timeouts"])
        for l in corder:
            k = cls[l]
            w.writerow([l, k["records"], k["clips"], k["scored"], k["answered"], k["fpga_ok"],
                        round(k["fpga_ok"] / k["answered"], 4) if k["answered"] else "",
                        round(k["best_ok"] / k["scored"], 4) if k["scored"] else "",
                        round(k["agree"] / k["answered"], 4) if k["answered"] else "",
                        round(k["sim_clip_ok"] / k["clips"], 4) if k["clips"] else "",
                        round(k["sim_cap_ok"] / k["clips"], 4) if k["clips"] else "", k["sim_missed"], k["timeouts"]])
    tot_n = sum(cls[l]["scored"] for l in cls) or 1
    with (out / "threshold.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ch", "f_c_hz", "target", "board", "board_minus_target", "quiet_rate"])
        for ch in range(16):
            b, t, qo, qs = on[ch]
            w.writerow([ch, fc.get(ch, ""), round(t / tot_n, 4), round(b / tot_n, 4), round((b - t) / tot_n, 4),
                        round(qo / qs, 4) if qs else ""])

    tf = sum(cls[l]["fpga_ok"] for l in cls)
    ta = sum(cls[l]["answered"] for l in cls)
    to = sum(cls[l]["timeouts"] for l in cls)
    lines = [f"# AFE 보드 기록 리포트 ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"기록 {len(recs)}개, 채점된 캡처 {tot_n}개 (타임아웃 {to}), FPGA 캡처 정확도 **{pct(tf, ta)}** ({tf}/{ta}).", "",
             "| 클래스 | 기록 | 클립 | 채점 캡처 | FPGA 정답률 | best.pt 정답률 | FPGA = best.pt | 시뮬 (클립) | 시뮬 (캡처) | 타임아웃 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for l in corder:
        k = cls[l]
        lines.append(f"| {l} | {k['records']} | {k['clips']} | {k['scored']} | {pct(k['fpga_ok'], k['answered'])} | "
                     f"{pct(k['best_ok'], k['scored'])} | {pct(k['agree'], k['answered'])} | "
                     f"{pct(k['sim_clip_ok'], k['clips'])} | {pct(k['sim_cap_ok'], k['clips'])} | {k['timeouts']} |")
    lines += ["", "- 시뮬 (클립): 각 클립을 소프트웨어 AFE + best.pt 로. 시뮬 (캡처): 펌웨어 트리거 규칙이 자른 100칸으로.",
              "- 채점 캡처 = 고른 직후 2 s · 일시 중지 · 슬롯에 안 맞는 캡처를 뺀 것.", "",
              "![confusion](confusion_all.png)", "", "## 원본 기록", ""] + [f"- `{r}`" for r in recs]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"records {len(recs)}, scored captures {tot_n}, FPGA {tf}/{ta}, timeouts {to}"


if __name__ == "__main__":
    main()
