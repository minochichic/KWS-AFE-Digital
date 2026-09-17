"""Figures for the interim report, in the plain style of an IEEE paper.

Everything is read from files -- runs/<tag>/history.json, the JSON written by
`experiments.fixed_accuracy --json-out`, and the streaming summaries -- never
from numbers typed in here. A figure whose values were copied by hand is a
figure nobody can regenerate after the next retrain.

Style: serif font, single-column width (3.5 in) or double (7.16 in), thin
black axes, no titles inside the plot (the caption carries that), grayscale-
safe markers. Output is both PDF (vector, for the document) and PNG at 600 dpi
(for Word, which embeds raster more reliably).

    python -m experiments.report_figures \
        --tag bd_base --tag bd_base_ft20_partial75 \
        --fixed out/report/bd_base_test.json \
        --fixed out/report/bd_base_ft20_partial75_test.json \
        --out out/report/fig
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL1 = 3.5    # IEEE single column, inches
COL2 = 7.16   # IEEE double column

SHORT = {"bd_base": "Baseline", "bd_base_ft20_partial75": "Partial-75"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=600)
    plt.close(fig)
    print(f"  {out / name}.pdf/.png")


def history(runs: Path, tag: str) -> List[Dict]:
    doc = json.loads((runs / tag / "history.json").read_text())
    return doc["history"] if isinstance(doc, dict) else doc


def fig_training(runs: Path, tags: List[str], out: Path) -> None:
    """Validation accuracy and training loss per epoch, one panel each."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.0))
    marks = ["-", "--", ":", "-."]
    # A fine-tune starts where its parent stopped. Plotting it from epoch 1
    # overlays it on the parent's first epochs and reads as a model that was
    # already at 82 % before training -- so shift it by the parent's length.
    offset = 0
    for i, tag in enumerate(tags):
        h = history(runs, tag)
        if i > 0 and "_ft" in tag:
            offset = len(history(runs, tags[0]))
        else:
            offset = 0
        ep = [r["epoch"] + offset for r in h]
        lab = SHORT.get(tag, tag)
        if "val_acc" in h[0]:
            a.plot(ep, [100 * r["val_acc"] for r in h], marks[i % 4],
                   color="k", label=lab)
        if "train_loss" in h[0]:
            b.plot(ep, [r["train_loss"] for r in h], marks[i % 4],
                   color="k", label=lab)
    a.set_xlabel("Epoch")
    a.set_ylabel("Validation accuracy (%)")
    b.set_xlabel("Epoch")
    b.set_ylabel("Training loss")
    for ax in (a, b):
        ax.grid(True, linewidth=0.3, color="0.8")
    for ax in (a, b):
        if len(tags) > 1 and "_ft" in tags[1]:
            ax.axvline(len(history(runs, tags[0])), color="0.5",
                       linewidth=0.5, linestyle=":")
    a.legend(loc="lower right")
    a.text(-0.18, 1.02, "(a)", transform=a.transAxes)
    b.text(-0.18, 1.02, "(b)", transform=b.transAxes)
    save(fig, out, "fig_training_curves")


def fig_confusion(doc: Dict, out: Path, name: str) -> None:
    """Row-normalised confusion matrix, percentages printed in each cell."""
    m = np.asarray(doc["confusion_fixed"], dtype=float)
    rows = m.sum(axis=1, keepdims=True)
    p = 100 * m / np.where(rows == 0, 1, rows)
    names = [n.strip("_") for n in doc["class_names"]]
    fig, ax = plt.subplots(figsize=(COL1, COL1))
    ax.imshow(p, cmap="Greys", vmin=0, vmax=100)
    for t in range(len(names)):
        for q in range(len(names)):
            if p[t, q] >= 0.5:
                ax.text(q, t, f"{p[t, q]:.0f}", ha="center", va="center",
                        fontsize=5, color="w" if p[t, q] > 55 else "k")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.tick_params(length=0)
    save(fig, out, name)


def fig_per_class(docs: List[Dict], out: Path) -> None:
    """Per-class recall of the integer path, grouped bars."""
    names = [n.strip("_") for n in docs[0]["class_names"]]
    x = np.arange(len(names))
    w = 0.8 / len(docs)
    fills = ["0.25", "0.75", "0.5"]
    fig, ax = plt.subplots(figsize=(COL2, 2.0))
    for i, d in enumerate(docs):
        m = np.asarray(d["confusion_fixed"], dtype=float)
        rec = 100 * np.diag(m) / np.maximum(m.sum(axis=1), 1)
        ax.bar(x + (i - (len(docs) - 1) / 2) * w, rec, w,
               color=fills[i % 3], edgecolor="k", linewidth=0.4,
               label=f"{SHORT.get(d['tag'], d['tag'])} "
                     f"({100 * d['fixed_acc']:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linewidth=0.3, color="0.8")
    ax.set_axisbelow(True)
    # above the axes: inside, it sat on top of the bars
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(docs))
    save(fig, out, "fig_per_class_recall")


def fig_channels(docs: List[Dict], out: Path) -> None:
    """Comparator firing probability per AFE channel (input statistics)."""
    fig, ax = plt.subplots(figsize=(COL1, 1.8))
    marks = ["o-", "s--", "^:"]
    for i, d in enumerate(docs):
        r = 100 * np.asarray(d["channel_fire_rate"])
        ax.plot(np.arange(len(r)), r, marks[i % 3], color="k", markersize=3,
                markerfacecolor="w", label=SHORT.get(d["tag"], d["tag"]))
    ax.set_xlabel("AFE channel (low to high frequency)")
    ax.set_ylabel("P(output = 1) (%)")
    ax.set_xticks(range(0, len(r), 2))
    ax.set_ylim(0, 40)
    ax.grid(True, linewidth=0.3, color="0.8")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(docs))
    save(fig, out, "fig_channel_firing")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--fixed", action="append", default=[],
                    help="fixed_accuracy --json-out files, same order as --tag")
    ap.add_argument("--out", default="out/report/fig")
    args = ap.parse_args()

    style()
    runs, out = Path(args.runs), Path(args.out)
    fig_training(runs, args.tag, out)
    docs = [json.loads(Path(p).read_text()) for p in args.fixed]
    for d in docs:
        fig_confusion(d, out, f"fig_confusion_{d['tag']}")
    if docs:
        fig_per_class(docs, out)
        fig_channels(docs, out)


if __name__ == "__main__":
    main()
