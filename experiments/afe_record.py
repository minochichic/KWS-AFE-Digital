"""Per-ARB records of the board's captures (wake plan step 4), shared by afe_web and afe_report.

A *record* is everything the board sent while one ARB was selected in the viewer: the
captures, the board's quiet (IDLE) firing, and -- derived on demand, so a later capture can
still improve the alignment -- which slot of the ARB each capture belongs to, the threshold
table and the answers. A *session* is one viewer run: a folder with one sub-folder per record
plus session-wide tables (records.csv, confusion.csv/png).

Slot alignment. A sequence ARB (experiments.afe_seq) has several clips at fixed places in one
AWG period P. The board stamps every capture with its frame tick (`end`, 10 ms, LSE timer); the
software reference has, per slot, where in the period its capture ends (`end_mod`). The AWG
start is unknown, so the offset o is the one that puts the most board captures within TOL
frames of a simulated one: (end - o) mod P ~ end_mod. A capture with no simulated one near
it is "unmatched" (an extra trigger) and is left out of the tables.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from experiments.afe_clip import bits_of

TOL = 20                 # frames: a board capture this close to a simulated one is that slot
MIN_ALIGN = 3            # captures needed before slots are assigned in a sequence
SETTLE = 2.0             # s after a record starts: may still be the previous waveform


def load_ref(path: Path, names: List[str]) -> Optional[dict]:
    """ref.json (afe_clip single clip, or afe_seq sequence) -> what a record needs."""
    ref = json.loads(path.read_text())
    if ref.get("kind") == "sequence":
        sim = [{"slot": c["slot"], "label": c["label"], "window": c["window"], "pred": names[c["pred"]],
                "end_mod": c["end_mod"]} for c in ref["sim_caps"]]
        segs = [{"slot": s["slot"], "label": s["label"], "src": s["src"], "clip_pred": names[s["clip_pred"]]}
                for s in ref["segments"]]
        period = ref["period_frames"]
    else:
        caps = ref["gains"]["0"]["captures"]
        if not caps:
            return None
        lab = names[ref["label"]]
        sim = [{"slot": 0, "label": lab, "window": caps[0]["window"], "pred": names[caps[0]["pred"]],
                "end_mod": None}]
        segs = [{"slot": 0, "label": lab, "src": Path(ref["wav"]).name,
                 "clip_pred": names[ref["gains"]["0"]["clip_pred"]]}]
        period = None
    labels = sorted({s["label"] for s in segs})
    single = len(sim) == 1
    if not sim:
        return None
    return {"ref": path.parent.name, "kind": "single" if single else "sequence", "period": period,
            "sim": sim, "segments": segs, "label": labels[0] if len(labels) == 1 else "mixed",
            "sim_pred": sim[0]["pred"] if single else
            f"{sum(s['pred'] == s['label'] for s in sim)}/{len(sim)} 맞음",
            "slots": len(segs), "default_target": 10 if single else len(sim)}


def align(ref: dict, ends: List[int]) -> dict:
    """Board capture ticks -> index into ref['sim'] per capture (None = unmatched / not yet)."""
    sim = ref["sim"]
    if len(sim) == 1:
        return {"assign": [0] * len(ends), "offset": None, "matched": len(ends), "state": "single"}
    if len(ends) < MIN_ALIGN:
        return {"assign": [None] * len(ends), "offset": None, "matched": 0, "state": "aligning"}
    P = float(ref["period"])                                  # frames; the colleague's files are not whole
    e = np.asarray(ends, dtype=np.float64)
    t = np.asarray([s["end_mod"] for s in sim], dtype=np.float64)
    o = np.arange(0.0, P, 1.0)
    d = (e[None, :, None] - o[:, None, None] - t[None, None, :]) % P
    d = np.minimum(d, P - d)                                  # [offset, capture, sim]
    best = d.min(2)
    hit = best <= TOL
    score = hit.sum(1) * 10_000 - np.where(hit, best, 0).sum(1)
    k = int(score.argmax())
    j = d[k].argmin(1)
    assign = [int(j[i]) if hit[k, i] else None for i in range(len(ends))]
    return {"assign": assign, "offset": round(float(o[k]), 1), "matched": int(hit[k].sum()), "state": "aligned"}


class Record:
    def __init__(self, idx: int, ref: Optional[dict], target: Optional[int]) -> None:
        self.idx, self.ref = idx, ref
        self.target = target if target else (ref["default_target"] if ref else None)
        self.caps: List[dict] = []
        self.quiet_n = 0
        self.quiet = [0] * 16
        self.started = time.time()
        self.started_s = time.strftime("%H:%M:%S")
        self.saved: Optional[str] = None
        self.dirty = False
        self.done = False

    @property
    def name(self) -> str:
        return self.ref["ref"] if self.ref else "(none)"

    def add(self, cap: dict) -> None:
        cap["settling"] = time.time() - self.started < SETTLE
        self.caps.append(cap)
        self.dirty = True

    def add_quiet(self, d: List[int]) -> None:
        self.quiet_n += d[0]
        self.quiet = [a + b for a, b in zip(self.quiet, d[1:])]
        if d[0]:
            self.dirty = True

    def view(self, names: List[str]) -> dict:
        """Everything derived: per-capture slot/label/sim, tables. Cheap enough to redo each poll."""
        ref = self.ref
        al = align(ref, [c["end"] for c in self.caps]) if ref else {"assign": [None] * len(self.caps),
                                                                    "offset": None, "matched": 0, "state": "none"}
        caps, board, simon, n = [], [0] * 16, [0] * 16, 0
        slots = {s["slot"]: {**s, "sim_pred": None, "n": 0, "answered": 0, "fpga": Counter()}
                 for s in (ref["segments"] if ref else [])}
        for s in (ref["sim"] if ref else []):
            slots[s["slot"]]["sim_pred"] = s["pred"]
        res = {"n": 0, "answered": 0, "correct": 0, "fpga": Counter(), "unmatched": 0, "settling": 0, "paused": 0}
        for c, j in zip(self.caps, al["assign"]):
            v = dict(c)
            v["playing"] = self.name if ref else None
            s = ref["sim"][j] if (ref and j is not None) else None
            why = ("paused" if c["paused"] else "settle" if c["settling"] else
                   None if s else ("aligning" if al["state"] == "aligning" else "unmatched" if ref else "no ARB"))
            if s:
                v.update(slot=s["slot"], label=s["label"], sim=s["window"], sim_pred=s["pred"])
            v["why"] = why
            v["ref"] = self.name if (ref and why is None) else None      # scored
            caps.append(v)
            if why in ("paused", "settle"):
                res[why if why == "paused" else "settling"] += 1
            elif why == "unmatched":
                res["unmatched"] += 1
            if why is not None:
                continue
            n += 1
            for k, (b, w) in enumerate(zip(bits_of(c["clip"]).sum(0), bits_of(s["window"]).sum(0))):
                board[k] += int(b)
                simon[k] += int(w)
            sl = slots[s["slot"]]
            sl["n"] += 1
            res["n"] += 1
            res["fpga"][c["fpga"]] += 1
            if c["fpga"] in names:
                sl["answered"] += 1
                sl["fpga"][c["fpga"]] += 1
                res["answered"] += 1
                res["correct"] += int(c["fpga"] == s["label"])
        return {"idx": self.idx, "ref": self.name, "kind": ref["kind"] if ref else None,
                "label": ref["label"] if ref else None, "sim_pred": ref["sim_pred"] if ref else None,
                "started": self.started_s, "target": self.target, "done": self.done, "saved": self.saved,
                "align": {k: al[k] for k in ("offset", "matched", "state")}, "caps": caps,
                "tune": {"n": n, "board": board, "sim": simon, "quiet_n": self.quiet_n, "quiet": self.quiet,
                         "since": self.started_s, "refs": 1 if ref else 0},
                "slots": [{**v, "fpga": dict(v["fpga"])} for _, v in sorted(slots.items())],
                "result": {**res, "fpga": dict(res["fpga"])}}

    def scored(self, names: List[str]) -> int:
        return self.view(names)["result"]["n"]


def confusion(views: List[dict], names: List[str]) -> dict:
    """label x FPGA answer over the scored captures of every record (timeouts in their own column)."""
    cols = sorted(names) + ["timeout", "bad"]                  # alphabetical, like the paper figure
    m = {l: Counter() for l in names}
    for v in views:
        for c in v["caps"]:
            if c.get("ref") and c.get("label") in m:
                m[c["label"]][c["fpga"] if c["fpga"] in cols else "bad"] += 1
    rows = [l for l in sorted(names) if m[l]]
    return {"rows": rows, "cols": cols, "m": {l: dict(m[l]) for l in rows}}


def write_record(folder: Path, view: dict, fc: Optional[List[int]], names: List[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    # what experiments.score_live / thr_calib read: ref + label only on scored captures
    with (folder / "captures.jsonl").open("w", encoding="utf-8") as f:
        for c in view["caps"]:
            f.write(json.dumps({k: v for k, v in c.items() if k != "sim"}) + "\n")
    with (folder / "captures.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["n", "time", "arb", "slot", "label", "scored", "why_not", "sim_pred", "fpga", "best_pt",
                    "prob", "fpga_correct", "end_tick", "bg_q"] + [f"on_ch{k}" for k in range(16)] + ["frames_hex"])
        for c in view["caps"]:
            on = bits_of(c["clip"]).sum(0)
            w.writerow([c["n"], c["time"], view["ref"], c.get("slot", ""), c.get("label", ""), int(bool(c["ref"])),
                        c["why"] or "", c.get("sim_pred", ""), c["fpga"], c["best_pt"], c["prob"],
                        "" if not c.get("label") else int(c["fpga"] == c["label"]), c["end"], c["bg_q"]]
                       + [int(x) for x in on] + [" ".join(f"{x:04x}" for x in c["clip"])])
    t = view["tune"]
    n, qn = t["n"], t["quiet_n"]
    with (folder / "threshold.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ch", "f_c_hz", "target", "board", "board_minus_target", "quiet_rate", "captures", "quiet_s"])
        for k in range(16):
            b = t["board"][k] / (100 * n) if n else None
            s = t["sim"][k] / (100 * n) if n else None
            w.writerow([k, fc[k] if fc else "", "" if s is None else round(s, 4), "" if b is None else round(b, 4),
                        "" if b is None else round(b - s, 4), round(t["quiet"][k] / qn, 4) if qn else "",
                        n, round(qn / 100, 1)])
    with (folder / "slots.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["slot", "src", "label", "sim_clip_pred", "sim_capture_pred", "board_captures",
                    "fpga_answered", "fpga_correct", "fpga_answers"])
        for s in view["slots"]:
            w.writerow([s["slot"], s["src"], s["label"], s["clip_pred"], s["sim_pred"] or "(no capture)", s["n"],
                        s["answered"], s["fpga"].get(s["label"], 0),
                        " ".join(f"{k}:{v}" for k, v in sorted(s["fpga"].items(), key=lambda kv: -kv[1]))])
    (folder / "record.json").write_text(json.dumps({k: v for k, v in view.items() if k != "caps"},
                                                   ensure_ascii=False, indent=1))
    sheet(view, folder / "sheet.png", fc)


def sheet(view: dict, path: Path, fc: Optional[List[int]], max_caps: int = 40) -> None:
    """Contact sheet: firing rates (board vs target) and up to max_caps captures with answers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    caps = view["caps"][-max_caps:]
    cols = 5
    rows = (len(caps) + cols - 1) // cols
    fig = plt.figure(figsize=(16, 2.6 + 1.55 * rows))
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[2.2] + [1.3] * rows)
    ax = fig.add_subplot(gs[0, :])
    t = view["tune"]
    n = t["n"]
    x = np.arange(16)
    if n:
        ax.bar(x - 0.2, np.array(t["sim"]) / (100 * n), 0.4, color="#9aa1ab", label="목표 (소프트웨어)")
        ax.bar(x + 0.2, np.array(t["board"]) / (100 * n), 0.4, color="#1f5fbf", label="보드")
    if t["quiet_n"]:
        ax.plot(x, np.array(t["quiet"]) / t["quiet_n"], "_", color="#c2410c", ms=18, mew=3, label="무음 발화")
    ax.set_xticks(x, [f"ch{k}\n{fc[k]}" if fc else f"ch{k}" for k in x], fontsize=8)
    ax.set_ylabel("발화율")
    ax.legend(fontsize=8, ncol=3)
    r = view["result"]
    ax.set_title(f"기록 {view['idx']:02d} · {view['ref']} ({view['label']}) · 시작 {view['started']} · "
                 f"채점 캡처 {r['n']} · FPGA 정답 {r['correct']}/{r['answered']} · 제외: 미정렬 {r['unmatched']}, "
                 f"일시중지 {r['paused']}, 전환직후 {r['settling']}", fontsize=11)
    for i, c in enumerate(caps):
        a = fig.add_subplot(gs[1 + i // cols, i % cols])
        a.imshow(bits_of(c["clip"]).T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1.4,
                 interpolation="nearest")
        a.set_xticks([])
        a.set_yticks([])
        good = c.get("label") and c["fpga"] == c["label"]
        tag = f"slot{c['slot']} " if c.get("slot") is not None and view["kind"] == "sequence" else ""
        a.set_title(f"#{c['n']} {tag}{c.get('label') or ''} → {c['fpga']}"
                    f"{(' ○' if good else ' ×') if c.get('label') else ''}"
                    f"{'' if c['ref'] else ' (' + (c['why'] or '') + ')'}",
                    fontsize=8, color="#1a7f37" if good else ("#c2410c" if c.get("label") else "#555"))
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_session(folder: Path, views: List[dict], names: List[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "records.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["record", "arb", "kind", "label", "sim", "started", "scored", "fpga_answered", "fpga_correct",
                    "unmatched", "paused", "settling", "target", "fpga_answers", "folder"])
        for v in views:
            r = v["result"]
            w.writerow([v["idx"], v["ref"], v["kind"], v["label"], v["sim_pred"], v["started"], r["n"],
                        r["answered"], r["correct"], r["unmatched"], r["paused"], r["settling"], v["target"],
                        " ".join(f"{k}:{c}" for k, c in sorted(r["fpga"].items(), key=lambda kv: -kv[1])),
                        v["saved"] or ""])
    from experiments.afe_report import report           # confusion matrices, per class, report.md
    report([Path(v["saved"]) for v in views if v.get("saved")], folder)
