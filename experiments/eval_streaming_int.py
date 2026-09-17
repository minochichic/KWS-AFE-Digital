"""Integer-exact streaming reference: the always-on path as the RTL computes it.

``eval_streaming`` measures the float model with the Python ``ConsecutiveVote``.
The chip does something slightly different in form, and this file writes that
form down exactly so the chip can be checked against it case by case:

  per window   pooled[c] = sum over 64 frames of conv4's Q8.6 output (integer,
               export/golden.py's chain, experiments.fixed_accuracy.fixed_logits)
               keyword_idx    = argmax(pooled[0:10]), ties -> lower index
               keyword_margin = max(pooled[0:10]) - max(pooled[10:12])
               (rtl/kws_tail.v, "argmax" block)

  per stream   IntVote below mirrors rtl/kws_vote.v register by register:
               accept = margin >= MARGIN_INT, streak of REQUIRED equal keywords,
               cooldown and quiet-before-rearm.

Policy constants are READ from rtl/gen/<tag>/parameters.vh (KWS_STREAM_*), not
typed here -- the file the bitstream is built from is the only source.

Cases are built exactly as in ``eval_streaming`` (same seeds, same balanced
selection, same silence splicing), so for the same arguments the float numbers
reproduce that tool and the integer numbers can be compared line by line.

    # A1-A3: validation, integer vs float
    python -m experiments.eval_streaming_int --tag bd_base_ft20_partial75 \
        --split val --clips-per-class 256

    # A4-A5: test once, and export chip vectors (20 cases per class = 240)
    python -m experiments.eval_streaming_int --tag bd_base_ft20_partial75 \
        --split test --clips-per-class 256 \
        --export rtl/gen/bd_base_ft20_partial75/stream_selftest --export-per-class 20
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

N_KEYWORDS = 10
QUIET = (10, 11)
TARGET_START_FRAME = 100
TARGET_END_FRAME = 200


# ---- policy from the generated header ---------------------------------------- #
def read_stream_params(vh: Path) -> Dict[str, int]:
    txt = vh.read_text()
    out = {}
    for key in ("KWS_STREAM_HOP_FRAMES", "KWS_STREAM_REQUIRED", "KWS_STREAM_COOLDOWN_WINDOWS",
                "KWS_STREAM_MARGIN_INT", "KWS_CONV4_POOL_BITS"):
        m = re.search(rf"`define\s+{key}\s+(-?\d+)", txt)
        if not m:
            raise SystemExit(f"{vh}: {key} not found -- re-run export.emit with the streaming policy")
        out[key] = int(m.group(1))
    return out


# ---- kws_tail keyword/margin --------------------------------------------------- #
def keyword_and_margin(pooled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """[N, 12] int64 -> (keyword_idx [N], keyword_margin [N]); np.argmax keeps the
    first maximum, which is the RTL's strict '>' scan from index 0."""
    kw = pooled[:, :N_KEYWORDS]
    idx = kw.argmax(axis=1)
    margin = kw.max(axis=1) - pooled[:, list(QUIET)].max(axis=1)
    return idx.astype(np.int64), margin.astype(np.int64)


# ---- kws_vote, register for register ------------------------------------------ #
@dataclass
class IntVote:
    margin_int: int
    required: int
    cooldown_windows: int
    candidate: int = 0
    streak: int = 0
    cooldown: int = 0
    quiet_seen: bool = True
    armed: bool = True

    def step(self, keyword_idx: int, margin: int) -> Tuple[Optional[int], bool]:
        """One score_valid pulse with window_gap = 0. Returns (detection, accept)."""
        accept = margin >= self.margin_int
        cooldown_after = 0 if self.cooldown == 0 else self.cooldown - 1
        quiet_after = self.quiet_seen or not accept
        effective_armed = self.armed or (quiet_after and cooldown_after == 0)

        # next-state values; every assignment below reads the OLD register values
        n_candidate, n_streak = self.candidate, self.streak
        n_cooldown, n_quiet, n_armed = cooldown_after, self.quiet_seen, self.armed
        detection = None
        if not self.armed and not accept:
            n_quiet = True
        if not self.armed and quiet_after and cooldown_after == 0:
            n_armed = True

        if not accept or not effective_armed:
            n_candidate, n_streak = 0, 0
        elif self.streak != 0 and keyword_idx == self.candidate:
            if self.streak + 1 >= self.required:
                detection = keyword_idx
                n_candidate, n_streak = 0, 0
                n_armed, n_quiet, n_cooldown = False, False, self.cooldown_windows
            else:
                n_streak = self.streak + 1
        elif self.required == 1:
            detection = keyword_idx
            n_candidate, n_streak = 0, 0
            n_armed, n_quiet, n_cooldown = False, False, self.cooldown_windows
        else:
            n_candidate, n_streak = keyword_idx, 1

        self.candidate, self.streak = n_candidate, n_streak
        self.cooldown, self.quiet_seen, self.armed = n_cooldown, n_quiet, n_armed
        return detection, accept


# ---- chip vector format ---------------------------------------------------------- #
MARGIN_BITS = 22  # KWS_CONV4_POOL_BITS + 1, asserted in main


def crc16_windows(idx: Sequence[int], margin: Sequence[int]) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no xorout)
    over one 26-bit word per window, MSB first: {keyword_idx[3:0], margin[21:0]}.
    Bit-serial on purpose -- this is exactly what the RTL scorer implements."""
    crc = 0xFFFF
    for k, m in zip(idx, margin):
        word = ((int(k) & 0xF) << MARGIN_BITS) | (int(m) & ((1 << MARGIN_BITS) - 1))
        for b in range(4 + MARGIN_BITS - 1, -1, -1):
            fb = ((crc >> 15) ^ (word >> b)) & 1
            crc = (crc << 1) & 0xFFFF
            if fb:
                crc ^= 0x1021
    return crc


def expected_word(target: int, dets: List[Tuple[int, int]], crc: int) -> int:
    """40-bit expected result per case.

        [39:24] crc16 over the 21 windows
        [23:20] target label (for on-chip hit / wrong / quiet-false counters)
        [19:10] detection slot 1: {valid, window_id[4:0], idx[3:0]}
        [ 9: 0] detection slot 0: {valid, window_id[4:0], idx[3:0]}

    At most two detections fit in 21 windows with N=5 and a 10-window cooldown;
    main() asserts that."""
    slots = [0, 0]
    for s, (win, k) in enumerate(dets[:2]):
        slots[s] = (1 << 9) | ((win & 0x1F) << 4) | (k & 0xF)
    return (crc << 24) | ((target & 0xF) << 20) | (slots[1] << 10) | slots[0]


# ---- metrics ------------------------------------------------------------------ #
def window_overlaps_target(window_id: int, hop: int, native: int) -> bool:
    start = window_id * hop
    return min(start + native, TARGET_END_FRAME) - max(start, TARGET_START_FRAME) > 0


def score_case(target: int, dets: List[Tuple[int, int]], hop: int, native: int) -> Dict[str, int]:
    correct = [d for d in dets if window_overlaps_target(d[0], hop, native)
               and target < N_KEYWORDS and d[1] == target]
    outside = [d for d in dets if not window_overlaps_target(d[0], hop, native)]
    wrong = [d for d in dets if d not in correct and d not in outside]
    return dict(hit=int(target < N_KEYWORDS and bool(correct)),
                quiet_false=int(target >= N_KEYWORDS and bool(dets)),
                wrong=len(wrong), outside=len(outside),
                duplicate=max(0, len(correct) - 1),
                first_correct_window=(correct[0][0] if correct else -1))


def summarize(cases, hop, native) -> Dict[str, object]:
    kw = [c for c in cases if c["target"] < N_KEYWORDS]
    qt = [c for c in cases if c["target"] >= N_KEYWORDS]
    lat = [((c["first_correct_window"] * hop + native) - TARGET_START_FRAME) * 10
           for c in kw if c["hit"]]
    return {
        "keyword_cases": len(kw), "keyword_hits": sum(c["hit"] for c in kw),
        "keyword_recall": sum(c["hit"] for c in kw) / len(kw) if kw else None,
        "quiet_cases": len(qt), "quiet_false_streams": sum(c["quiet_false"] for c in qt),
        "wrong_detections": sum(c["wrong"] for c in cases),
        "outside_detections": sum(c["outside"] for c in cases),
        "duplicate_detections": sum(c["duplicate"] for c in cases),
        "median_latency_ms": float(np.median(lat)) if lat else None,
    }


# ---- main ------------------------------------------------------------------------ #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--clips-per-class", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260913, help="same default as eval_streaming")
    ap.add_argument("--margin-float", type=float, default=None,
                    help="float gate for the side-by-side float result; default from "
                         "configs/streaming_*.json if its checkpoint_tag matches")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="")
    ap.add_argument("--export", default="", help="write chip vectors to this directory")
    ap.add_argument("--export-per-class", type=int, default=20)
    args = ap.parse_args()

    import torch
    from data.afe import AFEFrontend, load_afe_state
    from data.speech_commands import KEYWORDS, SILENCE_INDEX, build_dataloaders
    from experiments.eval_streaming import collect_balanced_waveforms, splice_clips, _resolve_device
    from experiments.fixed_accuracy import fixed_logits
    from experiments.streaming_window import VoteConfig, WindowSpec, make_snapshots, vote_sequence
    from export.fuse import binary_accumulator
    from export.tailbuild import tail_plan
    from models.binary_matchboxnet import BinaryMatchboxNet
    from train.config import load_config

    names = KEYWORDS + ["_silence_", "_unknown_"]
    P = read_stream_params(Path("rtl/gen") / args.tag / "parameters.vh")
    if P["KWS_CONV4_POOL_BITS"] + 1 != MARGIN_BITS:
        raise SystemExit(f"POOL_BITS {P['KWS_CONV4_POOL_BITS']} does not give a {MARGIN_BITS}-bit margin")
    spec = WindowSpec(hop_frames=P["KWS_STREAM_HOP_FRAMES"])

    margin_float = args.margin_float
    if margin_float is None:
        for f in Path("configs").glob("streaming_*.json"):
            j = json.loads(f.read_text())
            if j.get("checkpoint_tag") == args.tag:
                margin_float = float(j["margin_float"])
    if margin_float is None:
        raise SystemExit("no --margin-float and no configs/streaming_*.json for this tag")

    run = Path(args.runs) / args.tag
    cfg = load_config(str(run / "config.yaml"))
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model = BinaryMatchboxNet(cfg.model)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    # ---- cases: identical construction to eval_streaming ----
    loaders = build_dataloaders(cfg.data, args.batch_size, cfg.afe.sample_rate,
                                num_workers=0, seed=cfg.train.seed)
    split_index = {"val": 1, "test": 2}[args.split]
    sample_indices, waves, labels = collect_balanced_waveforms(
        loaders[split_index], clips_per_class=args.clips_per_class,
        n_classes=cfg.model.n_classes, seed=args.seed)
    afe = AFEFrontend(cfg.afe)
    load_afe_state(afe, ck["afe"])
    afe.to(device).eval()
    chunks = []
    with torch.no_grad():
        for lo in range(0, len(waves), args.batch_size):
            native = afe(waves[lo:lo + args.batch_size].to(device), target_T=spec.native_frames)
            chunks.append((native > 0).to(torch.uint8).cpu().numpy())
    bits = np.concatenate(chunks, axis=0)
    silence_pool = np.flatnonzero(labels == SILENCE_INDEX)
    rng = np.random.default_rng(args.seed + 1)
    streams, inputs, per_case = [], [], []
    for local_index in range(len(labels)):
        before_i, after_i = rng.choice(silence_pool, size=2, replace=True)
        stream = splice_clips(bits[int(before_i)], bits[local_index], bits[int(after_i)], spec)
        snaps = make_snapshots(stream, spec)
        streams.append(stream)
        per_case.append(len(snaps))
        inputs.extend(s.model_input for s in snaps)
    n_win = per_case[0]
    if any(n != n_win for n in per_case):
        raise RuntimeError("streams produced different window counts")
    x_all = np.stack(inputs)

    # ---- float logits and integer pooled sums, same forward pass ----
    sites = tail_plan(model)
    grabbed = {}
    h = model.stages["conv2"].pw.register_forward_pre_hook(
        lambda mod, inp: grabbed.__setitem__("acc", binary_accumulator(mod, inp[0]).detach()))
    fl_logits, pooled = [], []
    try:
        with torch.no_grad():
            for lo in range(0, len(x_all), args.batch_size):
                x = torch.as_tensor(x_all[lo:lo + args.batch_size], dtype=torch.float32, device=device)
                fl_logits.append(model(x).cpu().numpy())
                pooled.append(fixed_logits(model, sites, grabbed["acc"]).cpu().numpy().astype(np.int64))
    finally:
        h.remove()
    fl_logits = np.concatenate(fl_logits)
    pooled = np.concatenate(pooled)
    lim = 1 << (P["KWS_CONV4_POOL_BITS"] - 1)
    if pooled.min() < -lim or pooled.max() >= lim:
        raise SystemExit(f"pooled sum {pooled.min()}..{pooled.max()} overflows POOL_BITS "
                         f"{P['KWS_CONV4_POOL_BITS']} -- the RTL would wrap here")
    k_idx, k_margin = keyword_and_margin(pooled)

    # ---- float gate (reproduces sweep_streaming_gate) and integer vote ----
    vote_cfg = VoteConfig(required_consecutive=P["KWS_STREAM_REQUIRED"],
                          cooldown_windows=P["KWS_STREAM_COOLDOWN_WINDOWS"],
                          n_classes=cfg.model.n_classes, quiet_classes=QUIET)
    fl_pred = fl_logits.argmax(axis=1)
    fl_cases, int_cases, rows = [], [], []
    boundary = 0
    for c in range(len(labels)):
        sl = slice(c * n_win, (c + 1) * n_win)
        target = int(labels[c])

        gated = []
        for p, lg in zip(fl_pred[sl], fl_logits[sl]):
            p = int(p)
            if p in QUIET or lg[p] - lg[list(QUIET)].max() >= margin_float:
                gated.append(p)
            else:
                gated.append(QUIET[0])
        fsteps = vote_sequence(gated, range(n_win), vote_cfg)
        fdets = [(w, s.detection) for w, s in enumerate(fsteps) if s.detection is not None]
        fl_cases.append(dict(target=target, **score_case(target, fdets, spec.hop_frames, spec.native_frames)))

        vote = IntVote(P["KWS_STREAM_MARGIN_INT"], P["KWS_STREAM_REQUIRED"], P["KWS_STREAM_COOLDOWN_WINDOWS"])
        dets = []
        for w in range(n_win):
            ki, km = int(k_idx[sl][w]), int(k_margin[sl][w])
            det, acc = vote.step(ki, km)
            if abs(km - P["KWS_STREAM_MARGIN_INT"]) <= 64:   # within one Q.6 LSB of pooled average
                boundary += 1
            if det is not None:
                dets.append((w, det))
            rows.append(dict(case_id=c, target=target, window_id=w, keyword_idx=ki, keyword_margin=km,
                             accept=int(acc), candidate=vote.candidate, streak=vote.streak,
                             cooldown=vote.cooldown, armed=int(vote.armed), quiet_seen=int(vote.quiet_seen),
                             detection="" if det is None else det,
                             float_pred=int(fl_pred[sl][w]), float_gated=gated[w]))
        if len(dets) > 2:
            raise RuntimeError(f"case {c}: {len(dets)} detections, the 40-bit format holds 2")
        sc = score_case(target, dets, spec.hop_frames, spec.native_frames)
        int_cases.append(dict(case_id=c, sample_index=int(sample_indices[c]), target=target,
                              detections=dets, crc16=crc16_windows(k_idx[sl], k_margin[sl]), **sc))

    fsum = summarize(fl_cases, spec.hop_frames, spec.native_frames)
    isum = summarize(int_cases, spec.hop_frames, spec.native_frames)
    same_det = sum(1 for a, b in zip(fl_cases, int_cases)
                   if (a["hit"], a["quiet_false"], a["wrong"]) == (b["hit"], b["quiet_false"], b["wrong"]))
    summary = {
        "tag": args.tag, "split": args.split, "clips_per_class": args.clips_per_class, "seed": args.seed,
        "windows_per_case": n_win, "policy": P, "margin_float": margin_float,
        "float_gate": fsum, "integer_rtl": isum,
        "cases_with_same_outcome": same_det, "cases": len(int_cases),
        "windows_within_64_of_margin": boundary,
        "window_argmax_float_eq_integer": float((fl_pred == pooled.argmax(axis=1)).mean()),
    }

    prefix = Path(args.out) if args.out else Path("out/streaming") / f"{args.tag}_{args.split}_int"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.parent / f"{prefix.name}_summary.json").write_text(json.dumps(summary, indent=2))
    with open(prefix.parent / f"{prefix.name}_windows.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"{args.tag} {args.split}: {len(int_cases)} cases x {n_win} windows, policy {P}")
    print(f"{'':16}{'recall':>18}{'quiet false':>14}{'wrong':>8}{'outside':>9}{'latency':>9}")
    for name, s in (("float gate", fsum), ("integer (RTL)", isum)):
        print(f"{name:16}{s['keyword_hits']:>6}/{s['keyword_cases']} ({100 * s['keyword_recall']:.2f}%)"
              f"{s['quiet_false_streams']:>8}/{s['quiet_cases']}{s['wrong_detections']:>8}"
              f"{s['outside_detections']:>9}{str(s['median_latency_ms']):>9}")
    print(f"same case outcome float vs integer: {same_det}/{len(int_cases)}; "
          f"windows within one LSB of the margin: {boundary}; "
          f"window argmax float==integer: {100 * summary['window_argmax_float_eq_integer']:.2f}%")

    # ---- chip vectors ----
    if args.export:
        out = Path(args.export)
        out.mkdir(parents=True, exist_ok=True)
        # first N cases of each class, interleaved (c0, c1, ..., c11, c0, ...) so that
        # any prefix of the ROM is still class-balanced
        order = []
        per = {cls: [i for i, c in enumerate(int_cases) if c["target"] == cls][:args.export_per_class]
               for cls in range(cfg.model.n_classes)}
        for r in range(args.export_per_class):
            for cls in range(cfg.model.n_classes):
                if r < len(per[cls]):
                    order.append(per[cls][r])
        with open(out / "stream_frames.hex", "w") as fh:
            for i in order:
                for fr in streams[i]:
                    fh.write(f"{int(sum(int(b) << ch for ch, b in enumerate(fr))):04x}\n")
        with open(out / "stream_expected.hex", "w") as fh:
            for i in order:
                c = int_cases[i]
                fh.write(f"{expected_word(c['target'], c['detections'], c['crc16']):010x}\n")
        sub = [int_cases[i] for i in order]
        with open(out / "stream_cases.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["rom_case", "eval_case_id", "sample_index", "target", "target_name", "detections",
                         "crc16", "hit", "quiet_false", "wrong", "outside"])
            for r, c in enumerate(sub):
                wr.writerow([r, c["case_id"], c["sample_index"], c["target"], names[c["target"]],
                             ";".join(f"w{w}:{k}" for w, k in c["detections"]), f"{c['crc16']:04x}",
                             c["hit"], c["quiet_false"], c["wrong"], c["outside"]])
        with open(out / "stream_trace.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["rom_case"] + list(rows[0].keys()))
            wr.writeheader()
            for r, i in enumerate(order):
                for row in rows[i * n_win:(i + 1) * n_win]:
                    wr.writerow({"rom_case": r, **row})
        man = {"tag": args.tag, "split": args.split, "cases": len(order), "frames_per_case": len(streams[0]),
               "windows_per_case": n_win, "policy": P,
               "expected_format": "[39:24] crc16 | [23:20] target | [19:10] det1 {v,win5,idx4} | [9:0] det0",
               "crc": "CRC-16/CCITT-FALSE over {idx[3:0], margin[21:0]} per window, MSB first",
               "subset_summary": summarize(sub, spec.hop_frames, spec.native_frames)}
        (out / "stream_manifest.json").write_text(json.dumps(man, indent=2))
        print(f"exported {len(order)} cases to {out}: subset {man['subset_summary']}")


if __name__ == "__main__":
    main()
