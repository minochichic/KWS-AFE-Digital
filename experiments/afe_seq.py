"""Many GSC clips in one continuous ARB, and the software reference for it (wake plan step 4).

One AWG file per class (10 clips back to back) lets the board run a whole class without
anyone touching the generator. Each clip sits in a fixed slot -- lead quiet, 1 s clip, tail
quiet, 1.5 s by default, the same as the colleague's single-clip files -- so the n-th slot of
the file is the n-th clip, and a capture's place in the AWG period says which clip it was.

    # 10 test-split clips per class -> out/afe/seq/<name>/<class>_x10.arb + .json manifest
    python -m experiments.afe_seq build --gsc <speech_commands_v0.02> --per-class 10

    # software AFE + firmware trigger rule over one AWG period -> out/afe/<stem>/ref.json,
    # the reference experiments.afe_web lists (works on any .arb; the manifest adds labels)
    python -m experiments.afe_seq prep out/afe/seq/<name>/yes_x10.arb

Level. Like the colleague's converter, every keyword clip is scaled to the same Vpp (--norm
per-clip). The software side maps volts back to PCM with --pcm-per-volt (100 mVpp ~ PCM
peak-to-peak 0.29, from step 4's first clip; rough), the same calibration afe_clip uses for
single ARBs. Silence slots are background-noise crops kept at their dataset level through the
same factor, since scaling noise up to 100 mVpp would make it loud speech-band noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from experiments.afe_clip import ARB_WORDS, SR, afe_bits, classify, load, words_of
from experiments.eval_trigger import FIRMWARE_POLICY, WARMUP, simulate_triggers

PCM_PER_VOLT = 0.29 / 0.100          # PCM peak-to-peak 0.29 <-> 100 mVpp (afe_clip --arb-pp)
SHORT = 32767


def write_arb(path: Path, volts: np.ndarray, rate: int) -> Tuple[float, float]:
    """Keysight .arb, File Format 1.10, short data, CRLF -- the colleague's converter's format.
    Data spans -32767..32767 between Low Level and High Level (volts)."""
    lo, hi = float(volts.min()), float(volts.max())
    d = np.round((volts - lo) / max(hi - lo, 1e-12) * 2 * SHORT - SHORT).astype(np.int32)
    head = ["File Format:1.10", "Checksum:0", "Channel Count:1", f"Sample Rate:{rate:.6f}",
            f"High Level:{hi:.15g}", f"Low Level:{lo:.15g}", 'Data Type:"short"', 'Filter:"normal"',
            f"Data Points:{len(d)}", "Data:"]
    path.write_bytes(("\r\n".join(head + [str(int(x)) for x in d]) + "\r\n").encode("ascii"))
    return hi, lo


def read_arb(path: Path) -> Tuple[np.ndarray, int]:
    """.arb -> (volts, sample rate)."""
    lines = path.read_text().splitlines()
    i = lines.index("Data:")
    hdr = dict(l.split(":", 1) for l in lines[:i])
    hi, lo = float(hdr["High Level"]), float(hdr["Low Level"])
    d = np.array([int(v) for v in lines[i + 1:] if v.strip()], dtype=np.float64)
    return lo + (d + SHORT) / (2 * SHORT) * (hi - lo), int(float(hdr["Sample Rate"]))


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    import torch
    import torchaudio
    return torchaudio.functional.resample(torch.as_tensor(x, dtype=torch.float32), src, dst).numpy()


def pick(gsc: Path, word: str, n: int, seed: int, split: Optional[str]) -> List[Path]:
    """n clips of `word`, from the split's list if given (testing_list.txt), seeded."""
    files = sorted((gsc / word).glob("*.wav"))
    if split:
        lst = gsc / f"{split}_list.txt"
        if not lst.exists():
            raise SystemExit(f"{lst} not found -- pass --split none to draw from every clip")
        keep = {l.strip() for l in lst.read_text().splitlines()}
        files = [f for f in files if f"{word}/{f.name}" in keep]
    if len(files) < n:
        raise SystemExit(f"{word}: only {len(files)} clips available, asked for {n}")
    return sorted(random.Random(f"{seed}/{word}").sample(files, n))


def build(args) -> None:
    import soundfile as sf
    import torch
    from data.speech_commands import synthesize_silence

    gsc, out = Path(args.gsc), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = None if args.split == "none" else args.split
    lead, tail = round(args.lead_ms * SR / 1000), round(args.tail_ms * SR / 1000)
    vpp = args.vpp_mv / 1000
    print(f"{'file':<18} {'slots':>5} {'s':>6} {'points':>8} {'High V':>8} {'Low V':>8}")
    for word in args.classes:
        if word == "silence":
            noise = []
            for p in sorted((gsc / "_background_noise_").glob("*.wav")):
                w, sr = sf.read(str(p), dtype="float32")
                if sr == SR:
                    noise.append(torch.as_tensor(w))
            clips = synthesize_silence(noise, args.per_class, SR, args.seed, zero_fraction=0.0).numpy()
            srcs = [f"_background_noise_ crop {i} (seed {args.seed})" for i in range(args.per_class)]
        else:
            files = pick(gsc, word, args.per_class, args.seed, split)
            clips = []
            for f in files:
                w, sr = sf.read(str(f), dtype="float32", always_2d=False)
                if sr != SR:
                    raise SystemExit(f"{f}: {sr} Hz")
                clips.append(np.pad(w[:SR], (0, max(0, SR - len(w)))))
            srcs = [f"{word}/{f.name}" for f in files]
        volts, capped = [], 0
        for c in clips:
            v = resample(np.concatenate([np.zeros(lead), c, np.zeros(tail)]), SR, args.rate)
            if word != "silence" and args.norm == "per-clip":
                v = v * vpp / max(float(v.max() - v.min()), 1e-12)
            else:                                    # dataset level, same volts-per-PCM as the reference
                v = v / PCM_PER_VOLT if args.pcm_per_volt is None else v / args.pcm_per_volt
                pp = float(v.max() - v.min())
                if pp > args.max_mv / 1000:              # background noise at dataset level can be
                    v = v * (args.max_mv / 1000) / pp     # several hundred mVpp: keep the preamp linear
                    capped += 1
            volts.append(v)
        if capped:
            print(f"  {word}: {capped}/{len(clips)} slot(s) scaled down to {args.max_mv:g} mVpp "
                  f"(quieter than the dataset; prep reads the file, so the reference follows)")
        v = np.concatenate(volts)
        v = v - v.mean() + args.dc_mv / 1000         # like the colleague: mean at the bias
        stem = f"{word}_x{args.per_class}"
        hi, lo = write_arb(out / f"{stem}.arb", v, args.rate)
        man = {"kind": "sequence", "arb": f"{stem}.arb", "rate": args.rate, "lead_ms": args.lead_ms,
               "tail_ms": args.tail_ms, "slot_ms": args.lead_ms + 1000 + args.tail_ms, "norm": args.norm,
               "vpp_mv": args.vpp_mv, "dc_mv": args.dc_mv, "seed": args.seed, "split": split,
               "slots": [{"slot": i, "label": word, "src": s} for i, s in enumerate(srcs)]}
        (out / f"{stem}.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
        print(f"{stem + '.arb':<18} {len(clips):5d} {len(v) / args.rate:6.1f} {len(v):8d} {hi:8.4f} {lo:8.4f}")
    print(f"\nwrote {out}  -- next: python -m experiments.afe_seq prep {out}/<class>_x{args.per_class}.arb")


def label_of(word: str, names: List[str]) -> str:
    return word if word in names else "unknown"


def segments_of(arb: Path, volts: np.ndarray, rate: int, args, names: List[str]) -> Tuple[List[dict], float, str]:
    """Where each clip is in the file -> ([{start, end, word, label, src}] in samples, quiet level V, source).

    1. the colleague's converter: settings.json in the ARB's folder (export_mode "merged"):
       every clip at its own length, lead / gap / tail quiet between, per-clip 100 mVpp
    2. our afe_seq build manifest (<arb>.json): fixed slots
    3. neither: one clip between --lead-ms and --tail-ms"""
    st = arb.parent / "settings.json"
    if st.exists():
        s = json.loads(st.read_text(encoding="utf-8"))
        res = [r for r in s.get("results", []) if Path(r.get("output_name", "")).name == arb.name] \
            or s.get("results", [])[:1]
        if res and res[0].get("segments"):
            r = res[0]
            if int(r.get("sample_rate_hz") or rate) != rate:
                raise SystemExit(f"{st}: {r['sample_rate_hz']} Hz, the ARB says {rate}")
            if r.get("samples") and int(r["samples"]) != len(volts):
                raise SystemExit(f"{st}: {r['samples']} samples, the ARB has {len(volts)} -- not this file's settings")
            segs = [{"start": int(g["start_sample"]), "end": int(g["end_sample"]), "word": g["word"],
                     "label": label_of(g["word"], names), "src": f"{g['word']}/{g['source_filename']}"}
                    for g in sorted(r["segments"], key=lambda g: g["start_sample"])]
            q = r.get("silence_level_mv")
            return segs, (q / 1000 if q is not None else float(volts[0])), st.name
    man_path = Path(args.manifest) if args.manifest else arb.with_suffix(".json")
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        slot = round(man["slot_ms"] * rate / 1000)
        lead = round(man["lead_ms"] * rate / 1000)
        segs = [{"start": i * slot + lead, "end": i * slot + lead + rate, "word": s["label"],
                 "label": label_of(s["label"], names), "src": s["src"]} for i, s in enumerate(man["slots"])]
        return segs, float(volts[0]), man_path.name
    word = ARB_WORDS.get(re.match(r"[a-z]+", arb.stem).group(0), "unknown")
    lead, tail = round(args.lead_ms * rate / 1000), round(args.tail_ms * rate / 1000)
    return ([{"start": lead, "end": min(lead + rate, len(volts) - tail), "word": word,
              "label": label_of(word, names), "src": arb.name}], float(volts[0]), "file name")


def prep_one(arb: Path, args, model, afe, names: List[str]) -> Path:
    volts, rate = read_arb(arb)
    segs, quiet_v, how = segments_of(arb, volts, rate, args, names)
    # --scale: the AWG plays the file louder than its header says (Amplitude set by hand)
    pcm_per_volt = (args.pcm_per_volt or PCM_PER_VOLT) * args.scale
    vpp = float(volts.max() - volts.min())
    fps = rate / 100                                 # samples per 10 ms frame
    N = len(volts)
    P = N / fps                                      # AWG period in frames (need not be whole)

    clips = []
    for g in segs:                                   # the clip alone, zero-padded to 1 s like the dataset
        x = resample((volts[g["start"]:g["end"]] - quiet_v) * pcm_per_volt, rate, SR)
        x = np.pad(x[:SR], (0, max(0, SR - len(x))))
        clips.append(afe_bits(afe, x * 10 ** (args.gain_db / 20)))
    pred, prob = classify(model, np.stack(clips))

    # the firmware's view of the AWG looping: quiet, then the period repeated; each clip's image
    # at its start frame (OR: a short clip's zero padding may reach the next one)
    R = args.repeats
    T = args.quiet_frames + int(np.ceil((R + 1) * P)) + 100
    stream = np.zeros((T, 16), np.uint8)
    for r in range(R + 1):
        for g, c in zip(segs, clips):
            f0 = args.quiet_frames + round((r * N + g["start"]) / fps)
            stream[f0:f0 + 100] |= c[:max(0, min(100, T - f0))]
    caps = simulate_triggers(stream, FIRMWARE_POLICY, np.ones(16, bool), T)
    wins = [stream[c.start:c.end] for c in caps]
    wp, wq = classify(model, np.stack(wins)) if wins else ([], [])
    starts = np.array([g["start"] / fps for g in segs])
    sim_caps = []
    for c, w, p, q in zip(caps, wins, wp, wq):
        rel = c.onset - args.quiet_frames
        pas = int(rel // P)
        if pas != R - 1:                             # one steady-state period (the last full one)
            continue
        at = rel - pas * P
        i = int(np.searchsorted(starts, at + 0.5, side="right")) - 1   # clip whose start is before the onset
        if i < 0:
            continue
        sim_caps.append({"slot": i, "label": segs[i]["label"],
                         "end_mod": round(float((c.end - args.quiet_frames - pas * P) % P), 2),
                         "start_mod": round(float((c.start - args.quiet_frames - pas * P) % P), 2),
                         "window": words_of(w), "rates": w.mean(0).round(4).tolist(),
                         "pred": int(p), "prob": float(q)})
    out_segs = [{"slot": i, "label": g["label"], "word": g["word"], "src": g["src"],
                 "start_s": g["start"] / rate, "end_s": g["end"] / rate, "clip": words_of(c),
                 "clip_rates": c.mean(0).round(4).tolist(), "clip_pred": int(p), "clip_prob": float(q),
                 "captures": sum(s["slot"] == i for s in sim_caps)}
                for i, (g, c, p, q) in enumerate(zip(segs, clips, pred, prob))]
    labels = {g["label"] for g in segs}
    one = next(iter(labels)) if len(labels) == 1 else None
    first = sim_caps[0] if sim_caps else None
    ref = {"kind": "sequence", "wav": str(arb), "sha256": hashlib.sha256(arb.read_bytes()).hexdigest(),
           "label": names.index(one) if one in names else -1, "run": str(args.run), "layout": how,
           "rate": rate, "samples": N, "period_frames": P, "quiet_v": quiet_v,
           "pcm_per_volt": pcm_per_volt, "scale": args.scale, "gain_db": args.gain_db, "policy": FIRMWARE_POLICY.label,
           "segments": out_segs, "sim_caps": sim_caps,
           # what afe_clip refs carry, so older readers still find one capture at "0 dB"
           "gains": {"0": {"gain_db": 0.0, "clip": out_segs[0]["clip"], "clip_rates": out_segs[0]["clip_rates"],
                           "clip_pred": out_segs[0]["clip_pred"], "clip_prob": out_segs[0]["clip_prob"],
                           "captures": [first] if first else []}}}
    stem = arb.stem
    out = Path(args.out) / stem if args.out else Path(f"out/afe/{stem}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ref.json").write_text(json.dumps(ref))

    ok = sum(names[s["clip_pred"]] == s["label"] for s in out_segs)
    okc = sum(names[c["pred"]] == c["label"] for c in sim_caps)
    if quiet_v < 0.5 or vpp * args.scale < 0.05:
        print(f"\n!! {arb.name}: written as {vpp * 1000:.1f} mVpp around {quiet_v * 1000:.0f} mV, the others are "
              f"~100 mVpp around 900 mV. Played as written the AFE gets no 0.9 V bias and a tenth of the level: "
              f"remake it at 100 mVpp / 900 mV, or set the AWG by hand (Offset 900 mV, Amplitude k x "
              f"{vpp * 1000:.2f} mVpp) and prep with --scale k.")
    print(f"\n{arb.name}: {len(segs)} clips ({how}), AWG period {P / 100:.3f} s, quiet {quiet_v * 1000:.1f} mV, "
          f"volts -> PCM x{pcm_per_volt:.2f}, gain {args.gain_db:+g} dB")
    print(f"{'slot':>4} {'t (s)':>12} {'label':>8} {'clip ans':>9} {'prob':>5} {'caps':>4} {'cap ans':>9}  src")
    for s in out_segs:
        ca = [names[c["pred"]] for c in sim_caps if c["slot"] == s["slot"]]
        print(f"{s['slot']:4d} {s['start_s']:5.2f}-{s['end_s']:5.2f} {s['label']:>8} {names[s['clip_pred']]:>9} "
              f"{s['clip_prob']:5.2f} {s['captures']:4d} {','.join(ca) or '-':>9}  {s['src']}")
    print(f"software AFE: clips {ok}/{len(segs)} right; trigger rule takes {len(sim_caps)} capture(s) per period, "
          f"{okc} right -> {out / 'ref.json'}")
    return out


def prep(args) -> None:
    from data.speech_commands import KEYWORDS

    names = KEYWORDS + ["silence", "unknown"]
    arbs = []
    for a in map(Path, args.arb):
        arbs += sorted(a.glob("**/*.arb")) if a.is_dir() else [a]
    if not arbs:
        raise SystemExit("no .arb files")
    model, afe = load(Path(args.run))
    if afe_bits(afe, np.zeros(SR, np.float32)).any():
        raise SystemExit("software AFE fires on digital zero -- the quiet frames assume it does not")
    for a in arbs:
        prep_one(a, args, model, afe, names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="class-by-class continuous ARBs from GSC")
    b.add_argument("--gsc", required=True, help="speech_commands_v0.02 folder (class folders + testing_list.txt)")
    b.add_argument("--classes", nargs="+", default=["yes", "no", "up", "down", "left", "right", "on", "off",
                                                     "stop", "go", "silence"],
                   help="folder names; 'silence' = background-noise crops. For 'unknown', pass the "
                        "non-keyword folders you want, one ARB each")
    b.add_argument("--per-class", type=int, default=10)
    b.add_argument("--split", default="testing", help="testing | validation | none")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--rate", type=int, default=64000)
    b.add_argument("--lead-ms", type=float, default=250.0)
    b.add_argument("--tail-ms", type=float, default=250.0)
    b.add_argument("--norm", choices=["per-clip", "native"], default="per-clip",
                   help="per-clip: every keyword clip at --vpp-mv (the colleague's files); "
                        "native: dataset loudness through --pcm-per-volt")
    b.add_argument("--vpp-mv", type=float, default=100.0)
    b.add_argument("--dc-mv", type=float, default=900.0)
    b.add_argument("--max-mv", type=float, default=100.0,
                   help="cap on any slot's Vpp in native/silence slots (preamp headroom)")
    b.add_argument("--pcm-per-volt", type=float, default=None, help=f"default {PCM_PER_VOLT:.2f}")
    b.add_argument("--out", default="out/afe/seq/x10")
    p = sub.add_parser("prep", help="software reference for ARBs (one clip or many)")
    p.add_argument("arb", nargs="+", help=".arb files or folders (searched recursively)")
    p.add_argument("--manifest", default=None, help="default: the .json next to the .arb")
    p.add_argument("--run", default="runs/bd_base")
    p.add_argument("--out", default=None, help="parent folder; default out/afe -> out/afe/<arb stem>/ref.json")
    p.add_argument("--lead-ms", type=float, default=250.0, help="without a manifest")
    p.add_argument("--tail-ms", type=float, default=250.0, help="without a manifest")
    p.add_argument("--pcm-per-volt", type=float, default=None, help=f"default {PCM_PER_VOLT:.2f}")
    p.add_argument("--scale", type=float, default=1.0,
                   help="the AWG plays the file this many times its written Vpp (Amplitude set by hand)")
    p.add_argument("--gain-db", type=float, default=0.0, help="extra software gain on top of the calibration")
    p.add_argument("--quiet-frames", type=int, default=2 * WARMUP)
    p.add_argument("--repeats", type=int, default=3, help="AWG periods simulated; the last one is kept")
    args = ap.parse_args()
    build(args) if args.cmd == "build" else prep(args)


if __name__ == "__main__":
    main()
