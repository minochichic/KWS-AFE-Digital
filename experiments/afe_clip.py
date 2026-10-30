"""One GSC clip through the real AFE, against the software AFE (wake plan step 4).

The board path under test:

    AWG plays the clip (experiments/gsc_to_edu33212a.py CSV: 250 ms quiet,
    1 s clip, 500 ms quiet) -> AFE -> 16 comparators -> J5 -> STM32 (-D
    WAKE_AFELOG) triggers, captures 100 frames, sends them to the FPGA
    -> tools/read_stats.ps1 -Afe writes the captures to out/afe/*.json

The analog chain is not the STFT+mel simulation, so a bit-for-bit match is not
the test (docs/edu33212a_gsc_lab.md). What we can check is whether the board
sees the same *kind* of image: per-channel firing rate, where the trigger
lands, and what the network answers. Because `normalize: fixed` makes every
threshold absolute, the playback level is part of the test: `prep` runs the
software AFE at several gains, and `check` reports which gain the hardware
image looks like.

    # before the lab: reference for the clip (sim image, rates vs gain, the
    # capture the firmware rule should take, best.pt answer)
    python -m experiments.afe_clip prep --run runs/bd_base --wav left/54aecbd5_nohash_1.wav

    # after: the board's captures against that reference
    python -m experiments.afe_clip check out/afe/54aecbd5_nohash_1/ref.json out/afe/afe_clip_*.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiments.eval_trigger import FIRMWARE_POLICY, NATIVE, WARMUP, _padded, simulate_triggers

SR = 16000
FRAME = SR // 100                 # samples per 10 ms frame
GAINS = [-12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0]
SHIFT = 10                        # frames searched each way when aligning a board capture


def bits_of(words: List[int]) -> np.ndarray:
    """[100] uint16 frame words (bit k = ch k) -> [100, 16] 0/1."""
    w = np.asarray(words, dtype=np.uint32)
    return ((w[:, None] >> np.arange(16)) & 1).astype(np.uint8)


def words_of(bits: np.ndarray) -> List[int]:
    return [int(x) for x in (bits.astype(np.uint32) << np.arange(16)).sum(1)]


def load(run: Path):
    import torch
    from data.afe import AFEFrontend, load_afe_state
    from models.binary_matchboxnet import BinaryMatchboxNet
    from train.config import load_config

    cfg = load_config(str(run / "config.yaml"))
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model = BinaryMatchboxNet(cfg.model)
    model.load_state_dict(ck["model"])
    model.eval()
    afe = AFEFrontend(cfg.afe)
    load_afe_state(afe, ck["afe"])
    afe.eval()
    return model, afe


def classify(model, windows: np.ndarray):
    """[N, 100, 16] 0/1 -> (class, softmax of that class) per window."""
    import torch
    from experiments.eval_streaming import _infer

    scores, pred = _infer(model, np.stack([_padded(x) for x in windows]), torch.device("cpu"), 64)
    p = np.exp(scores - scores.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return pred, p[np.arange(len(pred)), pred]


def afe_bits(afe, wave: np.ndarray) -> np.ndarray:
    """One 1 s waveform -> [100, 16] 0/1. The software AFE always returns clip_ms of frames."""
    import torch
    with torch.no_grad():
        return (afe(torch.as_tensor(wave[None].astype(np.float32))) > 0)[0].to(torch.uint8).numpy().T


ARB_WORDS = {"dow": "down", "lef": "left", "rig": "right", "sto": "stop", "yes": "yes",
             "no": "no", "up": "up", "on": "on", "off": "off", "go": "go"}


def load_arb(path: Path, pp: float, lead_ms: float, tail_ms: float):
    """Keysight .arb (File Format 1.10, short data) from the colleague's WAV converter ->
    16 kHz waveform with peak-to-peak `pp` in PCM units (1.0 = full scale), and the word
    from the file name (rig00034.arb -> right).

    The converter scales every clip to the same Vpp, so the file keeps the shape but not the
    loudness: `pp` puts it back on the PCM scale. The first sample is the silent lead."""
    import torch
    import torchaudio

    lines = path.read_text().splitlines()
    i = lines.index("Data:")
    hdr = dict(l.split(":", 1) for l in lines[:i])
    rate = int(float(hdr["Sample Rate"]))
    s = np.array([int(v) for v in lines[i + 1:] if v.strip()], dtype=np.float64)
    s -= s[0]                                        # lead is digital silence
    s = s[round(lead_ms * rate / 1000):len(s) - round(tail_ms * rate / 1000)]
    x = torchaudio.functional.resample(torch.as_tensor(s, dtype=torch.float32), rate, SR).numpy()
    x *= pp / max(1e-9, float(x.max() - x.min()))
    word = ARB_WORDS.get(re.match(r"[a-z]+", path.stem).group(0), "unknown")
    return x[:SR], word, rate


def prep(args) -> None:
    import soundfile as sf
    from data.speech_commands import KEYWORDS

    names = KEYWORDS + ["silence", "unknown"]
    wav = Path(args.wav)
    if wav.suffix.lower() == ".arb":
        raw, word, _ = load_arb(wav, args.arb_pp, args.arb_lead_ms, args.arb_tail_ms)
        print(f"{wav.name}: ARB -> {word}, scaled to peak-to-peak {args.arb_pp} (PCM units)")
    else:
        raw, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if sr != SR:
            raise SystemExit(f"{wav}: {sr} Hz, expected {SR}")
        word = wav.parent.name
    raw = raw.reshape(-1)
    x = np.pad(raw[:SR], (0, max(0, SR - len(raw))))
    label = KEYWORDS.index(word) if word in KEYWORDS else len(KEYWORDS) + 1
    model, afe = load(Path(args.run))
    out = Path(args.out or f"out/afe/{wav.stem}")
    out.mkdir(parents=True, exist_ok=True)

    lead, tail = round(args.lead_ms / 10), round(args.tail_ms / 10)         # frames
    every = np.ones(16, bool)
    if afe_bits(afe, np.zeros(SR, np.float32)).any():
        raise SystemExit("software AFE fires on digital zero -- the quiet frames below assume it does not")

    gains: Dict[str, dict] = {}
    for g in args.gains:
        clip = afe_bits(afe, x * 10 ** (g / 20))
        # what the board should do: quiet (warmup), then the burst, repeated. Digital zero gives
        # all-off frames, so the AWG's quiet parts are all-off frames (eval_trigger_history "zero")
        burst = np.concatenate([np.zeros((lead, 16), np.uint8), clip, np.zeros((tail, 16), np.uint8)])
        stream = np.concatenate([np.zeros((args.quiet_frames, 16), np.uint8)] + [burst] * args.repeats)
        caps = simulate_triggers(stream, FIRMWARE_POLICY, every, len(stream))
        wins = [stream[c.start:c.end] for c in caps]
        pred, prob = classify(model, np.stack([clip] + wins))
        b0 = args.quiet_frames + lead                              # clip's first frame, burst 0
        blen = len(burst)
        gains[f"{g:g}"] = {
            "gain_db": g,
            "clip": words_of(clip),
            "clip_rates": clip.mean(0).round(4).tolist(),
            "clip_pred": int(pred[0]), "clip_prob": float(prob[0]),
            "captures": [{"start": c.start, "onset": c.onset,
                          "burst": int((c.start - args.quiet_frames) // blen),
                          "start_vs_clip": int(c.start - b0 - ((c.start - args.quiet_frames) // blen) * blen),
                          "window": words_of(w), "rates": w.mean(0).round(4).tolist(),
                          "pred": int(p), "prob": float(q)}
                         for c, w, p, q in zip(caps, wins, pred[1:], prob[1:])],
        }

    ref = {"wav": str(wav), "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
           "label": label, "run": str(args.run), "lead_ms": args.lead_ms, "tail_ms": args.tail_ms,
           "quiet_frames": args.quiet_frames, "repeats": args.repeats,
           "policy": FIRMWARE_POLICY.label, "gains": gains}
    (out / "ref.json").write_text(json.dumps(ref))

    rms = float(np.sqrt((x ** 2).mean()))
    print(f"{wav.name}: label {names[label]}, {len(raw) / SR:.3f} s, RMS {20 * np.log10(max(rms, 1e-9)):.1f} dBFS, "
          f"peak {20 * np.log10(max(np.abs(x).max(), 1e-9)):.1f} dBFS")
    print(f"burst = {args.lead_ms:g} ms quiet + 1 s clip + {args.tail_ms:g} ms quiet "
          f"(matches gsc_to_edu33212a defaults), {args.repeats}x after {args.quiet_frames} quiet frames\n")
    print(f"{'gain':>5} | {'mean on':>7} {'clip ans':>9} | {'caps':>4} {'start vs clip':>13} {'capture ans':>14}")
    for k, v in gains.items():
        on = float(np.sum(v["clip_rates"]))
        cs = v["captures"]
        st = sorted({c["start_vs_clip"] for c in cs})
        ans = sorted({names[c["pred"]] for c in cs})
        print(f"{k:>5} | {on:7.2f} {names[v['clip_pred']]:>9} | {len(cs):4d} {','.join(map(str, st)) or '-':>13} "
              f"{','.join(ans) or '-':>14}")
    print(f"\nper-channel firing rate of the clip at 0 dB (fraction of the 100 frames):")
    print("  " + " ".join(f"{r:.2f}" for r in gains["0"]["clip_rates"]))
    print(f"\nwrote {out / 'ref.json'}")
    plot_ref(ref, out / "ref.png", names)


def plot_ref(ref: dict, path: Path, names: List[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g0 = ref["gains"]["0"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [2, 1]})
    ax[0].imshow(bits_of(g0["clip"]).T, aspect="auto", origin="lower", cmap="Greys", interpolation="nearest")
    ax[0].set(xlabel="frame (10 ms)", ylabel="channel",
              title=f"{Path(ref['wav']).name} software AFE at 0 dB -> {names[g0['clip_pred']]}")
    gs = sorted(ref["gains"].values(), key=lambda v: v["gain_db"])
    ax[1].plot([v["gain_db"] for v in gs], [sum(v["clip_rates"]) for v in gs], "o-", color="#2a6fdb")
    ax[1].set(xlabel="playback gain (dB)", ylabel="mean channels on / frame", title="level sensitivity")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}")


def check(args) -> None:
    from data.speech_commands import KEYWORDS

    names = KEYWORDS + ["silence", "unknown"]
    ref = json.loads(Path(args.ref).read_text())
    model, _ = load(Path(ref["run"]))
    g0 = ref["gains"]["0"]
    if not g0["captures"]:
        raise SystemExit("the reference has no capture at 0 dB -- re-run prep")
    want = bits_of(g0["captures"][0]["window"])                   # what the rule should send
    label = ref["label"]

    rows = []
    for path in args.board:
        log = json.loads(Path(path).read_text())
        q = log["quiet"]
        if q["n"]:
            print(f"{Path(path).name}: {log['cap_n']} captures since reset, quiet {q['n'] / 100:.1f} s, "
                  f"quiet on-rate ch0..15 = " + " ".join(f"{o / q['n']:.2f}" for o in q["on"]))
        for c in log["caps"]:
            rows.append((Path(path).name, c, bits_of(c["clip"])))
    if not rows:
        raise SystemExit("no captures in the board logs")

    pred, prob = classify(model, np.stack([b for _, _, b in rows]))
    print(f"\nreference: {Path(ref['wav']).name} ({names[label]}), sim capture at 0 dB starts "
          f"{g0['captures'][0]['start_vs_clip']} frames vs clip -> {names[g0['captures'][0]['pred']]}")
    print(f"\n{'#':>2} {'tick':>7} {'bg':>5} {'on':>5} | {'shift':>5} {'bits =':>6} | {'like gain':>9} | "
          f"{'FPGA':>7} {'best.pt':>12} | label")
    gain_keys = sorted(ref["gains"], key=lambda k: ref["gains"][k]["gain_db"])
    for i, ((_, c, b), p, pr) in enumerate(zip(rows, pred, prob)):
        best = max(range(-SHIFT, SHIFT + 1), key=lambda s: agree(b, want, s))
        rates = b.mean(0)
        # the gain whose simulated capture has the closest per-channel firing rates
        like = min(gain_keys, key=lambda k: np.abs(rates - capture_rates(ref["gains"][k])).sum())
        g = c["got"]
        print(f"{i + 1:2d} {c['end']:7d} {c['bg_q'] / 32:5.2f} {rates.sum():5.2f} | {best:+5d} "
              f"{agree(b, want, best):6.3f} | {like:>6} dB | "
              f"{names[g] if g < 12 else hex(g):>7} {names[p]:>7} {pr:4.2f} | "
              f"{'ok' if p == label else 'WRONG'}{'' if g == p else '  <- FPGA != best.pt'}")
    print("\nshift: board capture vs the simulated one (+ = the board triggered later); "
          "bits =: fraction of the 1600 bits equal at that shift")
    print("like gain: the software gain whose capture has the closest per-channel firing rates")
    if args.plot:
        plot_check(rows, want, ref, pred, names, Path(args.plot))


def capture_rates(v: dict) -> np.ndarray:
    return np.asarray(v["captures"][0]["rates"]) if v["captures"] else np.zeros(16)


def agree(b: np.ndarray, want: np.ndarray, s: int) -> float:
    """Bits equal between b shifted by s frames and want, over the overlap."""
    if s >= 0:
        x, y = b[s:], want[:NATIVE - s]
    else:
        x, y = b[:NATIVE + s], want[-s:]
    return float((x == y).mean())


def plot_check(rows, want, ref, pred, names, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, ax = plt.subplots(n + 1, 2, figsize=(11, 1.9 * (n + 1)), squeeze=False,
                           gridspec_kw={"width_ratios": [3, 1]})
    ax[0, 0].imshow(want.T, aspect="auto", origin="lower", cmap="Greys", interpolation="nearest")
    ax[0, 0].set_title(f"simulated capture at 0 dB -> {names[ref['gains']['0']['captures'][0]['pred']]}", fontsize=9)
    ax[0, 1].bar(range(16), want.mean(0), color="#999")
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].set_title("firing rate per channel", fontsize=9)
    for i, ((name, c, b), p) in enumerate(zip(rows, pred), 1):
        ax[i, 0].imshow(b.T, aspect="auto", origin="lower", cmap="Blues", interpolation="nearest")
        g = c["got"]
        ax[i, 0].set_title(f"board capture {i} (tick {c['end']}): FPGA {names[g] if g < 12 else hex(g)}, "
                           f"best.pt {names[p]}", fontsize=9)
        ax[i, 1].bar(range(16), want.mean(0), color="#ccc", label="sim")
        ax[i, 1].bar(range(16), b.mean(0), color="#2a6fdb", width=0.5, label="board")
        ax[i, 1].set_ylim(0, 1)
    ax[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep", help="software reference for one clip")
    p.add_argument("--run", default="runs/bd_base")
    p.add_argument("--wav", required=True, help="GSC wav; its folder name is the label")
    p.add_argument("--out", default=None, help="default out/afe/<wav stem>")
    p.add_argument("--gains", type=float, nargs="+", default=GAINS)
    p.add_argument("--lead-ms", type=float, default=250.0, help="quiet before the clip in one burst")
    p.add_argument("--tail-ms", type=float, default=500.0, help="quiet after it")
    p.add_argument("--quiet-frames", type=int, default=2 * WARMUP, help="quiet before the first burst")
    p.add_argument("--repeats", type=int, default=3, help="bursts in the simulated stream")
    # .arb input: the colleague's files are each scaled to 100 mVpp. Step 4's first clip
    # (left/54aecbd5_nohash_1, PCM peak-to-peak 0.82) at 100 mVpp looked like the software
    # AFE at -9 dB -> 100 mVpp ~ PCM peak-to-peak 0.82 * 10^(-9/20) = 0.29. One clip, rough.
    p.add_argument("--arb-pp", type=float, default=0.29, help=".arb: PCM peak-to-peak for 0 dB")
    p.add_argument("--arb-lead-ms", type=float, default=250.0, help=".arb: silent lead in the file")
    p.add_argument("--arb-tail-ms", type=float, default=250.0, help=".arb: silent tail in the file")
    c = sub.add_parser("check", help="board captures against the reference")
    c.add_argument("ref", help="ref.json from prep")
    c.add_argument("board", nargs="+", help="out/afe/afe_*.json from read_stats.ps1 -Afe")
    c.add_argument("--plot", default=None, help="write a PNG: board vs simulated capture")
    args = ap.parse_args()
    prep(args) if args.cmd == "prep" else check(args)


if __name__ == "__main__":
    main()
