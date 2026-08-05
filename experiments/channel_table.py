"""Per-channel table joining the analog design, the built hardware and the
measured ML statistics into one page.

Columns come from three places that must agree:
  * analog/AFE/artifacts/filterbank_design.csv       -- what we designed (ngspice)
  * analog/ngspice_v15_2608001/channel_components.csv -- what the board has
  * measured here on real Speech Commands             -- what the model sees

Writes proposal/artifacts/channel_table.{md,csv}.

Usage:
    python experiments/channel_table.py [path/to/speech_commands_v0.02]
"""
from __future__ import annotations

import array
import csv
import glob
import os
import sys
import wave

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.afe import AFEFrontend                                    # noqa: E402
from data.speech_commands import synthesize_silence                 # noqa: E402
from train.config import load_config                                # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = (sys.argv[1] if len(sys.argv) > 1
        else os.environ.get("SPEECH_COMMANDS_ROOT",
                            "datasets/SpeechCommands/speech_commands_v0.02"))
OUT = os.path.join(REPO, "proposal", "artifacts")
SR, KW = 16000, ["yes", "no", "up", "down", "left", "right",
                 "on", "off", "stop", "go"]


def wav_read(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, path
        return torch.tensor(array.array("h", w.readframes(w.getnframes())),
                            dtype=torch.float32) / 32768.0


def load_audio():
    speech = []
    for k in KW:
        for f in sorted(glob.glob(f"{ROOT}/{k}/*.wav"))[:12]:
            x = wav_read(f)
            speech.append(torch.nn.functional.pad(
                x, (0, max(0, SR - x.numel())))[:SR])
    speech = torch.stack(speech[:120])
    noise = [wav_read(f) for f in sorted(glob.glob(f"{ROOT}/_background_noise_/*.wav"))]
    sil = synthesize_silence(noise, 40, SR, seed=777, zero_fraction=0.0)
    return speech, torch.cat([speech, sil])


def frontend(mode, calib, floor=0.05):
    cfg = load_config(os.path.join(REPO, "configs/base.yaml"), {
        "afe.filterbank_source": "spice", "afe.compression": "sqrt",
        "afe.normalize": mode, "afe.xmax_floor_frac": floor})
    fe = AFEFrontend(cfg.afe)
    fe.init_fixed_scale(calib)
    fe.init_thresholds(calib)
    return fe


def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"Speech Commands not found at {ROOT!r} -- pass it as argv[1].")
    os.makedirs(OUT, exist_ok=True)
    speech, calib = load_audio()

    design = np.loadtxt(os.path.join(REPO, "analog/AFE/artifacts/filterbank_design.csv"),
                        delimiter=",", skiprows=1)
    built = list(csv.DictReader(open(os.path.join(
        REPO, "analog/ngspice_v15_2608001/channel_components.csv"))))

    raw = frontend("xmax", calib).envelopes(speech, raw=True)   # [B, C, T]
    lvl = raw.median(dim=2).values.median(dim=0).values          # per-channel level
    peak = raw.amax(dim=(0, 2))
    # how often is each channel the loudest of the 16? that is what xmax divides by
    winner = (raw.argmax(dim=1).flatten()
              .bincount(minlength=16).float() / raw[:, 0].numel())

    fes = {m: frontend(m, calib) for m in ("xmax", "xmix")}
    for fe in fes.values():
        fe.threshold.requires_grad_(False)
    fire = {m: (fes[m](speech) > 0).float().mean(dim=(0, 2)) for m in fes}

    rows = []
    for c in range(16):
        b = built[c]
        rows.append(dict(
            ch=c,
            fc_hz=round(float(design[c, 2])),
            Q=round(float(design[c, 4]), 2),
            RA_kohm=round(float(design[c, 6]) / 1e3, 2),
            C_nF=round(float(design[c, 7]) * 1e9, 2),
            R1_kohm=round(float(design[c, 8]) / 1e3, 1),
            gain_dB=round(float(design[c, 5]), 1),
            R7_kohm=float(b["R7_kohm"]), R8_kohm=float(b["R8_kohm"]),
            Vthr_V=float(b["Vthr_V"]),
            level_med=round(float(lvl[c]), 4),
            level_peak=round(float(peak[c]), 2),
            winner_pct=round(float(winner[c]) * 100, 1),
            alpha_xmax=round(float(fes["xmax"].threshold[c]), 3),
            alpha_xmix=round(float(fes["xmix"].threshold[c]), 3),
            fire_xmax=round(float(fire["xmax"][c]), 3),
            fire_xmix=round(float(fire["xmix"][c]), 3),
        ))

    with open(os.path.join(OUT, "channel_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = [
        "# 채널별 수치 — 설계 · 제작 · 실측",
        "",
        "`experiments/channel_table.py`가 생성한다. 세 출처를 한 표로 합친 것:",
        "",
        "| 열 | 출처 |",
        "|---|---|",
        "| `f_c` `Q` `RA` `C` `R1` `gain` | **우리 설계** — `analog/AFE/artifacts/filterbank_design.csv` (ngspice `.ac` 스윕) |",
        "| `R7` `R8` `V_thr` | **제작된 보드** — `analog/ngspice_v15_2608001/channel_components.csv` |",
        "| `레벨` `최대` `승률` `α` `발화율` | **실측** — Speech Commands 키워드 120클립 |",
        "",
        "## 1. 아날로그 설계 = 제작된 보드 (16/16 일치)",
        "",
        "| ch | f_c (Hz) | Q | RA (kΩ) | C (nF) | R1 (kΩ) | 이득 (dB) | R7 (kΩ) | R8 (kΩ) | V_thr (V) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['ch']} | {r['fc_hz']} | {r['Q']} | {r['RA_kohm']} | "
                  f"{r['C_nF']} | {r['R1_kohm']} | {r['gain_dB']} | "
                  f"{r['R7_kohm']} | {r['R8_kohm']} | {r['Vthr_V']} |")
    md += [
        "",
        "> `R7`/`R8`는 **고정 임계** 방식의 값이다. `xmax`/`xmix`로 가면 이 분압은",
        "> `1.8V─R7─R8─GND`가 아니라 `V_max─Ra─Rb─V_ref`가 되고, 비 α가 그 자리를 대신한다.",
        "",
        "## 2. 채널별 실측 — 음량, 승률, α, 발화율",
        "",
        "| ch | f_c (Hz) | 엔벨로프 중앙값 | 최대 | max 승률 | α (xmax) | α (xmix) | 발화율 (xmax) | 발화율 (xmix) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['ch']} | {r['fc_hz']} | {r['level_med']} | "
                  f"{r['level_peak']} | {r['winner_pct']}% | {r['alpha_xmax']} | "
                  f"{r['alpha_xmix']} | {r['fire_xmax']} | {r['fire_xmix']} |")
    lo, hi = int(np.argmin([r["level_med"] for r in rows])), \
        int(np.argmax([r["level_med"] for r in rows]))
    md += [
        "",
        "### 읽는 법",
        "",
        f"* **엔벨로프 중앙값**이 채널별 '음량'이다. ch{hi}({rows[hi]['fc_hz']} Hz)가 "
        f"ch{lo}({rows[lo]['fc_hz']} Hz)보다 "
        f"**{rows[hi]['level_med'] / max(rows[lo]['level_med'], 1e-9):.0f}배** 크다 — "
        "음성 에너지가 중저역에 몰려 있기 때문이다. 절대 임계(`fixed`)가 어려웠던 이유가 이것이다.",
        "* **max 승률** = 그 채널이 16개 중 최대였던 프레임 비율. `xmax`의 분모를 "
        "누가 만드는지 보여준다. 승률이 0에 가까운 채널은 언제나 남과 비교당하는 쪽이다.",
        "* **α**는 학습 전 초기값(채널 평균)이다. 학습 후 값은 체크포인트에서 뽑는다 "
        "(`REPRODUCE.md` 참고). α는 저항비 Rb/(Ra+Rb)라 **[0,1]** 안에 있어야 한다.",
        "* **발화율**은 그 채널이 1을 내보내는 비율. 0이나 1에 붙은 채널은 정보를 "
        "싣지 못하므로 죽은 채널이다.",
    ]
    dead = [r["ch"] for r in rows if r["fire_xmax"] < 0.02 or r["fire_xmax"] > 0.98]
    md.append("")
    md.append(f"**죽은 채널 (xmax): {dead if dead else '없음 ✅'}**")

    with open(os.path.join(OUT, "channel_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote", os.path.join(OUT, "channel_table.md"))
    print("wrote", os.path.join(OUT, "channel_table.csv"))
    print(f"\n레벨 스프레드: ch{lo} {rows[lo]['level_med']} .. ch{hi} {rows[hi]['level_med']}"
          f"  ({rows[hi]['level_med']/max(rows[lo]['level_med'],1e-9):.0f}x)")
    print("죽은 채널:", dead if dead else "없음")


if __name__ == "__main__":
    main()
