"""비교기 임계 전압 -- AWG 로 GSC 클립을 프리앰프 출력 노드에 넣는 벤치 실험용.

벤치 (2026-10-29): 마이크 대신 파형 발생기가 GSC wav 를 arb 로 내보내고, 그것을
**프리앰프 출력 노드**에 직접 넣는다 (오프셋 900 mV, 100 mVpp). 이 조건에서는
"정규화 1.0 이 몇 mV 인가"(board_thresholds 의 미지수 K)를 추정할 필요가 없다 --
주입 레벨을 알기 때문이다.

  1. 소프트웨어: 채널 f_c 정현파를 진폭을 바꿔 가며 bd_base 의 AFE(같은 필터뱅크,
     같은 fixed 정규화)에 넣어, 정규화 엔벨로프가 그 채널의 학습 임계값 thr 와
     같아지는 디지털 진폭 a_thr 를 이분법으로 찾는다.
  2. AWG: arb 변환이 클립을 최대값 기준으로 맞췄다면 디지털 1 = (Vpp/2) / peak.
     a_thr -> 프리앰프 출력 노드 진폭 A_thr [mV].
  3. 하드웨어: A_thr 이 검출기 출력(V+)을 올리는 폭 dV 를 swing_board.md 의 두 점
     (프리앰프 출력 1 mV = 마이크 0.1 mV, 4 mV = 마이크 0.4 mV; 프리앰프 x10)과
     원점을 잇는 꺾은선으로 환산한다. 검출기는 작은 신호에서 비선형이다 (ch15:
     1 mV 에서 1.65, 4 mV 에서 15.99 -- 4배 입력에 9.7배). 그래서 선 하나가 아니라
     꺾은선이고, 1 mV 아래는 원점까지 직선이라는 가정이 들어간다.
  4. V_thr = 그 채널의 V_Env 실측 (dc_measured_*.csv: 프리앰프 분리, 입력 0.9 V DC) + dV.

⚠️ AWG 가 클립마다 최대값을 100 mVpp 에 맞추면, 모델이 학습한 절대 음량이 클립마다
달라진다 (bd_base 는 normalize=fixed -- 음량이 곧 입력이다). 여러 클립을 같은 임계
전압으로 돌리려면 클립마다 Vpp 를 peak 에 비례시켜 "디지털 1 당 mV" 를 고정한다.
--wav 를 여러 개 주면 그 Vpp 를 같이 낸다.

"디지털 1 = 몇 mV" 는 arb 파일과 원본 wav 를 최소제곱으로 맞춰 얻는 것이 가장 정확하다
(--arb + --wav). arb 를 만들 때 4배 보간이 꼭짓점을 바꾸므로 Vpp/peak 로 계산한 값과
~1% 다르다. 그 값을 이미 알면 --mv-per-unit 로 바로 준다.

--wav 를 주면 그 클립에 대해 모델이 기대하는 비교기 출력(16채널 x 100칸)도 찍는다 --
벤치에서 비교기 출력을 볼 때의 정답지다.

체크포인트가 있는 학습 박스에서:
    python -m experiments.awg_thresholds --tag bd_base --mv-per-unit 121.52 --wav <원본 wav>
    python -m experiments.awg_thresholds --tag bd_base --arb lef00005.arb --wav <원본 wav>
    python -m experiments.awg_thresholds --tag bd_base --vpp-mv 100 --wav <원본 wav>   # peak 로 근사
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from experiments.board_thresholds import (DC_CSV, VOS_MV, board_rows,
                                          measured, measured_dc)

PREAMP_GAIN = 10.0           # swing_board.md: 스윙은 마이크 단 구동, 프리앰프 x10 포함
PRE_MV = (1.0, 4.0)          # 마이크 0.1 / 0.4 mV 열의 프리앰프 출력 진폭


def swing_at(a_pre_mv: float, s1: float, s4: float) -> Tuple[float, str]:
    """프리앰프 출력 진폭 [mV] -> V+ 상승 [mV]. (0,0)-(1,s1)-(4,s4) 꺾은선.

    4 mV 위는 마지막 기울기로 연장하되 'extrap' 을 단다 -- 저역은 100 mV 근처에서
    압축되므로 그 구간은 과대평가다."""
    if a_pre_mv <= 0:
        return 0.0, ""
    if a_pre_mv <= PRE_MV[0]:
        return s1 * a_pre_mv / PRE_MV[0], "lin<1mV"
    if a_pre_mv <= PRE_MV[1]:
        f = (a_pre_mv - PRE_MV[0]) / (PRE_MV[1] - PRE_MV[0])
        return s1 + f * (s4 - s1), ""
    slope = (s4 - s1) / (PRE_MV[1] - PRE_MV[0])
    return s4 + slope * (a_pre_mv - PRE_MV[1]), "extrap>4mV"


def tone_threshold_amplitudes(afe, thr, fcs, sr: int, n: int) -> List[float]:
    """채널마다: 정규화 엔벨로프가 thr[ch] 가 되는 f_c 정현파의 디지털 진폭.

    엔벨로프는 클립 가운데 프레임들의 중앙값(정상 상태). 디지털 진폭 1.0 에서도
    못 닿으면 nan."""
    import torch
    t = torch.arange(n, dtype=torch.float32) / sr

    def env_at(ch: int, a: float) -> float:
        w = (a * torch.sin(2 * np.pi * fcs[ch] * t)).unsqueeze(0)
        with torch.no_grad():
            e = afe.envelopes(w)[0, ch]
        k = e.shape[-1]
        return float(e[k // 5: 4 * k // 5].median())

    out = []
    for ch in range(len(fcs)):
        target = float(thr[ch])
        if env_at(ch, 1.0) < target:
            out.append(float("nan"))
            continue
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if env_at(ch, mid) < target:
                lo = mid
            else:
                hi = mid
        out.append(hi)
    return out


def read_arb(path: Path) -> Tuple[float, np.ndarray]:
    """Keysight 형식 .arb -> (샘플레이트, mV). 코드 -32767..32767 이 Low..High Level."""
    lines = Path(path).read_text().splitlines()
    hdr, i = {}, 0
    while i < len(lines) and not lines[i].startswith("Data:"):
        k, _, v = lines[i].partition(":")
        hdr[k.strip()] = v.strip()
        i += 1
    codes = np.array([int(x) for x in lines[i + 1:] if x.strip()], dtype=float)
    hi, lo = float(hdr["High Level"]), float(hdr["Low Level"])
    mv = (lo + (codes + 32767) * (hi - lo) / 65534) * 1000.0
    return float(hdr["Sample Rate"]), mv


def fit_mv_per_unit(arb_sr: float, arb_mv: np.ndarray, wav: np.ndarray,
                    wav_sr: int) -> Tuple[float, float]:
    """arb(전압) = g * wav(디지털) 인 g 와 정규화 상관. 정수배 업샘플만 가정한다."""
    k = int(round(arb_sr / wav_sr))
    if abs(arb_sr - k * wav_sr) > 1e-6:
        raise SystemExit(f"arb {arb_sr} Hz 가 wav {wav_sr} Hz 의 정수배가 아니다")
    best = None
    for phase in range(k):
        d = arb_mv[phase::k]
        if len(d) < len(wav):
            continue
        # wav 를 arb 안에서 미끄러뜨려 가장 잘 맞는 위치
        c = np.correlate(d - d.mean(), wav - wav.mean(), mode="valid")
        lag = int(np.argmax(c))
        seg = d[lag:lag + len(wav)]
        g = float(seg @ wav / (wav @ wav))
        r = float(np.corrcoef(seg, wav)[0, 1])
        if best is None or r > best[1]:
            best = (g, r)
    if best is None:
        raise SystemExit("arb 가 wav 보다 짧다")
    return best


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    """16비트 PCM wav -> ([-1, 1) float, 샘플레이트). GSC 형식이면 충분하고, torchaudio
    백엔드가 없는 기계(Mac 의 venv)에서도 돈다. 첫 채널만."""
    import wave
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: 16비트 PCM 이 아니다")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        x = x.reshape(-1, w.getnchannels())[:, 0].astype(np.float32) / 32768.0
        return x, w.getframerate()


def wav_peak(path: Path) -> float:
    return float(np.abs(load_wav(path)[0]).max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="bd_base")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--vpp-mv", type=float, default=100.0, help="AWG 진폭 설정 [mVpp]")
    ap.add_argument("--wav", type=Path, nargs="+",
                    help="AWG 로 넣은 클립의 원본 GSC wav. 첫 파일이 기준; 나머지는 같은 "
                         "보정을 유지할 Vpp 를 계산한다. 첫 파일의 예상 비교기 출력도 찍는다")
    ap.add_argument("--arb", type=Path, help="AWG 에 넣은 .arb -- --wav 첫 파일과 맞춰 mV/단위를 구한다")
    ap.add_argument("--mv-per-unit", type=float, help="디지털 1 = 몇 mV (이미 알면)")
    ap.add_argument("--peak", type=float, help="arb 변환 때 Vpp 에 맞춘 클립 최대값 (디지털)")
    ap.add_argument("--dc-csv", type=Path, default=DC_CSV)
    args = ap.parse_args()

    import torch
    from data.afe import AFEFrontend, load_afe_state
    from train.config import load_config

    run = Path(args.runs) / args.tag
    cfg = load_config(str(run / "config.yaml"))
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if "afe" not in ck:
        raise SystemExit("AFE 상태가 없는 런이다 (프레임 학습?)")
    afe = AFEFrontend(cfg.afe)
    load_afe_state(afe, ck["afe"])
    afe.eval()
    if cfg.afe.normalize != "fixed":
        raise SystemExit(f"normalize={cfg.afe.normalize!r}: 절대 임계가 아니라 전압으로 못 옮긴다")
    thr = afe.threshold.detach().reshape(-1)

    bom_name, board = board_rows()
    fcs = [board[c]["fc"] for c in range(16)]
    _, s1, up1 = measured(0.1)
    _, s4, up4 = measured(0.4)
    dc = measured_dc(args.dc_csv)

    sr = int(cfg.afe.sample_rate)
    peaks = [wav_peak(p) for p in args.wav] if args.wav else []
    if args.mv_per_unit:
        mv_per_unit, how = args.mv_per_unit, "--mv-per-unit"
    elif args.arb:
        if not args.wav:
            raise SystemExit("--arb 는 원본 --wav 와 같이 준다")
        w0, wsr = load_wav(args.wav[0])
        if wsr != sr:
            raise SystemExit(f"{args.wav[0]}: {wsr} Hz, 모델은 {sr} Hz")
        asr, amv = read_arb(args.arb)
        mv_per_unit, r = fit_mv_per_unit(asr, amv, w0.astype(float), wsr)
        how = f"{args.arb.name} 와 원본을 최소제곱으로 맞춤 (상관 {r:.4f})"
        if r < 0.99:
            print(f"⚠️  상관 {r:.3f} -- arb 가 이 wav 에서 온 게 아닐 수 있다")
    elif peaks or args.peak:
        pk = peaks[0] if peaks else args.peak
        mv_per_unit, how = (args.vpp_mv / 2.0) / pk, f"Vpp/2 / peak {pk:.4f} (근사)"
    else:
        raise SystemExit("--mv-per-unit, --arb + --wav, --wav, --peak 중 하나가 필요하다")
    n = int(sr * cfg.afe.clip_ms / 1000)
    a_thr = tone_threshold_amplitudes(afe, thr, fcs, sr, n)

    print(f"체크포인트: {run}/best.pt  (normalize=fixed, lo={float(afe.fixed_lo):.4g}, "
          f"hi={float(afe.fixed_hi):.4g})")
    print(f"AWG:        디지털 1 = {mv_per_unit:.2f} mV (프리앰프 출력 노드) -- {how}"
          + (f"  [{args.wav[0].name}]" if args.wav else ""))
    print(f"            = 마이크 환산 {mv_per_unit / PREAMP_GAIN:.2f} mV / 디지털 1 "
          f"(설계점 0.4 mV 정현파 = 프리앰프 4 mV)")
    print(f"DC:         {args.dc_csv.name}   스윙: swing_board.md (시뮬), BOM {bom_name}\n")
    print(f"{'ch':>3}{'f_c':>7}{'thr':>8}{'a_thr':>9}{'A_thr':>9}{'dV':>8}"
          f"{'V_Env':>8}{'V_thr':>9}  비고")
    rows = []
    for c in range(16):
        if not np.isfinite(a_thr[c]):
            print(f"{c:>3}{fcs[c]:>7.0f}{float(thr[c]):>8.3f}   (디지털 1.0 에서도 못 닿음)")
            rows.append(None)
            continue
        A = a_thr[c] * mv_per_unit
        dv, how = swing_at(A, s1[c], s4[c])
        vthr = dc[c][1] + dv
        notes = [how] if how else []
        if (how == "lin<1mV" and c in up1) or (how == "" and (c in up1 or c in up4)):
            notes.append("스윙 상한(≤)")
        if dv < VOS_MV:
            notes.append(f"<{VOS_MV:g}mV 오프셋")
        rows.append(vthr)
        print(f"{c:>3}{fcs[c]:>7.0f}{float(thr[c]):>8.3f}{a_thr[c]:>9.5f}{A:>8.3f}m"
              f"{dv:>7.2f}m{dc[c][1]:>8.1f}{vthr:>8.1f}m  {', '.join(notes)}")
    ok = [v for v in rows if v is not None]
    if ok:
        print(f"\n  V_thr 범위 {min(ok):.1f} .. {max(ok):.1f} mV  (가변저항 900..990 mV)")

    if args.wav:
        w0, _ = load_wav(args.wav[0])
        with torch.no_grad():
            bits = (afe(torch.from_numpy(w0).unsqueeze(0)) > 0)[0].numpy()  # [16, native_T]
        print(f"\n  모델이 기대하는 비교기 출력: {args.wav[0].name}  "
              f"(10 ms 칸, # = 켜짐, 위가 ch15)")
        for c in range(15, -1, -1):
            print(f"  ch{c:>2} {''.join('#' if b else '.' for b in bits[c])} "
                  f"{int(bits[c].sum()):>3}")

    if args.wav and len(args.wav) > 1:
        print(f"\n  같은 V_thr 로 여러 클립을 돌리려면 '디지털 1 = {mv_per_unit:.1f} mV' 를 "
              f"유지하는 Vpp 로 넣는다:")
        for p, pk in zip(args.wav, peaks):
            print(f"    {p.name:40s} peak {pk:.4f}  ->  {2 * pk * mv_per_unit:6.1f} mVpp "
                  f"(peak 기준 근사)")


if __name__ == "__main__":
    main()
