"""왜 fixed_scale_quantile 이 1.0 이면 안 되는가 — 그림.

`normalize="fixed"` 는 데이터셋 상수 lo/hi 로 (env-lo)/(hi-lo) 를 한다. hi 를
**전역 max** 로 잡으면(q=1.0) 그건 제일 시끄러운 클립 하나이고, 나머지 전부가
[0,1] 의 얇은 조각으로 눌린다. 학습되는 임계값이 그 조각 안에 앉으므로 Adam 한
스텝이 임계값 대비 큰 비율로 움직여 학습이 요동친다.

두 종류의 숫자가 섞여 있다. 섞어 읽지 말 것:
  * 위 두 패널 = **메커니즘**. 실제 AFEFrontend.init_fixed_scale() 을 돌린 결과지만
    입력이 합성 파형이다 (이 머신에 데이터셋이 없다 -- 학습은 원격 RTX 박스에서 돈다).
    보여주는 것은 "q 가 hi 를 어디에 놓고 그게 분포를 어떻게 누르는가"라는 구조이며,
    이 구조는 입력이 무엇이든 성립한다.
  * 아래 두 패널 = **실측**. Speech Commands v2 로 실제 학습해서 나온 값.

  .venv/bin/python experiments/plot_fixed_scale_quantile.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm                              # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.afe import AFEFrontend                                  # noqa: E402
from train.config import load_config                              # noqa: E402

_HAVE = {f.name for f in fm.fontManager.ttflist}
for _f in ("AppleGothic", "Apple SD Gothic Neo", "Nanum Gothic",
           "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"):
    if _f in _HAVE:
        matplotlib.rcParams["font.family"] = _f
        break
matplotlib.rcParams["axes.unicode_minus"] = False


def plain_log(ax, which: str = "y") -> None:
    """로그축 틱을 평문으로. 한글 폰트에 U+2212 가 없어서 mathtext 지수(10^-2)가
    '10¤2' 로 깨진다."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{v:g}")
    (ax.yaxis if which == "y" else ax.xaxis).set_major_formatter(f)
    (ax.yaxis if which == "y" else ax.xaxis).set_minor_formatter(
        FuncFormatter(lambda v, _: ""))

QS = (0.6, 0.75, 1.0)
COL = {0.6: "#7f8c8d", 0.75: "#2e7d32", 1.0: "#c0392b"}

# --- 실측 (Speech Commands v2, 원격 학습) ------------------------------------
# q 스윕: 2026-08-05 Stage 3. spice/sqrt 이전 전단이라 절대값은 낡았고 ORDER 만 쓴다.
SWEEP = {0.6: 0.653, 0.75: 0.726, 1.0: 0.693}
# 현행 전단(spice+sqrt+fixed)에서 q=1.0 을 쓴 런. 죽은 비교기 4개.
FX_G12 = 0.5763
# data/afe.py init_fixed_scale() 주석에 기록된 측정치.
ENV_STD = {"q=1.0 (전역 max)": 0.034, "클립별 min-max": 0.138}


def clips(n: int = 512, sr: int = 16000) -> torch.Tensor:
    """합성 클립. 음성처럼 클립마다 음량이 크게 다르고 꼬리가 길게."""
    g = torch.Generator().manual_seed(0)
    # 로그정규 음량 + 소수의 아주 큰 클립: 전역 max 가 이상치가 되는 상황을 만든다
    amp = torch.exp(torch.randn(n, 1, generator=g) * 0.55 - 3.0)
    amp[torch.randperm(n, generator=g)[:3]] *= 6.0        # 이상치 3개
    return amp * torch.randn(n, sr, generator=g)


def main() -> int:
    cfg = load_config(str(ROOT / "configs" / "base.yaml"), {
        "afe.spice_matrix_path":
            "analog/AFE/artifacts/filterbank_matrix_board.csv"})
    w = clips()

    # 원시 엔벨로프는 q 와 무관하다 -> 한 번만 계산해서 클립별 max 를 얻는다
    probe = AFEFrontend(copy.deepcopy(cfg.afe))
    with torch.no_grad():
        raw = probe.envelopes(w, raw=True)                 # [B, C, T]
    per_clip_max = raw.amax(dim=(1, 2)).numpy()

    # 실제 코드 경로로 q 별 hi 와 정규화 결과를 얻는다
    res = {}
    for q in QS:
        c = copy.deepcopy(cfg.afe)
        c.fixed_scale_quantile = q
        fe = AFEFrontend(c)
        fe.init_fixed_scale(w)
        fe.init_thresholds(w)
        with torch.no_grad():
            env = fe.envelopes(w)
        res[q] = dict(hi=float(fe.fixed_hi), env=env.flatten().numpy(),
                      thr=fe.threshold.detach().numpy())

    fig = plt.figure(figsize=(13.5, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24)

    # --- 1. q 가 hi 를 어디에 놓는가 -----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(per_clip_max, bins=60, color="#b0bec5", edgecolor="none")
    for q in QS:
        ax.axvline(res[q]["hi"], color=COL[q], lw=2,
                   label=f"q={q}  hi={res[q]['hi']:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("클립별 엔벨로프 최대값")
    ax.set_ylabel("클립 수 (log)")
    ax.set_title("① q 는 hi 를 어디에 놓는가\n"
                 "q=1.0 은 제일 시끄러운 클립 하나를 집는다",
                 fontsize=11, fontweight="bold", loc="left")
    ax.title.set_linespacing(1.6)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, lw=0.5)
    plain_log(ax)
    ax.annotate("이상치", xy=(per_clip_max.max(), 1.4),
                xytext=(per_clip_max.max() * 0.62, 14), fontsize=9,
                color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

    # --- 2. 그 결과 엔벨로프가 얼마나 눌리는가 --------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for q in QS:
        e = res[q]["env"]
        ax.hist(e, bins=np.linspace(0, 1.25, 110), histtype="step", lw=2,
                color=COL[q], density=True,
                label=f"q={q}   std {e.std():.3f}")
    ax.set_xlim(0, 1.25)
    ax.axvline(1.0, color="#455a64", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("정규화된 엔벨로프  (env - lo) / (hi - lo)")
    ax.set_ylabel("밀도 (log)")
    ax.set_title("② 정규화 후 분포 (합성)\n"
                 "q=1.0 이면 전형적 클립이 [0,1] 왼쪽 조각에 몰린다",
                 fontsize=11, fontweight="bold", loc="left")
    ax.title.set_linespacing(1.6)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.25, lw=0.5)
    plain_log(ax)
    ax.text(0.02, 0.04,
            "점선 오른쪽(>1.0) = 잘림. q<1 이 조건수를 얻는 대가다.\n"
            "실측 대조: q=1.0 std 0.034 / 클립별 min-max 0.138",
            transform=ax.transAxes, fontsize=8, color="#555", linespacing=1.5)

    # --- 3. 임계값이 어디 앉고, 한 스텝이 몇 %인가 (실측) ---------------------
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(16)
    for q in (1.0, 0.75):
        ax.plot(x, res[q]["thr"], "o-", color=COL[q], ms=4, lw=1.5,
                label=f"q={q}")
    ax.set_xticks(x)
    ax.set_xlabel("channel")
    ax.set_ylabel("초기 임계값 (정규화 스케일)")
    ax.set_yscale("log")
    ax.set_title("③ 임계값이 앉는 자리 (합성)\n"
                 "q=1.0 이면 두 자릿수 아래로 내려간다",
                 fontsize=11, fontweight="bold", loc="left")
    ax.title.set_linespacing(1.6)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    plain_log(ax)
    ax.text(0.98, 0.05,
            "실측: q=1.0 에서 임계값이 0.002~0.025 에 앉아\n"
            "Adam 한 스텝(lr 1e-3)이 약 15% 를 움직인다 (q=0.75 는 약 3%)",
            transform=ax.transAxes, fontsize=8.5, color="#c0392b", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#c0392b",
                      alpha=0.9, lw=0.8))

    # --- 4. 실측 정확도 -------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ks = list(SWEEP)
    bars = ax.bar([f"q={k}" for k in ks], [SWEEP[k] for k in ks],
                  color=[COL[k] for k in ks], width=0.5)
    for b, k in zip(bars, ks):
        ax.text(b.get_x() + b.get_width() / 2, SWEEP[k] + 0.004,
                f"{SWEEP[k]:.3f}", ha="center", fontsize=9.5,
                fontweight="bold")
    ax.bar(["q=1.0\n(현행 전단)"], [FX_G12], color="#c0392b", width=0.5,
           hatch="//", edgecolor="white")
    ax.text(3, FX_G12 + 0.004, f"{FX_G12:.3f}\n죽은 비교기 4개", ha="center",
            fontsize=9, fontweight="bold", color="#c0392b")
    ax.set_ylim(0.5, 0.78)
    ax.set_ylabel("test accuracy")
    ax.set_title("④ 실측 (Speech Commands v2)\n"
                 "왼쪽 셋은 2026-08-05 전단이라 절대값은 낡았다 — 순서만 본다",
                 fontsize=11, fontweight="bold", loc="left")
    ax.title.set_linespacing(1.6)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    fig.suptitle("fixed_scale_quantile — 왜 0.75 이고 1.0 이 아닌가",
                 fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.5, 0.005,
             "①② 는 실제 init_fixed_scale() 코드를 합성 파형에 돌린 메커니즘 시연이고, "
             "③의 인용 수치와 ④ 는 Speech Commands v2 실측이다.",
             ha="center", fontsize=8.5, color="#555")

    out = ROOT / "docs" / "figures" / "fixed_scale_quantile.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"저장: {out.relative_to(ROOT)}")
    for q in QS:
        e = res[q]["env"]
        print(f"  q={q:<5} hi={res[q]['hi']:.4f}  env std={e.std():.4f}  "
              f"thr {res[q]['thr'].min():.4f}~{res[q]['thr'].max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
