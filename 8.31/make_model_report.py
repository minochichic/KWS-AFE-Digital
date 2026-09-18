"""학습 모델 설명 문서를 만든다 — 실제로 학습을 돌린 결과에서.

숫자를 손으로 옮겨 적지 않는다. 여기서 학습·측정한 값이 그대로 HTML 로 간다.

  python make_model_report.py          ->  학습모델.html      (8.31/ 에서)
"""
from __future__ import annotations

import html
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wakeup.config import Config
from wakeup.model import WakeupModel, detect_start, gather_states
from wakeup.search import search_timing, offset_scores
from wakeup.synthetic import make_batch, TARGET_HZ, DISTRACTOR_HZ
from wakeup.export import build_report

OUT = Path(__file__).parent / "학습모델.html"


# ════════════════════════════════════════════════════════ 학습
def fit(seed=0, steps=800, n=768, search=True, ste_clip=0.03, l1=0.0,
        min_ch=0, collect=False):
    torch.manual_seed(seed)
    cfg = Config()
    cfg.head.temperature = cfg.head.and_temperature = 0.7
    cfg.head.l1_gate, cfg.head.min_channels = l1, min_ch
    cfg.frontend.ste_clip = ste_clip
    cfg.train.target_word = "합성 대상"
    m = WakeupModel(cfg)
    xtr, ytr, on_tr = make_batch(n, cfg.head.tau, seed=seed)
    xte, yte, on_te = make_batch(256, cfg.head.tau, seed=seed + 999)
    m.frontend.init_fixed_scale(xtr[:256])
    m.frontend.init_thresholds(xtr[:256])

    info = {}
    if collect:
        with torch.no_grad():
            x0 = m.features(xtr)
            s0, _ = detect_start(x0, cfg.start.k)
            info["auc"] = offset_scores(x0, s0, ytr, 60)[0].tolist()
            info["start_err"] = (s0 - on_tr).tolist()
    if search:
        search_timing(m, xtr, ytr)
        if collect:
            info["tau"] = list(cfg.head.tau)

    opt = torch.optim.Adam([{"params": m.head.parameters(), "lr": 3e-2},
                            {"params": m.frontend.parameters(), "lr": 3e-3}])
    m.train()
    for i in range(steps):
        j = torch.randint(0, n - 64, (1,)).item()
        o = m.loss(xtr[j:j + 64], ytr[j:j + 64])
        opt.zero_grad(); o["loss"].backward(); opt.step()
        if (i + 1) % 50 == 0:
            m.refit_k(xtr[:256], ytr[:256])

    m.eval()
    wk = m.hard(xte)["wake"]
    res = {"acc": (wk == yte).float().mean().item(),
           "tpr": wk[yte == 1].mean().item(),
           "fpr": wk[yte == 0].mean().item()}
    if collect:
        with torch.no_grad():
            xf = m.features(xte)
            st, _ = detect_start(xf, m.cfg.start.k)
            xs = gather_states(xf, st, m.tau)
            info["count"] = m.head(xs)["count"].tolist()
            info["y"] = yte.tolist()
            ip = int((yte == 1).nonzero()[0]); ineg = int((yte == 0).nonzero()[0])
            info["raster_pos"] = xf[ip].int().tolist()
            info["raster_neg"] = xf[ineg].int().tolist()
            info["start_pos"] = int(st[ip]); info["start_neg"] = int(st[ineg])
            info["on_rate"] = xf.mean().item()
    return m, res, info


# ════════════════════════════════════════════════════════ SVG
K, HOT, COOL = "var(--stroke)", "var(--hot)", "var(--cool)"


def svg_raster(rows, start, tau, title, w=740, h=190):
    """[C, T] 이진 이미지 + START/tau 표시."""
    C, T = len(rows), len(rows[0])
    x0, y0, pw = 52, 26, (w - 70) / T
    ph = (h - 60) / C
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
         f'aria-label="{html.escape(title)}">']
    for c in range(C):
        p.append(f'<text x="{x0-8}" y="{y0+c*ph+ph*0.72}" font-size="9" '
                 f'fill="var(--faint)" text-anchor="end" class="m">ch{c}</text>')
        for t in range(T):
            if rows[c][t]:
                p.append(f'<rect x="{x0+t*pw:.1f}" y="{y0+c*ph:.1f}" '
                         f'width="{pw:.2f}" height="{ph-1:.1f}" fill="#2F3A46"/>')
    p.append(f'<rect x="{x0}" y="{y0}" width="{T*pw:.1f}" height="{C*ph:.1f}" '
             f'fill="none" stroke="var(--rule)" stroke-width="1"/>')
    yb = y0 + C * ph
    p.append(f'<line x1="{x0+start*pw:.1f}" y1="{y0-6}" x2="{x0+start*pw:.1f}" '
             f'y2="{yb+6}" stroke="{HOT}" stroke-width="1.4"/>')
    p.append(f'<text x="{x0+start*pw:.1f}" y="{y0-10}" font-size="9.5" fill="{HOT}" '
             f'text-anchor="middle" class="m">START</text>')
    for i, d in enumerate(tau):
        xx = x0 + (start + d) * pw
        p.append(f'<line x1="{xx:.1f}" y1="{y0-2}" x2="{xx:.1f}" y2="{yb+4}" '
                 f'stroke="{COOL}" stroke-width="1.1" stroke-dasharray="3 2"/>')
        p.append(f'<text x="{xx:.1f}" y="{yb+17}" font-size="9" fill="{COOL}" '
                 f'text-anchor="middle" class="m">τ{i+1}</text>')
    p.append(f'<text x="{x0}" y="{h-4}" font-size="9" fill="var(--faint)">0 ms</text>')
    p.append(f'<text x="{x0+T*pw:.0f}" y="{h-4}" font-size="9" fill="var(--faint)" '
             f'text-anchor="end">1000 ms</text>')
    return "\n".join(p) + "</svg>"


def svg_auc(auc, tau, w=740, h=200):
    n = len(auc)
    x0, y0, pw, ph = 46, 20, (w - 66) / (n - 1), h - 58
    def Y(v): return y0 + ph * (1 - (v - 0.5) / 0.5)
    pts = " ".join(f"{x0+i*pw:.1f},{Y(v):.1f}" for i, v in enumerate(auc))
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
         f'aria-label="START 이후 오프셋별 분리도">']
    for v in (0.5, 0.75, 1.0):
        p.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x0+(n-1)*pw:.1f}" '
                 f'y2="{Y(v):.1f}" stroke="var(--rule-soft)" stroke-width="1"/>')
        p.append(f'<text x="{x0-6}" y="{Y(v)+3:.1f}" font-size="9" '
                 f'fill="var(--faint)" text-anchor="end" class="m">{v:.2f}</text>')
    for d in tau:
        p.append(f'<line x1="{x0+d*pw:.1f}" y1="{y0}" x2="{x0+d*pw:.1f}" '
                 f'y2="{y0+ph}" stroke="{HOT}" stroke-width="1.2" '
                 f'stroke-dasharray="3 3"/>')
        p.append(f'<text x="{x0+d*pw:.1f}" y="{y0+ph+15}" font-size="9.5" '
                 f'fill="{HOT}" text-anchor="middle" class="m">{d}</text>')
    p.append(f'<polyline points="{pts}" fill="none" stroke="{COOL}" stroke-width="1.6"/>')
    p.append(f'<text x="{x0}" y="{h-6}" font-size="9.5" fill="var(--faint)">'
             f'START 이후 오프셋 [프레임]</text>')
    p.append(f'<text x="{x0-6}" y="{y0-6}" font-size="9.5" fill="var(--faint)" '
             f'text-anchor="end" class="m">AUC</text>')
    return "\n".join(p) + "</svg>"


def svg_counts(count, y, k, C, w=740, h=210):
    """상태별 일치 개수 분포 + k 위치."""
    S = len(k)
    cw = (w - 40) / S
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
         f'aria-label="상태별 일치 개수 분포와 허용 오차">']
    for s in range(S):
        bx = 24 + s * cw
        pos = [c[s] for c, yy in zip(count, y) if yy > 0.5]
        neg = [c[s] for c, yy in zip(count, y) if yy <= 0.5]
        hp = [sum(1 for v in pos if round(v) == i) for i in range(C + 1)]
        hn = [sum(1 for v in neg if round(v) == i) for i in range(C + 1)]
        mx = max(max(hp), max(hn), 1)
        bw = (cw - 34) / (C + 1)
        base, hh = h - 40, h - 82
        for i in range(C + 1):
            for vals, col, off in ((hn, "var(--faint)", 0), (hp, COOL, bw * 0.45)):
                bh = hh * vals[i] / mx
                p.append(f'<rect x="{bx+16+i*bw+off:.1f}" y="{base-bh:.1f}" '
                         f'width="{bw*0.42:.1f}" height="{bh:.1f}" fill="{col}"/>')
        kx = bx + 16 + (k[s] - 0.5) * bw + bw * 0.45
        p.append(f'<line x1="{kx:.1f}" y1="{base-hh-6:.1f}" x2="{kx:.1f}" '
                 f'y2="{base+4}" stroke="{HOT}" stroke-width="1.6"/>')
        p.append(f'<text x="{kx:.1f}" y="{base-hh-10:.1f}" font-size="9.5" '
                 f'fill="{HOT}" text-anchor="middle" class="m">k={k[s]}</text>')
        p.append(f'<text x="{bx+cw/2:.1f}" y="{h-6}" font-size="10" '
                 f'fill="var(--ink)" text-anchor="middle">상태 {s+1}</text>')
        p.append(f'<line x1="{bx+16:.1f}" y1="{base}" x2="{bx+cw-16:.1f}" '
                 f'y2="{base}" stroke="var(--rule)" stroke-width="1"/>')
    p.append(f'<text x="{w-24}" y="14" font-size="9.5" text-anchor="end" '
             f'fill="{COOL}">■ 양성</text>')
    p.append(f'<text x="{w-84}" y="14" font-size="9.5" text-anchor="end" '
             f'fill="var(--faint)">■ 음성</text>')
    return "\n".join(p) + "</svg>"


def svg_startdist(errs, w=740, h=140):
    lo, hi = min(errs), max(errs)
    vals = list(range(lo, hi + 1))
    cnt = [errs.count(v) for v in vals]
    mx = max(cnt)
    bw = min(56, (w - 90) / max(len(vals), 1))
    x0, base, hh = 60, h - 34, h - 62
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
         f'aria-label="START 검출 오차 분포">']
    for i, (v, c) in enumerate(zip(vals, cnt)):
        bh = hh * c / mx
        p.append(f'<rect x="{x0+i*bw:.1f}" y="{base-bh:.1f}" width="{bw*0.72:.1f}" '
                 f'height="{bh:.1f}" fill="{COOL}"/>')
        p.append(f'<text x="{x0+i*bw+bw*0.36:.1f}" y="{base+14}" font-size="10" '
                 f'fill="var(--ink)" text-anchor="middle" class="m">{v:+d}</text>')
        p.append(f'<text x="{x0+i*bw+bw*0.36:.1f}" y="{base-bh-5:.1f}" '
                 f'font-size="9.5" fill="var(--muted)" text-anchor="middle">'
                 f'{100*c/len(errs):.0f}%</text>')
    p.append(f'<line x1="{x0-6}" y1="{base}" x2="{x0+len(vals)*bw:.1f}" y2="{base}" '
             f'stroke="var(--rule)" stroke-width="1"/>')
    p.append(f'<text x="{x0-12}" y="{base+14}" font-size="9.5" fill="var(--faint)" '
             f'text-anchor="end">프레임</text>')
    return "\n".join(p) + "</svg>"


def svg_pipeline(w=740, h=132):
    boxes = [("Binary feature", "X[c,t] ∈ {0,1}", "고정"),
             ("START + Align", "시간 원점", "탐색"),
             ("Template Head", "M(s,c), k(s)", "학습"),
             ("AND", "WAKE", "고정")]
    bw, gap = 148, 40
    x = 24
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
         f'aria-label="학습 모델 파이프라인">']
    for i, (nm, sub, kind) in enumerate(boxes):
        col = HOT if kind in ("학습", "탐색") else "var(--fill)"
        st = HOT if kind in ("학습", "탐색") else K
        fill = "var(--hot-bg)" if kind in ("학습", "탐색") else "var(--fill)"
        p.append(f'<rect x="{x}" y="34" width="{bw}" height="54" rx="4" '
                 f'fill="{fill}" stroke="{st}" stroke-width="1.3"/>')
        p.append(f'<text x="{x+bw/2}" y="56" font-size="12" fill="var(--ink)" '
                 f'text-anchor="middle">{nm}</text>')
        p.append(f'<text x="{x+bw/2}" y="72" font-size="9.5" fill="var(--muted)" '
                 f'text-anchor="middle" class="m">{sub}</text>')
        p.append(f'<text x="{x+bw/2}" y="24" font-size="9.5" fill="{st}" '
                 f'text-anchor="middle">{kind}</text>')
        if i < len(boxes) - 1:
            p.append(f'<line x1="{x+bw}" y1="61" x2="{x+bw+gap-8}" y2="61" '
                     f'stroke="{K}" stroke-width="1.2"/>')
            p.append(f'<path d="M {x+bw+gap-8} 57 L {x+bw+gap-2} 61 '
                     f'L {x+bw+gap-8} 65 z" fill="{K}"/>')
        x += bw + gap
    p.append(f'<text x="22" y="112" font-size="10" fill="var(--faint)">'
             f'회로와 같은 순서로 계산한다 — 학습이 끝나면 각 상자가 부품이 된다</text>')
    return "\n".join(p) + "</svg>"


# ════════════════════════════════════════════════════════ 측정
CACHE = Path(__file__).parent / "runs" / "_report_cache.json"


def measure() -> dict:
    """학습을 실제로 돌려 문서에 들어갈 값을 전부 잰다."""
    t0 = time.time()
    print("상세 모델 학습 …")
    m, res, info = fit(seed=0, l1=0.05, min_ch=3, collect=True)
    rep = build_report(m)

    print("5시드 통계 …")
    multi = [fit(seed=s)[1] for s in range(5)]
    acc = [r["acc"] for r in multi]
    tpr = [r["tpr"] for r in multi]
    fpr = [r["fpr"] for r in multi]

    print("절제: tau 탐색 …")
    ab_tau = [fit(seed=s, search=False, steps=400)[1] for s in range(3)]
    ab_tau_on = [fit(seed=s, search=True, steps=400)[1] for s in range(3)]

    print("절제: ste_clip …")
    ste = {c: fit(seed=0, ste_clip=c)[1] for c in (0.003, 0.03, 0.1)}

    print("절제: L1 …")
    l1s = {}
    for lam in (0.0, 0.05, 0.2):
        mm, rr, _ = fit(seed=0, l1=lam, min_ch=3)
        l1s[lam] = (rr, mm.export()["n_resistors"])

    print(f"측정 완료 {time.time()-t0:.0f}s")
    return {
        "rep": rep, "info": info,
        "multi": multi,
        "ab_tau": ab_tau, "ab_tau_on": ab_tau_on,
        "ste": {str(k): v for k, v in ste.items()},
        "l1": {str(k): [v[0], v[1]] for k, v in l1s.items()},
    }


# ════════════════════════════════════════════════════════ 렌더
def build(d: dict) -> str:
    rep, info = d["rep"], d["info"]
    multi = d["multi"]
    acc = [r["acc"] for r in multi]
    tpr = [r["tpr"] for r in multi]
    fpr = [r["fpr"] for r in multi]
    ab_tau, ab_tau_on = d["ab_tau"], d["ab_tau_on"]
    ste = {float(k): v for k, v in d["ste"].items()}
    l1s = {float(k): tuple(v) for k, v in d["l1"].items()}

    S = rep["n_states"]; C = rep["n_channels"]
    kk = [s["k"] for s in rep["states"]]
    tau = rep_tau = [s["tau_frames"] for s in rep["states"]]

    def tmpl_cell(v):
        if v > 0:
            return '<td class="t1">1</td>'
        if v < 0:
            return '<td class="t0">0</td>'
        return '<td class="tx">X</td>'

    tmpl_rows = "\n".join(
        f'<tr><th>상태 {s["state"]}</th><td class="mono">{s["tau_ms"]:.0f} ms</td>'
        + "".join(tmpl_cell(v) for v in s["template"])
        + f'<td class="mono">{s["m"]}</td><td class="mono">{s["k"]}</td></tr>'
        for s in rep["states"])

    circ_rows = "\n".join(
        f'<tr><td>s{s["state"]}</td><td class="mono">{s["m"]}</td>'
        f'<td class="mono">{s["k"]}</td><td class="mono">{s["V_TH"]:.3f}</td>'
        f'<td class="mono">{s["margin_V"]:.3f}</td>'
        f'<td class="mono">{s["R7_ohm"]/1000:.0f}</td>'
        f'<td class="mono">{s["R8_ohm"]/1000:.0f}</td></tr>'
        for s in rep["states"])

    def mstd(v):
        return f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}"

    tau_off = statistics.mean([r["tpr"] for r in ab_tau])
    tau_on = statistics.mean([r["tpr"] for r in ab_tau_on])
    errs = info["start_err"]
    med = statistics.median(errs)

    return TEMPLATE.format(
        pipeline=svg_pipeline(),
        raster_pos=svg_raster(info["raster_pos"], info["start_pos"], tau,
                              "양성 클립의 이진 특징"),
        raster_neg=svg_raster(info["raster_neg"], info["start_neg"], tau,
                              "음성 클립의 이진 특징"),
        on_rate=f"{info['on_rate']*100:.1f}",
        startdist=svg_startdist(errs),
        start_med=f"{med:+.0f}",
        start_sd=f"{statistics.pstdev(errs):.2f}",
        start_pm1=f"{100*sum(1 for e in errs if abs(e-med)<=1)/len(errs):.0f}",
        auc=svg_auc(info["auc"], tau),
        tau=", ".join(str(t) for t in tau),
        tau_ms=", ".join(f"{t*10}" for t in tau),
        start_k=rep["start_k"],
        counts=svg_counts(info["count"], info["y"], kk, C),
        n_states=S, n_channels=C,
        chan_head="".join(f"<th>ch{c}</th>" for c in range(C)),
        tmpl_rows=tmpl_rows,
        n_res=rep["n_resistors_template"], n_res_full=C * S, n_div=2 * S,
        circ_rows=circ_rows,
        timeout=rep["timeout_frames"], timeout_ms=f"{rep['timeout_ms']:.0f}",
        wake_ms=f"{rep['wake_width_ms']:.0f}",
        acc=mstd(acc), tpr=mstd(tpr), fpr=mstd(fpr),
        acc_min=f"{min(acc):.3f}",
        tau_off=f"{tau_off:.3f}", tau_on=f"{tau_on:.3f}",
        ste_a=f"{ste[0.003]['acc']:.3f}", ste_b=f"{ste[0.03]['acc']:.3f}",
        ste_c=f"{ste[0.1]['acc']:.3f}",
        l1_rows="\n".join(
            f'<tr><td class="mono">{lam}</td>'
            f'<td class="mono">{v[1]}</td>'
            f'<td class="mono">{v[0]["tpr"]:.3f}</td>'
            f'<td class="mono">{v[0]["fpr"]:.3f}</td></tr>'
            for lam, v in l1s.items()),
        target_hz=", ".join(f"{f:.0f}" for f in TARGET_HZ),
        distract_hz=", ".join(f"{f:.0f}" for f in DISTRACTOR_HZ),
        stamp=time.strftime("%Y-%m-%d"),
    )


TEMPLATE = """<title>학습 모델 — 무엇을 학습하고 무엇이 회로가 되는가</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@300;400;500&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{{
    --paper:#F7F8FA; --card:#FFFFFF; --sunk:#EDF0F4;
    --ink:#10161F; --muted:#59636F; --faint:#8A95A2;
    --rule:#D3DAE2; --rule-soft:#E7ECF1;
    --hot:#A9571F; --hot-bg:#F7EDE4;
    --cool:#25646F; --cool-bg:#E4EEF0;
    --stroke:#10161F; --fill:#F0F2F5;
  }}
  @media (prefers-color-scheme: dark){{
    :root:not([data-theme="light"]){{
      --paper:#0D1218; --card:#141B23; --sunk:#111820;
      --ink:#E8EDF3; --muted:#9CA8B5; --faint:#6E7B89;
      --rule:#28323D; --rule-soft:#1C242D;
      --hot:#E09456; --hot-bg:#2A1D12;
      --cool:#6FBECC; --cool-bg:#0F2429;
      --stroke:#D8DEE6; --fill:#1B232C;
    }}
  }}
  :root[data-theme="dark"]{{
    --paper:#0D1218; --card:#141B23; --sunk:#111820;
    --ink:#E8EDF3; --muted:#9CA8B5; --faint:#6E7B89;
    --rule:#28323D; --rule-soft:#1C242D;
    --hot:#E09456; --hot-bg:#2A1D12;
    --cool:#6FBECC; --cool-bg:#0F2429;
    --stroke:#D8DEE6; --fill:#1B232C;
  }}
  body{{background:var(--paper); color:var(--ink); margin:0;
       font-family:"IBM Plex Sans KR",-apple-system,sans-serif; font-weight:300;
       line-height:1.75; font-size:15.5px}}
  .wrap{{max-width:860px; margin:0 auto; padding:54px 26px 90px}}
  header{{margin-bottom:46px}}
  .eyebrow{{font-family:"IBM Plex Mono",monospace; font-size:11px;
           letter-spacing:.14em; text-transform:uppercase; color:var(--faint)}}
  h1{{font-size:31px; line-height:1.32; margin:12px 0 14px; font-weight:400}}
  .lede{{color:var(--muted); font-size:16.5px; max-width:62ch; margin:0}}
  section{{margin-top:52px}}
  .kicker{{font-family:"IBM Plex Mono",monospace; font-size:11px;
          letter-spacing:.14em; color:var(--faint); margin-bottom:8px}}
  h2{{font-size:23px; margin:0 0 14px; font-weight:400}}
  h3{{font-size:17px; margin:28px 0 9px; font-weight:500}}
  p{{margin:0 0 14px}}
  code{{font-family:"IBM Plex Mono",monospace; font-size:13px;
       background:var(--sunk); padding:1px 5px; border-radius:3px}}
  .mono{{font-family:"IBM Plex Mono",monospace; font-size:13px}}
  figure{{margin:22px 0 24px}}
  .stage{{background:var(--card); border:1px solid var(--rule-soft);
         border-radius:3px; padding:16px 14px 12px; overflow-x:auto}}
  figcaption{{margin-top:9px; font-size:13px; color:var(--muted);
             line-height:1.66; max-width:66ch}}
  table{{width:100%; border-collapse:collapse; margin:18px 0; font-size:14px}}
  th{{text-align:left; font-family:"IBM Plex Mono",monospace; font-weight:500;
     font-size:10.5px; letter-spacing:.09em; text-transform:uppercase;
     color:var(--faint); padding:0 9px 8px 0; border-bottom:1px solid var(--rule)}}
  td{{padding:8px 9px 8px 0; border-bottom:1px solid var(--rule-soft);
     vertical-align:top}}
  tbody th{{text-transform:none; letter-spacing:0; font-size:13.5px;
           color:var(--ink); border-bottom:1px solid var(--rule-soft);
           padding:8px 9px 8px 0}}
  .tm td{{text-align:center; font-family:"IBM Plex Mono",monospace}}
  .t1{{color:var(--ink); font-weight:500; background:var(--cool-bg)}}
  .t0{{color:var(--muted)}}
  .tx{{color:var(--faint); background:var(--sunk)}}
  .note{{background:var(--cool-bg); border-left:2px solid var(--cool);
        padding:14px 17px; margin:20px 0; font-size:14.5px; line-height:1.72}}
  .warn{{background:var(--hot-bg); border-left-color:var(--hot)}}
  .big{{display:flex; gap:26px; flex-wrap:wrap; margin:22px 0}}
  .big div{{flex:1; min-width:130px}}
  .big .v{{font-family:"IBM Plex Mono",monospace; font-size:26px;
          color:var(--ink); line-height:1.2}}
  .big .l{{font-size:12px; color:var(--faint)}}
  svg text{{font-family:"IBM Plex Sans KR",sans-serif; font-weight:300}}
  svg .m{{font-family:"IBM Plex Mono",monospace}}
  footer{{margin-top:60px; padding-top:20px; border-top:1px solid var(--rule);
         font-size:12.5px; color:var(--faint)}}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">IT창의챌린지 · 디지털 판별부</div>
  <h1>학습 모델 —<br>무엇을 학습하고, 무엇이 회로가 되는가</h1>
  <p class="lede">
    형판·허용 오차·시각을 데이터에서 뽑아 저항값과 배선으로 바꾸는 과정.
    이 문서의 모든 숫자는 실제로 학습을 돌려 측정한 값이다.
  </p>
</header>

<section>
  <div class="kicker">1</div>
  <h2>구조</h2>
  <p>모델은 회로와 <b>같은 순서로</b> 계산한다. 그래야 학습에서 얻은 정확도가
  하드웨어로 그대로 넘어간다. 네 단계 중 학습되는 것은 형판뿐이고, 시각은
  탐색으로, 나머지는 고정이다.</p>
  <figure><div class="stage">{pipeline}</div></figure>

  <table>
    <tr><th>단계</th><th>정하는 방법</th><th>회로에서는</th></tr>
    <tr><td>Binary feature</td><td>고정 (문턱 θ만 학습)</td><td>비교기 출력</td></tr>
    <tr><td>START + Align</td><td>탐색 — 정수라 미분 안 됨</td><td>START detect 게이트</td></tr>
    <tr><td>Template Head</td><td>M은 경사하강, k는 좌표상승</td><td>저항망 + 비교기</td></tr>
    <tr><td>AND</td><td>고정</td><td>2입력 AND ×3</td></tr>
  </table>

  <div class="note">
    <b>검증용 과제.</b> 실제 음성을 붙이기 전에, 발성 시작으로부터 정해진 네
    시각에 정해진 주파수({target_hz} Hz)가 오는 합성 파형으로 확인했다. 음성은
    같은 구조에 주파수만 다르다({distract_hz} Hz 중 무작위). 발성 시작 시각은
    매번 달라 START 검출과 정렬까지 함께 시험한다.
    <b>이 과제를 못 풀면 실제 데이터로 갈 이유가 없다.</b>
  </div>
</section>

<section>
  <div class="kicker">2</div>
  <h2>입력 — 이진 특징</h2>
  <p>아날로그 전단이 내보낼 0/1 이미지다. 세로가 채널, 가로가 10 ms 시간.
  붉은 선이 검출된 START, 파선이 채점 시각 τ다.</p>

  <figure>
    <div class="stage">{raster_pos}</div>
    <figcaption><b>양성</b> — 네 τ 마다 정해진 채널이 켜져 있다.</figcaption>
  </figure>
  <figure>
    <div class="stage">{raster_neg}</div>
    <figcaption><b>음성</b> — 구조는 같지만 켜지는 채널이 다르다.
    이 차이만으로 갈라내야 한다.</figcaption>
  </figure>
  <p>전체 켜짐률은 <b>{on_rate}%</b>다. 이 값이 너무 높으면 잡음이 형판을 맞히고,
  너무 낮으면 양성도 못 맞힌다. 문턱 θ가 이걸 정한다.</p>
</section>

<section>
  <div class="kicker">3</div>
  <h2>START — 시간 원점</h2>
  <p>형판은 절대 시각이 아니라 발성 시작으로부터의 상대 시각에 정의된다.
  START가 흔들리면 네 상태가 <b>한꺼번에</b> 빗나가므로, 지터는 곧 검출률이다.</p>
  <figure>
    <div class="stage">{startdist}</div>
    <figcaption>검출된 START에서 진짜 발성 시각을 뺀 값.
    중앙 <b>{start_med}</b> 프레임, 표준편차 <b>{start_sd}</b>.</figcaption>
  </figure>
  <p>중앙값이 0이 아닌 것은 문제가 아니다. STFT 창과 포락선 격자 때문에 생기는
  <b>일정한 치우침</b>이라, τ가 그만큼 밀려서 흡수된다. 중요한 것은 흩어짐이고,
  자기 중앙값 기준 ±1 프레임 안에 <b>{start_pm1}%</b>가 들어온다.</p>
</section>

<section>
  <div class="kicker">4</div>
  <h2>τ — 언제 볼 것인가</h2>
  <p>τ는 정수 시각이라 미분되지 않는다. START 이후 모든 오프셋에 대해
  "그 시각의 채널 패턴만으로 양성과 음성이 얼마나 갈리는가"를 재고, 분리도가
  높은 곳을 간격을 두고 고른다.</p>
  <figure>
    <div class="stage">{auc}</div>
    <figcaption>오프셋별 분리도(AUC). 0.5가 무작위, 1.0이 완전 분리.
    붉은 파선이 고른 τ = <b>{tau}</b> 프레임 ({tau_ms} ms).
    START 조건은 채널 {start_k}개 이상 동시 ON.</figcaption>
  </figure>
  <div class="note warn">
    <b>τ를 추측으로 두면 무너진다.</b> 고정 τ에서 검출률 <b>{tau_off}</b>,
    탐색하면 <b>{tau_on}</b>. START가 발성보다 앞서 뜨기 때문에 추측한 τ는
    버스트 시작 직전에 떨어져 계속 빗나간다. 한 클립의 정렬 오차가 네 상태를
    동시에 밀어버리므로 손실이 그대로 검출률이 된다.
  </div>
</section>

<section>
  <div class="kicker">5</div>
  <h2>형판 — 무엇을 볼 것인가</h2>
  <p>상태마다 채널별로 세 값 중 하나를 학습한다. <code>1</code>은 켜져 있어야,
  <code>0</code>은 꺼져 있어야, <code>X</code>는 보지 않는다.
  X는 저항이 아예 빠지므로 부품 수가 줄고 화자 변동에 둔감해진다.</p>

  <table class="tm">
    <tr><th style="text-align:left">상태</th><th style="text-align:left">τ</th>
        {chan_head}<th>m</th><th>k</th></tr>
    {tmpl_rows}
  </table>
  <p>형판 저항 <b>{n_res}개</b> (전 채널을 다 쓰면 {n_res_full}개).
  <code>m</code>이 그 상태가 실제로 보는 채널 수이고, <code>k</code>가 몇 개
  이상 맞으면 통과인지다.</p>
</section>

<section>
  <div class="kicker">6</div>
  <h2>k — 몇 개 맞으면 통과인가</h2>
  <p>일치 개수의 분포다. 파란색이 양성, 회색이 음성. 붉은 선이 고른 k.</p>
  <figure>
    <div class="stage">{counts}</div>
    <figcaption>양성과 음성의 분포가 겹치는 구간에서 k를 어디에 두느냐가
    곧 TPR과 FPR의 맞바꿈이다.</figcaption>
  </figure>
  <div class="note warn">
    <b>k는 경사하강으로 배우면 안 된다.</b> 학습 초기에는 형판이 무작위라
    양성이 전부 탈락하고, 그 압력으로 k가 바닥까지 내려간다. 그 뒤에는 양성이
    여유롭게 통과해 버려 다시 올릴 기울기가 사라진다. 실제로 k가 채널 절반
    (우연 수준)에 눌러앉는 것을 확인했다.<br><br>
    그래서 k는 <b>정수 좌표상승으로 탐색</b>한다. 목적함수는
    <b>"FPR을 상한 아래로 묶고 그 안에서 TPR 최대화"</b>다. 흔히 쓰는
    <code>TPR − w·FPR</code> 가중합은 w가 2 이상이면 <b>전부 거부</b>하는 해
    (TPR 0, FPR 0)가 최적이 되어 버린다. 회로에서도 k는 결국 반올림된 정수이므로
    탐색이 정직하다.
  </div>
</section>

<section>
  <div class="kicker">7</div>
  <h2>결과</h2>
  <div class="big">
    <div><div class="v">{acc}</div><div class="l">정확도 (5시드)</div></div>
    <div><div class="v">{tpr}</div><div class="l">검출률 TPR</div></div>
    <div><div class="v">{fpr}</div><div class="l">오검출 FPR</div></div>
  </div>
  <p>최저 시드에서도 {acc_min}. 경성 판정(반올림한 정수 상수로 회로와 똑같이
  계산)으로 측정한 값이다.</p>

  <h3>L1 — 정확도와 부품 수의 맞바꿈</h3>
  <table>
    <tr><th>L1 계수</th><th>형판 저항</th><th>TPR</th><th>FPR</th></tr>
    {l1_rows}
  </table>
  <p>X를 늘리는 압력을 키우면 저항이 줄어든다. 이 곡선 위에서 실장 가능한
  지점을 고르면 된다.</p>

  <h3>문턱 STE 폭</h3>
  <p>이진화의 기울기는 <code>|env − θ|</code>가 clip 안에 있을 때만 흐른다.
  너무 좁으면 문턱이 초기값에 묶인다.</p>
  <table>
    <tr><th>ste_clip</th><th>정확도</th><th>비고</th></tr>
    <tr><td class="mono">0.003</td><td class="mono">{ste_a}</td>
        <td>문턱이 거의 안 움직인다</td></tr>
    <tr><td class="mono">0.03</td><td class="mono">{ste_b}</td><td>채택</td></tr>
    <tr><td class="mono">0.1</td><td class="mono">{ste_c}</td><td></td></tr>
  </table>
  <div class="note">
    <b>처음 잰 값과 다르다.</b> τ 탐색을 넣기 전에는 이 세 값이
    0.855 / 0.918 / 0.680 으로 벌어져 <code>ste_clip</code>이 결정적으로 보였다.
    탐색을 넣고 다시 재니 차이가 위 표처럼 줄었다 — <b>당시 관측한 붕괴의
    상당 부분은 문턱이 아니라 어긋난 정렬 탓</b>이었다. 0.003이 낮은 것은 지금도
    유효하므로 0.03을 유지하되, 실제 음성에서는 포락선 분포가 다르니 다시 재야 한다.
  </div>
</section>

<section>
  <div class="kicker">8</div>
  <h2>회로 상수</h2>
  <p>학습이 내놓는 최종 산출물이다. 이 표가 곧 배선표다.</p>
  <table>
    <tr><th>상태</th><th>m</th><th>k</th><th>V_TH [V]</th><th>여유 [V]</th>
        <th>R7 [kΩ]</th><th>R8 [kΩ]</th></tr>
    {circ_rows}
  </table>
  <p><code>V_TH(s) = VDD × (k − 0.5) / m(s)</code>. m이 상태마다 달라
  기준전압을 공유할 수 없고, 상태마다 분압 저항 두 개가 따로 필요하다.</p>
  <div class="note">
    <b>판정 여유</b>는 한 채널이 뒤집힐 때의 전압 변화의 절반,
    <code>VDD / (2·m)</code>이다. 위 표에서 100 mV를 넘으므로 비교기 오프셋
    (수 mV)에 비해 충분하다. 그리고 <b>m이 작을수록 여유가 커진다</b> — X를
    늘리는 L1이 부품 수뿐 아니라 공차 내성도 같이 개선한다는 뜻이다.
  </div>
  <table>
    <tr><th>항목</th><th>값</th></tr>
    <tr><td>START 조건</td><td class="mono">채널 {start_k}개 이상 동시 ON</td></tr>
    <tr><td>τ(1)…τ({n_states})</td><td class="mono">{tau} 프레임 = {tau_ms} ms</td></tr>
    <tr><td>timeout</td><td class="mono">{timeout} 프레임 = {timeout_ms} ms</td></tr>
    <tr><td>WAKE 폭</td><td class="mono">{wake_ms} ms</td></tr>
    <tr><td>형판 저항</td><td class="mono">{n_res}개</td></tr>
    <tr><td>기준전압 분압 저항</td><td class="mono">{n_states} × 2 = {n_div}개</td></tr>
  </table>
</section>

<section>
  <div class="kicker">9</div>
  <h2>남은 일</h2>
  <p>여기까지는 합성 과제다. 실제 Speech Commands로 옮기면서 다시 재야 하는 것들:</p>
  <table>
    <tr><th>항목</th><th>이유</th></tr>
    <tr><td>ste_clip 재측정</td><td>실제 음성의 포락선 분포가 다르다</td></tr>
    <tr><td>SPICE 필터뱅크로 교체</td><td>동료 회로의 중심주파수·Q·이득 반영</td></tr>
    <tr><td>θ를 VTHR 격자에 양자화</td><td>기준전압이 10 mV 간격으로만 존재</td></tr>
    <tr><td>연속 오디오 FA/h</td><td>지금은 클립 단위 FPR이라 시간당 값이 아니다</td></tr>
    <tr><td>소자 공차 몬테카를로</td><td>저항 1%, 비교기 오프셋에서 판정 여유 확인</td></tr>
    <tr><td>상태 수·채널 수 축소 곡선</td><td>회로 규모 대비 이득</td></tr>
  </table>
</section>

<footer>
  IT창의챌린지 · 학습 모델 · 합성 검증 과제 기준 · {stamp} 측정
</footer>
</div>
"""


if __name__ == "__main__":
    # 레이아웃만 고칠 때 3분짜리 재학습을 반복하지 않도록 측정값을 캐시한다.
    refit = "--refit" in sys.argv or not CACHE.exists()
    if refit:
        d = measure()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(d, ensure_ascii=False))
        print(f"측정값 캐시 -> {CACHE}")
    else:
        d = json.loads(CACHE.read_text())
        print(f"캐시 사용 ({CACHE}). 다시 재려면 --refit")
    OUT.write_text(build(d), encoding="utf-8")
    print(f"-> {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
