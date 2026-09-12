#!/usr/bin/env python3
"""`start` 를 누가 언제 올릴 것인가 -- 설계 아이디어를 그림으로만 설명한다.

코드는 안 건드린다. 이 파일이 만드는 것은 문서 하나뿐이다.

    python docs/artifacts/gen_windowing.py   ->  docs/artifacts/windowing.html

타임라인을 손으로 안 그리는 이유: 창이 100 ms 씩 밀리는 그림을 눈대중으로
그리면 밀림 간격이 제각각이 되고, 그러면 **그림이 주장하는 것과 다른 것을
보여준다**. 좌표는 전부 ms 에서 계산한다.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "windowing.html"

# --- 시간 상수 (전부 실제 설계값) ------------------------------------------ #
FRAME_MS   = 10      # sticky OR 창 = kws_frame_ctrl 의 FRAME_CYCLES
NATIVE_T   = 100     # 1 초 = 프레임 100 개
PAD        = 14      # 좌우 패딩 프레임 (값은 -1)
T          = 128     # NATIVE_T + 2*PAD
CLIP_MS    = T * FRAME_MS          # 1280
INFER_MS   = 57.4    # 2.87 M 사이클 / 50 MHz
PERIOD_MS  = 100     # 10 Hz 트리거


def lane(x0: float, w: float) -> "tuple":
    """ms -> px 변환기. 축을 두 개 이상 쓰므로 매번 만들어 쓴다."""
    def mk(span_ms: float):
        return lambda t: x0 + t * w / span_ms
    return mk


# --- 그림 ------------------------------------------------------------------ #

def svg_boundary() -> str:
    """겹치지 않는 창: 단어가 경계에 걸리면 어느 쪽도 온전히 못 본다."""
    X = lane(58, 800)(3840)
    w_a, w_b = 1000, 1900          # 단어가 놓인 구간
    o = ['<svg viewBox="0 0 900 232" role="img" aria-label="겹치지 않는 창에서 '
         '단어가 경계에 걸려 잘리는 모습">']

    # 시간 축
    o.append(f'<line class="ax" x1="{X(0):.1f}" y1="46" x2="{X(3840):.1f}" y2="46"/>')
    for t in range(0, 3841, 480):
        o.append(f'<line class="tick" x1="{X(t):.1f}" y1="42" x2="{X(t):.1f}" y2="50"/>')
        o.append(f'<text class="ttl" x="{X(t):.1f}" y="34" text-anchor="middle">'
                 f'{t/1000:.1f}s</text>')

    # 단어
    o.append(f'<rect class="word" x="{X(w_a):.1f}" y="58" '
             f'width="{X(w_b)-X(w_a):.1f}" height="26" rx="4"/>')
    o.append(f'<text class="wordl" x="{(X(w_a)+X(w_b))/2:.1f}" y="76" '
             f'text-anchor="middle">"yes" 발화 · 900 ms</text>')

    # 창 세 개
    for i in range(3):
        a, b = i * CLIP_MS, (i + 1) * CLIP_MS
        y = 104 + i * 40
        ov = max(0, min(b, w_b) - max(a, w_a))
        hit = ov / (w_b - w_a)
        cls = "win miss" if 0 < hit < 1 else "win"
        o.append(f'<rect class="{cls}" x="{X(a):.1f}" y="{y}" '
                 f'width="{X(b)-X(a):.1f}" height="26" rx="4"/>')
        o.append(f'<text class="winl" x="{X(a)+8:.1f}" y="{y+18}">창 {i}</text>')
        if ov > 0:
            o.append(f'<rect class="ovl" x="{X(max(a,w_a)):.1f}" y="{y}" '
                     f'width="{X(min(b,w_b))-X(max(a,w_a)):.1f}" height="26"/>')
            o.append(f'<text class="pct" x="{X(b)-8:.1f}" y="{y+18}" '
                     f'text-anchor="end">단어의 {hit*100:.0f}% 만</text>')
    o.append("</svg>")
    return "\n".join(o)


def svg_sliding() -> str:
    """100 ms 마다 미는 창: 여러 정렬 중 하나가 단어를 가운데 둔다."""
    X = lane(58, 800)(3840)
    w_a, w_b = 1000, 1900
    best = (w_a + w_b) / 2 - CLIP_MS / 2        # 단어를 정중앙에 두는 시작점
    starts = [500 + 100 * i for i in range(8)]
    o = ['<svg viewBox="0 0 900 320" role="img" aria-label="100 ms 마다 미는 '
         '슬라이딩 창이 여러 정렬을 시도하는 모습">']

    o.append(f'<line class="ax" x1="{X(0):.1f}" y1="46" x2="{X(3840):.1f}" y2="46"/>')
    for t in range(0, 3841, 480):
        o.append(f'<line class="tick" x1="{X(t):.1f}" y1="42" x2="{X(t):.1f}" y2="50"/>')
        o.append(f'<text class="ttl" x="{X(t):.1f}" y="34" text-anchor="middle">'
                 f'{t/1000:.1f}s</text>')

    o.append(f'<rect class="word" x="{X(w_a):.1f}" y="58" '
             f'width="{X(w_b)-X(w_a):.1f}" height="26" rx="4"/>')
    o.append(f'<text class="wordl" x="{(X(w_a)+X(w_b))/2:.1f}" y="76" '
             f'text-anchor="middle">"yes"</text>')

    pick = min(starts, key=lambda s: abs(s - best))
    for i, s in enumerate(starts):
        y = 100 + i * 24
        good = s == pick
        o.append(f'<rect class="{"win good" if good else "win faint"}" '
                 f'x="{X(s):.1f}" y="{y}" width="{X(s+CLIP_MS)-X(s):.1f}" '
                 f'height="18" rx="3"/>')
        if good:
            o.append(f'<text class="pct good" x="{X(s+CLIP_MS)+8:.1f}" y="{y+13}">'
                     f'단어가 거의 가운데 — 학습 분포와 같은 모양</text>')
    o.append(f'<text class="pct" x="{X(starts[0]):.1f}" y="{100+8*24+16:.0f}">'
             f'창 간격 100 ms · 창 폭 1.28 s · 같은 단어를 여덟 가지 정렬로 본다</text>')
    o.append("</svg>")
    return "\n".join(o)


def svg_two_rates() -> str:
    """같은 100 프레임을 담는 데 1 초, 꺼내는 데 45 ms."""
    X = lane(150, 660)(1000)
    o = ['<svg viewBox="0 0 900 250" role="img" aria-label="포착과 재생의 '
         '속도 차이">']

    for i, (lab, ms, cls, y) in enumerate(
            [("포착", 1000.0, "cap", 56), ("재생", 45.0, "rep", 146)]):
        o.append(f'<text class="lane" x="140" y="{y+20}" text-anchor="end">{lab}</text>')
        o.append(f'<rect class="{cls}" x="{X(0):.1f}" y="{y}" '
                 f'width="{X(1000)-X(0):.1f}" height="30" rx="4"/>')
        # 프레임 눈금 -- 둘 다 100 개다. 길이가 같다는 것이 요점이다.
        for k in range(0, 101, 5):
            xx = X(0) + (X(1000) - X(0)) * k / 100
            o.append(f'<line class="fr" x1="{xx:.1f}" y1="{y}" x2="{xx:.1f}" y2="{y+30}"/>')
        o.append(f'<text class="rate" x="{X(1000)+12:.1f}" y="{y+20}">'
                 f'{ms:.0f} ms</text>')
        o.append(f'<text class="sub2" x="{X(0):.1f}" y="{y+48}">'
                 f'{"프레임 하나에 10 ms — 비교기가 정하는 속도" if i == 0 else "프레임 하나에 0.45 ms — kws_top 의 in_ready 가 정하는 속도"}</text>')

    o.append(f'<text class="big" x="{(X(0)+X(1000))/2:.1f}" y="118" '
             f'text-anchor="middle">같은 100 프레임 · 22배</text>')
    o.append(f'<text class="sub2" x="{X(0):.1f}" y="222">'
             f'그래서 재생이 도는 동안 새 프레임이 들어와도 덮이지 않는다 — '
             f'쓰기가 4.5칸 나아갈 때 읽기는 100칸을 끝낸다.</text>')
    o.append("</svg>")
    return "\n".join(o)


def svg_budget() -> str:
    """추론 57.4 ms 가 트리거 주기 안에 들어가는가."""
    # 폭을 700 으로 줄인다. 760 이면 오른쪽 "여유 N.Nx" 라벨이 x=864 에서
    # 시작해 viewBox(900) 을 넘어 잘린다 -- 렌더해 보고 알았다.
    X = lane(92, 700)(400)
    o = ['<svg viewBox="0 0 900 210" role="img" aria-label="10 Hz 와 5 Hz 에서의 '
         '연산 시간 예산">']
    for lab, per, y in [("10 Hz", 100, 52), ("5 Hz", 200, 132)]:
        o.append(f'<text class="lane" x="82" y="{y+22}" text-anchor="end">{lab}</text>')
        o.append(f'<rect class="idle" x="{X(0):.1f}" y="{y}" '
                 f'width="{X(400)-X(0):.1f}" height="32" rx="4"/>')
        t = 0
        while t < 400:
            o.append(f'<rect class="busy" x="{X(t):.1f}" y="{y}" '
                     f'width="{(X(INFER_MS)-X(0)):.1f}" height="32" rx="3"/>')
            t += per
        o.append(f'<text class="rate" x="{X(400)+12:.1f}" y="{y+21}">'
                 f'여유 {per/INFER_MS:.1f}×</text>')
    for t in range(0, 401, 100):
        o.append(f'<line class="tick" x1="{X(t):.1f}" y1="40" x2="{X(t):.1f}" y2="176"/>')
        o.append(f'<text class="ttl" x="{X(t):.1f}" y="32" text-anchor="middle">{t} ms</text>')
    o.append(f'<text class="sub2" x="{X(0):.1f}" y="198">'
             f'짙은 칸이 추론 {INFER_MS} ms (2.87 M 사이클 / 50 MHz). '
             f'빠듯하면 주기를 늘리면 되지 설계를 고칠 일이 아니다.</text>')
    o.append("</svg>")
    return "\n".join(o)


def svg_vote() -> str:
    """10 Hz 면 초당 답이 10 개다. 그대로 쓰면 떨린다."""
    seq = ["sil", "sil", "sil", "unk", "yes", "yes", "unk", "yes",
           "yes", "yes", "yes", "unk", "sil", "sil", "no", "sil",
           "sil", "sil", "sil", "sil"]
    bw, gap = 36, 4
    x0, y = 58, 64
    o = ['<svg viewBox="0 0 900 210" role="img" aria-label="연속 투표로 출력을 '
         '평활하는 방법">']
    o.append(f'<text class="lane" x="{x0}" y="44">kws_top 이 내는 답 — 100 ms 마다 하나</text>')
    run, fired = 0, None
    for i, s in enumerate(seq):
        x = x0 + i * (bw + gap)
        o.append(f'<rect class="ans {s}" x="{x}" y="{y}" width="{bw}" height="30" rx="3"/>')
        o.append(f'<text class="ansl" x="{x+bw/2}" y="{y+20}" '
                 f'text-anchor="middle">{s}</text>')
        run = run + 1 if s == "yes" else 0
        if run == 3 and fired is None:
            fired = i
    fx = x0 + fired * (bw + gap) + bw
    o.append(f'<path class="fire" d="M{fx:.0f} {y+30} V{y+58}"/>')
    o.append(f'<text class="firel" x="{fx+10:.0f}" y="{y+76}">'
             f'yes 3연속 → 검출 선언, 이후 1초 락아웃</text>')
    o.append(f'<text class="sub2" x="{x0}" y="{y+118}">'
             f'평활이 없으면 LED 가 100 ms 마다 바뀐다. 우리 출력은 class_idx '
             f'(argmax) 뿐이라 확률 평균은 못 내지만, 연속 투표면 충분하다.</text>')
    o.append("</svg>")
    return "\n".join(o)


def svg_block() -> str:
    """새 모듈이 어디에 들어가는가."""
    return """
<svg viewBox="0 0 900 210" role="img" aria-label="새 창 모듈이 들어갈 자리">
  <rect class="blk have" x="16" y="56" width="176" height="86" rx="6"/>
  <text class="bt" x="32" y="84">kws_frame_ctrl</text>
  <text class="bs" x="32" y="104">있음 · 검증됨</text>
  <text class="bi" x="32" y="126">sticky OR · 10 ms</text>

  <rect class="blk new" x="336" y="56" width="200" height="86" rx="6"/>
  <text class="bt" x="352" y="84">kws_window</text>
  <text class="bs new" x="352" y="104">새로 만들 것</text>
  <text class="bi" x="352" y="126">원형버퍼 100 · 재생 · 트리거</text>

  <rect class="blk have" x="676" y="56" width="176" height="86" rx="6"/>
  <text class="bt" x="692" y="84">kws_top</text>
  <text class="bs" x="692" y="104">있음 · 검증됨</text>
  <text class="bi" x="692" y="126">57.4 ms / 추론</text>

  <path class="lnk" d="M192 99 H336"/>
  <text class="ll" x="264" y="88" text-anchor="middle">프레임 100 Hz</text>
  <path class="lnk" d="M536 99 H676"/>
  <text class="ll" x="606" y="88" text-anchor="middle">창 128 프레임</text>

  <text class="sub2" x="16" y="186">기존 두 모듈을 고치지 않는다. 둘 다 검증이 끝나 있고, 창을 고르는 일은 프레임을 포착하는 일과 다른 일이다.</text>
</svg>
"""


HTML = """<title>언제 추론할 것인가</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=Source+Sans+3:wght@400;600&display=swap">
<style>
:root{
  --paper:#F1F2EE; --surface:#FBFBF9; --sunk:#E6E8E2;
  --ink:#1B1D1A; --ink2:#4C514A; --ink3:#7E847B;
  --rule:#D2D6CC; --rule2:#BFC4B8;
  --sig:#1B6E70; --word:#8A5A1E; --miss:#BB3A2B; --ok:#2F7A4A;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#131512; --surface:#1B1E1A; --sunk:#0E100D;
    --ink:#E9EBE4; --ink2:#AFB5AA; --ink3:#7C8379;
    --rule:#2C312B; --rule2:#3C423A;
    --sig:#4EB6B2; --word:#D2A054; --miss:#E4705B; --ok:#6FC08C;
  }
}
:root[data-theme="dark"]{
  --paper:#131512; --surface:#1B1E1A; --sunk:#0E100D;
  --ink:#E9EBE4; --ink2:#AFB5AA; --ink3:#7C8379;
  --rule:#2C312B; --rule2:#3C423A;
  --sig:#4EB6B2; --word:#D2A054; --miss:#E4705B; --ok:#6FC08C;
}

*{box-sizing:border-box}
body{background:var(--paper); color:var(--ink); margin:0;
  font-family:'Source Sans 3',system-ui,-apple-system,sans-serif;
  font-size:16px; line-height:1.62; -webkit-font-smoothing:antialiased}
.wrap{max-width:960px; margin:0 auto; padding:48px 24px 96px}

header.mast{border-bottom:2px solid var(--ink); padding-bottom:20px;
  display:flex; flex-wrap:wrap; align-items:flex-end; gap:18px 32px}
h1{font-family:Archivo,system-ui,sans-serif; font-weight:700; font-size:38px;
  line-height:1.08; letter-spacing:-.015em; margin:0; text-wrap:balance; flex:1 1 320px}
.sub{font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--ink3);
  letter-spacing:.06em; text-transform:uppercase; line-height:1.9}
.sub b{color:var(--ink2); font-weight:500}
.lede{margin:26px 0 0; font-size:18px; color:var(--ink2); max-width:64ch}
.lede strong{color:var(--ink); font-weight:600}

section{margin-top:54px}
.eyebrow{display:flex; align-items:baseline; gap:14px; margin-bottom:6px}
.num{font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:500;
  color:var(--sig); letter-spacing:.1em; border:1px solid var(--rule2);
  padding:2px 7px; border-radius:3px}
h2{font-family:Archivo,sans-serif; font-weight:600; font-size:25px;
  letter-spacing:-.012em; margin:0; text-wrap:balance}
.dek{color:var(--ink2); margin:8px 0 22px; max-width:66ch}
p{margin:0 0 14px; max-width:68ch}
code{font-family:'IBM Plex Mono',monospace; font-size:.9em;
  background:var(--sunk); padding:1px 5px; border-radius:3px}

.plate{background:var(--surface); border:1px solid var(--rule);
  border-radius:8px; padding:20px; overflow-x:auto}
.plate svg{display:block; width:100%; height:auto; min-width:640px}
.cap{font-family:'IBM Plex Mono',monospace; font-size:11.5px; color:var(--ink3);
  margin-top:12px; padding-top:10px; border-top:1px solid var(--rule)}

/* 타임라인 공통 */
.ax{stroke:var(--ink3); stroke-width:1.2}
.tick{stroke:var(--rule2); stroke-width:1}
.ttl{font-family:'IBM Plex Mono',monospace; font-size:10.5px; fill:var(--ink3)}
.word{fill:var(--word); opacity:.85}
.wordl{font-family:'IBM Plex Mono',monospace; font-size:11px; fill:var(--surface)}
.win{fill:none; stroke:var(--sig); stroke-width:1.5}
.win.miss{stroke:var(--miss); stroke-dasharray:5 4}
.win.faint{stroke:var(--rule2); stroke-width:1.2}
.win.good{stroke:var(--ok); stroke-width:2.2; fill:var(--ok); fill-opacity:.1}
.winl{font-family:'IBM Plex Mono',monospace; font-size:11px; fill:var(--ink2)}
.ovl{fill:var(--word); opacity:.28}
.pct{font-family:'IBM Plex Mono',monospace; font-size:11px; fill:var(--ink3)}
.pct.good{fill:var(--ok)}
.lane{font-family:Archivo,sans-serif; font-size:14px; font-weight:600; fill:var(--ink)}
.sub2{font-family:'IBM Plex Mono',monospace; font-size:11px; fill:var(--ink3)}
.rate{font-family:'IBM Plex Mono',monospace; font-size:12.5px; fill:var(--ink)}
.big{font-family:Archivo,sans-serif; font-size:17px; font-weight:700; fill:var(--sig)}
rect.cap{fill:var(--sig); opacity:.22; stroke:var(--sig); stroke-width:1.4}
rect.rep{fill:var(--word); opacity:.22; stroke:var(--word); stroke-width:1.4}
.fr{stroke:var(--ink3); stroke-width:.6; opacity:.55}

/* 예산 */
.idle{fill:var(--sunk); stroke:var(--rule2); stroke-width:1}
.busy{fill:var(--sig); opacity:.8}

/* 투표 */
.ans{stroke:var(--rule2); stroke-width:1}
.ans.sil{fill:var(--sunk)}
.ans.unk{fill:var(--rule)}
.ans.yes{fill:var(--ok); opacity:.8}
.ans.no{fill:var(--rule)}
.ansl{font-family:'IBM Plex Mono',monospace; font-size:10.5px; fill:var(--ink)}
.fire{stroke:var(--ok); stroke-width:2; fill:none}
.firel{font-family:'IBM Plex Mono',monospace; font-size:12px; fill:var(--ok)}

/* 블록도 */
.blk{fill:var(--surface); stroke-width:1.5}
.blk.have{stroke:var(--rule2); stroke-dasharray:5 4}
.blk.new{stroke:var(--sig)}
.bt{font-family:Archivo,sans-serif; font-size:16px; font-weight:700; fill:var(--ink)}
.bs{font-family:'IBM Plex Mono',monospace; font-size:10.5px; fill:var(--ink3)}
.bs.new{fill:var(--sig)}
.bi{font-family:'Source Sans 3',sans-serif; font-size:13px; fill:var(--ink2)}
.lnk{stroke:var(--ink2); stroke-width:1.5; fill:none}
.ll{font-family:'IBM Plex Mono',monospace; font-size:11px; fill:var(--ink3)}

.warn{border-left:3px solid var(--miss); background:var(--surface);
  border-top:1px solid var(--rule); border-right:1px solid var(--rule);
  border-bottom:1px solid var(--rule); border-radius:0 7px 7px 0;
  padding:16px 20px; margin:20px 0 0}
.warn h3{font-family:Archivo,sans-serif; font-size:15px; font-weight:600;
  margin:0 0 6px; color:var(--miss)}
.warn p{font-size:14.5px; color:var(--ink2)}

.steps{list-style:none; padding:0; margin:0; display:grid; gap:2px;
  counter-reset:s}
.steps li{background:var(--surface); border:1px solid var(--rule);
  padding:14px 18px 14px 52px; position:relative; font-size:15px; color:var(--ink2)}
.steps li:first-child{border-radius:8px 8px 0 0}
.steps li:last-child{border-radius:0 0 8px 8px}
.steps li b{color:var(--ink); font-weight:600}
.steps li::before{content:counter(s); counter-increment:s;
  font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--sig);
  position:absolute; left:18px; top:15px; border:1px solid var(--rule2);
  border-radius:3px; padding:1px 6px}

footer{margin-top:64px; padding-top:18px; border-top:1px solid var(--rule);
  font-family:'IBM Plex Mono',monospace; font-size:11.5px; color:var(--ink3);
  display:flex; flex-wrap:wrap; gap:8px 24px}
@media (max-width:640px){h1{font-size:29px} h2{font-size:21px} .wrap{padding:32px 16px 64px}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>

<div class="wrap">

<header class="mast">
  <h1>언제 추론할 것인가</h1>
  <div class="sub">
    프레임 <b>10 ms</b> · 창 <b>1.28 s</b><br>
    추론 <b>57.4 ms</b> @ 50 MHz<br>
    아이디어만 · 코드 변경 없음
  </div>
</header>

<p class="lede">지금 <code>start</code> 는 확장 포트의 핀 하나이고 DIP 스위치를
가정하고 있다. 보드에서는 <strong>아무도 누르지 않는다.</strong> 비교기는 쉬지 않고
들어오는데 언제 1초를 잘라 신경망에 넣을지를 정하는 것이 아직 없다.</p>

<section>
  <div class="eyebrow"><span class="num">문제</span><h2>그냥 반복하면 단어가 잘린다</h2></div>
  <p class="dek">가장 단순한 답은 <code>start = ~busy</code> 로 1.28초마다 쉬지 않고
  도는 것이다. 창이 서로 겹치지 않으므로 <strong>단어가 경계에 걸리면 어느 쪽도
  온전히 보지 못한다.</strong></p>
  <div class="plate">
    __BOUNDARY__
    <div class="cap">창 1 과 창 2 가 단어를 나눠 갖는다 · 어느 쪽도 학습 때 본
    모양이 아니다 · 단어가 1초 가까이라 이 일은 자주 일어난다</div>
  </div>
</section>

<section>
  <div class="eyebrow"><span class="num">해법</span><h2>창을 100 ms 씩 밀면서 여러 번 본다</h2></div>
  <p class="dek">같은 소리를 <strong>여러 정렬로</strong> 본다. 학습 데이터(GSC)는
  단어가 1초 안에 대충 가운데 놓여 있으므로, 미는 창들 중 하나는 그 모양에 가깝게
  떨어진다. 경계 문제가 사라지는 것이 아니라 <strong>매번 다른 곳에 생겼다가
  지나간다.</strong></p>
  <div class="plate">
    __SLIDING__
    <div class="cap">창 폭은 그대로 1.28 s · 시작점만 100 ms 씩 민다 ·
    같은 단어에 대해 열 번 넘게 답이 나오고 그 중 잘 맞은 것이 섞인다</div>
  </div>
</section>

<section>
  <div class="eyebrow"><span class="num">구조</span><h2>담는 속도와 꺼내는 속도가 다르다</h2></div>
  <p class="dek">이것이 설계의 핵심이다. <strong>프레임을 10 ms 간격으로 신경망에
  줄 필요가 없다.</strong> <code>kws_top</code> 은 ready/valid 로 받으므로 받아들일 수
  있는 만큼 빨리 밀어넣으면 되고, 그 한계는 conv1 이 정해서 프레임당 0.45 ms 다.</p>
  <div class="plate">
    __RATES__
    <div class="cap">포착은 비교기·sticky OR 이 정하고 재생은 kws_top 이 정한다 ·
    이 22배가 원형 버퍼 하나로 두 속도를 잇게 해 준다</div>
  </div>
  <p style="margin-top:20px">그래서 <strong>최근 100 프레임을 늘 들고 있는 원형
  버퍼</strong> 하나면 된다. 100 × 16 비트 = 1,600 비트로, BRAM 한 개도 안 쓴다.
  트리거가 오면 그 100 개를 앞뒤에 <code>-1</code> 패딩 14 개씩 붙여 빠르게 밀어넣는다.</p>

  <div class="warn">
    <h3>패딩을 학습과 똑같이 유지하는 것이 조건이다</h3>
    <p>네트워크는 <b>−1 로 채운 14 프레임 + 실제 100 프레임 + −1 14 프레임</b>을
    보도록 학습됐다. 버퍼를 100 으로 두고 재생기가 패딩을 붙이면 입력 분포가
    학습과 <b>완전히 같다</b> — 재학습도, 가중치 변경도, ICD 변경도 없다.
    최근 프레임을 패딩 자리에 끼워 넣고 싶은 유혹이 생기는데, 그러면 학습 때 본 적
    없는 입력이 된다.</p>
  </div>
</section>

<section>
  <div class="eyebrow"><span class="num">예산</span><h2>57.4 ms 가 주기 안에 들어가는가</h2></div>
  <div class="plate">
    __BUDGET__
    <div class="cap">2.87 M 사이클은 tb_top 이 실제로 기록한 값이다 ·
    10 Hz 는 1.7배, 5 Hz 는 3.5배</div>
  </div>
  <p style="margin-top:18px">10 Hz 가 <code>CLAUDE.md</code> §0 이 가정한 추론율이지만
  여유가 1.7배로 빠듯하다. <strong>5 Hz 로 시작하는 편이 안전하다</strong> — 1초짜리
  단어를 200 ms 간격으로 훑어도 놓치지 않고, 여유가 두 배가 된다. 파라미터 하나다.</p>
</section>

<section>
  <div class="eyebrow"><span class="num">출력</span><h2>초당 열 개의 답을 어떻게 다룰 것인가</h2></div>
  <p class="dek">빠뜨리기 쉬운 부분이다. 10 Hz 로 돌리면 답도 10 Hz 로 나온다.
  그대로 LED 에 걸면 <strong>100 ms 마다 바뀐다.</strong></p>
  <div class="plate">
    __VOTE__
    <div class="cap">"동작한다" 와 "동작하는 것처럼 보인다" 를 가르는 부분이다</div>
  </div>
</section>

<section>
  <div class="eyebrow"><span class="num">자리</span><h2>어디에 넣을 것인가</h2></div>
  <div class="plate">
    __BLOCK__
    <div class="cap">frame_ctrl 과 kws_top 은 손대지 않는다</div>
  </div>
</section>

<section>
  <div class="eyebrow"><span class="num">순서</span><h2>단계로 나누면</h2></div>
  <ol class="steps">
    <li><b><code>start = ~busy</code> 로 자동 반복.</b> RTL 추가가 0 이고, 보드에서
      end-to-end 가 도는지부터 본다. 경계에 걸린 단어는 놓치지만 <b>도는 것</b>은
      확인된다.</li>
    <li><b>원형 버퍼 + 재생기 + 주기 트리거.</b> 위 구조. 새 모듈 하나이고
      기존 둘은 안 건드린다.</li>
    <li><b>연속 투표 평활.</b> 여기서부터 데모로 보인다.</li>
    <li><b>외부 <code>start</code> 핀은 수동 단발로 남긴다.</b> 브링업과 XSim
      테스트벤치가 그것을 쓴다 — 자동화가 들어와도 그 경로를 없애면 안 된다.</li>
  </ol>
</section>

<footer>
  <span>근거: rtl/kws_frame_ctrl.v · rtl/kws_top.v · rtl/tb/tb_top.v</span>
  <span>CLAUDE.md §2.8 · docs/ICD.md §5</span>
  <span>생성: docs/artifacts/gen_windowing.py</span>
</footer>

</div>
"""


def main() -> None:
    html = (HTML
            .replace("__BOUNDARY__", svg_boundary())
            .replace("__SLIDING__", svg_sliding())
            .replace("__RATES__", svg_two_rates())
            .replace("__BUDGET__", svg_budget())
            .replace("__VOTE__", svg_vote())
            .replace("__BLOCK__", svg_block()))
    assert "__" not in html.split("</style>")[-1], "치환 안 된 자리표시자"
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({len(html):,} bytes)")
    # 그림이 주장하는 숫자를 숫자로 확인한다
    assert NATIVE_T + 2 * PAD == T
    assert T * FRAME_MS == CLIP_MS == 1280
    assert abs(PERIOD_MS / INFER_MS - 1.7) < 0.05
    print(f"T={T} clip={CLIP_MS}ms infer={INFER_MS}ms "
          f"10Hz 여유={PERIOD_MS/INFER_MS:.1f}x 5Hz 여유={2*PERIOD_MS/INFER_MS:.1f}x")


if __name__ == "__main__":
    main()
