#!/usr/bin/env python3
"""AFE <-> FPGA 킷 링크 조립도를 생성한다.

손으로 안 쓰는 이유: 커넥터 핀이 50 + 20 개고 `핀 = ch + 3` 대응을 타이핑하면
반드시 어딘가 틀린다. 그리고 틀려도 그림은 멀쩡해 보인다 -- 채널이 섞였을 때
증상이 "정확도가 좀 낮다" 뿐인 것과 같은 종류의 사고다.

    python docs/artifacts/gen_afe_link.py

산출물: docs/artifacts/afe_link.html
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "afe_link.html"

# --- 핀 배정 (docs/hanback_kit.md 6) -------------------------------------- #
# 두 커넥터가 같은 규칙을 쓴다: 핀 1,2 = 5V / 핀 3..18 = ch00..ch15 / 끝 두 핀 = GND.
# 그래서 `핀 = ch + 3` 이 양쪽에서 그대로 성립하고, 어댑터 배선이 대부분 직결이 된다.
N_CH = 16
KIT_PINS, AFE_PINS = 50, 20


def role(pin: int, n_pins: int) -> str:
    if pin <= 2:
        return "v5"
    if 3 <= pin <= 2 + N_CH:
        return "sig"
    if pin > n_pins - 2:
        return "gnd"
    return "nc"


def ch_of(pin: int) -> int | None:
    return pin - 3 if 3 <= pin <= 2 + N_CH else None


def wires() -> list[tuple[int, int, str]]:
    """킷쪽 50핀 -> AFE쪽 20핀. (from, to, role)"""
    w = [(p, p, role(p, KIT_PINS)) for p in range(1, 19)]  # 1..18 직결
    w += [(49, 19, "gnd"), (50, 20, "gnd")]
    return w


# --- SVG 조각 -------------------------------------------------------------- #

def pin_grid(n_pins: int, *, cell=30.0, r=10.5, pad_x=54.0, top=54.0) -> str:
    """2 x (n/2) 핀 그리드. 홀수 핀이 윗줄, 짝수 핀이 아랫줄."""
    cols = n_pins // 2
    row_a, row_b = top, top + 40
    w = pad_x * 2 + (cols - 1) * cell
    h = row_b + 62
    o = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" role="img" '
         f'aria-label="{n_pins}핀 커넥터 핀 배치">']

    # 줄 이름
    o.append(f'<text class="rl" x="{pad_x - 20:.1f}" y="{row_a + 4:.1f}" '
             f'text-anchor="end">홀수</text>')
    o.append(f'<text class="rl" x="{pad_x - 20:.1f}" y="{row_b + 4:.1f}" '
             f'text-anchor="end">짝수</text>')

    for c in range(cols):
        x = pad_x + c * cell
        for pin, y in ((2 * c + 1, row_a), (2 * c + 2, row_b)):
            rl = role(pin, n_pins)
            o.append(f'<circle class="pin {rl}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>')
            o.append(f'<text class="pn {rl}" x="{x:.1f}" y="{y + 3.4:.1f}" '
                     f'text-anchor="middle">{pin}</text>')
            ch = ch_of(pin)
            if ch is not None:
                ly = row_a - 20 if y == row_a else row_b + 26
                o.append(f'<text class="chl" x="{x:.1f}" y="{ly:.1f}" '
                         f'text-anchor="middle">ch{ch:02d}</text>')

    # 키홈: 가운데 아래쪽 벽
    kx = pad_x + (cols - 1) * cell / 2
    o.append(f'<path class="key" d="M{kx - 16:.1f} {h - 26:.1f} '
             f'h32 v-13 h-32 z"/>')
    o.append(f'<text class="kl" x="{kx:.1f}" y="{h - 8:.1f}" '
             f'text-anchor="middle">키홈</text>')
    o.append("</svg>")
    return "\n".join(o)


def perfboard(cols=35, rows=16, s=13.0, pad=26.0) -> str:
    # rows 는 그리는 범위지 기판 크기가 아니다. 90x70 만능기판은 27줄이지만
    # 배선이 전부 위 13줄에서 끝나므로 아래는 잘라서 그린다 -- 빈 판을 절반이나
    # 그리면 도면이 아니라 여백이 된다.
    """만능기판 위 배치도. 두 헤더의 핀 1 을 같은 열에 두면 배선이 수직선이 된다."""
    C0 = 6                      # 두 헤더 공통 시작 열 (1-based)
    R50A, R50B = 3, 4           # 50핀 홀수/짝수 줄
    R20A, R20B = 12, 13         # 20핀 홀수/짝수 줄

    def X(c): return pad + (c - 1) * s
    def Y(r): return pad + (r - 1) * s

    w = pad * 2 + (cols - 1) * s
    h = pad * 2 + (rows - 1) * s
    o = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" role="img" '
         f'aria-label="어댑터 만능기판 배치도">']

    o.append(f'<rect class="board" x="6" y="6" width="{w - 12:.0f}" '
             f'height="{h - 12:.0f}" rx="6"/>')
    # 홀 그리드
    o.append('<defs><pattern id="holes" width="%.1f" height="%.1f" '
             'patternUnits="userSpaceOnUse" patternTransform="translate(%.1f %.1f)">'
             '<circle class="hole" cx="0" cy="0" r="1.7"/></pattern></defs>'
             % (s, s, X(1), Y(1)))
    o.append(f'<rect x="{X(1) - 4:.1f}" y="{Y(1) - 4:.1f}" '
             f'width="{(cols - 1) * s + 8:.1f}" height="{(rows - 1) * s + 8:.1f}" '
             f'fill="url(#holes)"/>')

    # 헤더 몸통
    def body(c0, npins, ra, rb, label, sub):
        span = (npins // 2 - 1) * s
        x0, y0 = X(c0) - 10, Y(ra) - 11
        o.append(f'<rect class="hdr" x="{x0:.1f}" y="{y0:.1f}" '
                 f'width="{span + 20:.1f}" height="{(rb - ra) * s + 22:.1f}" rx="3"/>')
        o.append(f'<text class="hl" x="{x0 + 8:.1f}" y="{y0 - 9:.1f}">{label}</text>')
        o.append(f'<text class="hs" x="{x0 + span + 12:.1f}" y="{y0 - 9:.1f}" '
                 f'text-anchor="end">{sub}</text>')
        for i in range(npins // 2):
            for r in (ra, rb):
                pin = 2 * i + 1 if r == ra else 2 * i + 2
                o.append(f'<circle class="hp {role(pin, npins)}" '
                         f'cx="{X(c0 + i):.1f}" cy="{Y(r):.1f}" r="3.6"/>')

    body(C0, KIT_PINS, R50A, R50B, "50핀 박스헤더", "→ 킷 J6")
    body(C0, AFE_PINS, R20A, R20B, "20핀 박스헤더", "→ AFE 기판")

    # 배선
    for src, dst, rl in wires():
        if src <= 18:
            c = C0 + (src - 1) // 2
            ra, rb = (R50A, R20A) if src % 2 else (R50B, R20B)
            o.append(f'<path class="w {rl}" d="M{X(c):.1f} {Y(ra):.1f} '
                     f'V{Y(rb):.1f}"/>')
        else:
            # GND 두 가닥. 두 헤더 사이 빈 띠로 가로지르되, 세로 점퍼가 있는
            # 열 6~14 의 **오른쪽**으로만 지나가게 한다 -- 목적지가 20핀의 오른쪽
            # 끝(열 15)이라 교차가 한 번도 안 생긴다.
            c_s = C0 + (src - 1) // 2
            c_d = C0 + (dst - 1) // 2
            r_s, r_d = (R50A, R20A) if src % 2 else (R50B, R20B)
            mid = Y(R50B) + (2.7 if src % 2 else 4.2) * s
            o.append(f'<path class="w {rl}" d="M{X(c_s):.1f} {Y(r_s):.1f} '
                     f'V{mid:.1f} H{X(c_d):.1f} V{Y(r_d):.1f}"/>')

    o.append(f'<text class="note" x="{X(C0):.1f}" y="{Y(rows) + 4:.1f}">'
             f'열 {C0}부터 두 헤더의 1번 핀을 맞추면 18가닥이 수직 직결이 된다</text>')
    o.append("</svg>")
    return "\n".join(o)


def chain() -> str:
    """전체 체인 블록도."""
    return """
<svg viewBox="0 0 900 250" role="img" aria-label="AFE에서 FPGA 킷까지의 연결 체인">
  <rect class="blk afe" x="14" y="52" width="196" height="128" rx="7"/>
  <text class="bt" x="30" y="80">AFE 기판</text>
  <text class="bs" x="30" y="100">동료 담당</text>
  <text class="bi" x="30" y="128">비교기 ×16</text>
  <text class="bi" x="30" y="148">LDO 5V→3.3/1.8</text>
  <text class="bi" x="30" y="168">2×10 박스헤더</text>

  <rect class="blk adp" x="352" y="52" width="196" height="128" rx="7"/>
  <text class="bt" x="368" y="80">어댑터</text>
  <text class="bs" x="368" y="100">만능기판 · 우리 담당</text>
  <text class="bi" x="368" y="128">2×25 박스헤더</text>
  <text class="bi" x="368" y="148">2×10 박스헤더</text>
  <text class="bi" x="368" y="168">점퍼 20가닥</text>

  <rect class="blk kit" x="690" y="52" width="196" height="128" rx="7"/>
  <text class="bt" x="706" y="80">FPGA 킷</text>
  <text class="bs" x="706" y="100">XC7S75 · 연구실</text>
  <text class="bi" x="706" y="128">J6 Exp. Port</text>
  <text class="bi" x="706" y="148">2×25, 키홈</text>
  <text class="bi" x="706" y="168">직렬 33Ω 내장</text>

  <path class="lnk" d="M210 116 H352"/>
  <text class="ll" x="281" y="106" text-anchor="middle">20핀 리본</text>
  <text class="lv" x="281" y="140" text-anchor="middle">16 + 5V + GND</text>

  <path class="lnk" d="M548 116 H690"/>
  <text class="ll" x="619" y="106" text-anchor="middle">50핀 리본 20cm</text>
  <text class="lv" x="619" y="140" text-anchor="middle">18/50 사용</text>

  <path class="flow sig" d="M690 210 H210"/>
  <text class="fl sig" x="450" y="202" text-anchor="middle">비교기 16가닥 ─ 아날로그에서 FPGA 로</text>
  <path class="flow v5" d="M210 236 H690"/>
  <text class="fl v5" x="450" y="228" text-anchor="middle">5V ─ 킷에서 AFE 로</text>
</svg>
"""


def afe_side() -> str:
    """AFE 기판 쪽: 비교기 16개가 어떻게 헤더 한 개로 모이는가.

    좌변의 블록과 우변의 헤더 핀을 **같은 y 에** 두어 배선이 수평 직선이 되게 한다.
    선이 꺾이면 보는 사람이 "왜 꺾였지" 를 먼저 묻는데, 여기엔 이유가 없다.
    """
    rows = [("1", "5V", "v5"), ("2", "5V", "v5"), ("3", "ch00", "sig"),
            ("4", "ch01", "sig"), ("5–16", "ch02–ch13", "sig"), ("17", "ch14", "sig"),
            ("18", "ch15", "sig"), ("19", "GND", "gnd"), ("20", "GND", "gnd")]
    step, y0 = 36, 54
    ys = [y0 + i * step for i in range(len(rows))]
    h = ys[-1] + 46

    o = [f'<svg viewBox="0 0 880 {h}" role="img" '
         f'aria-label="AFE 기판에서 비교기 16개를 2x10 헤더로 모으는 방법">']

    def block(y, label, cls="cmp", dashed=False):
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        o.append(f'<rect class="{cls}" x="18" y="{y - 15}" width="188" height="30" '
                 f'rx="4"{dash}/>')
        o.append(f'<text class="ct" x="32" y="{y + 4}">{label}</text>')

    # 전원부는 핀 1,2 / 19,20 네 개에 걸친다
    o.append(f'<rect class="cmp" x="18" y="{ys[0] - 15}" width="188" '
             f'height="{ys[1] - ys[0] + 30}" rx="4"/>')
    o.append(f'<text class="ct" x="32" y="{(ys[0] + ys[1]) / 2 + 4:.0f}">'
             f'LDO 입력 (5V)</text>')
    block(ys[2], "비교기 ch00")
    block(ys[3], "비교기 ch01")
    block(ys[4], "비교기 ch02 ~ ch13   (12개)", dashed=True)
    block(ys[5], "비교기 ch14")
    block(ys[6], "비교기 ch15")
    o.append(f'<rect class="cmp" x="18" y="{ys[7] - 15}" width="188" '
             f'height="{ys[8] - ys[7] + 30}" rx="4"/>')
    o.append(f'<text class="ct" x="32" y="{(ys[7] + ys[8]) / 2 + 4:.0f}">'
             f'기판 접지</text>')

    # 헤더
    top, bot = ys[0] - 24, ys[-1] + 24
    o.append(f'<rect class="hdr2" x="470" y="{top}" width="230" '
             f'height="{bot - top}" rx="6"/>')
    o.append(f'<text class="hl2" x="585" y="{top - 10}" text-anchor="middle">'
             f'2×10 박스헤더 → 어댑터</text>')

    for (p, lab, rl), y in zip(rows, ys):
        o.append(f'<path class="w {rl}" d="M206 {y} H500"/>')
        o.append(f'<circle class="hp {rl}" cx="500" cy="{y}" r="6"/>')
        o.append(f'<text class="hr" x="518" y="{y + 4}">핀 {p}</text>')
        o.append(f'<text class="hv {rl}" x="686" y="{y + 4}" '
                 f'text-anchor="end">{lab}</text>')
    o.append("</svg>")
    return "\n".join(o)


# --- 페이지 ---------------------------------------------------------------- #

def wire_table() -> str:
    rows = []
    for src, dst, rl in wires():
        ch = ch_of(src)
        what = {"v5": "5V", "gnd": "GND"}.get(rl, f"ch{ch:02d}" if ch is not None else "")
        rows.append(f'<tr><td class="m">{src}</td><td class="m">{dst}</td>'
                    f'<td><span class="tag {rl}">{what}</span></td>'
                    f'<td class="q">{"직결 (같은 열, 수직)" if src <= 18 else "끝에서 끌어옴"}</td></tr>')
    return "\n".join(rows)


HTML = """<title>AFE 링크 조립도</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=Source+Sans+3:wght@400;600&display=swap">
<style>
:root{
  --paper:#F1F2EE; --surface:#FBFBF9; --sunk:#E6E8E2;
  --ink:#1B1D1A; --ink2:#4C514A; --ink3:#7E847B;
  --rule:#D2D6CC; --rule2:#BFC4B8;
  --v5:#BB3A2B; --gnd:#31353A; --sig:#1B6E70; --nc:#B2B7AE;
  --board:#20482F; --hole:#8FA398;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#131512; --surface:#1B1E1A; --sunk:#0E100D;
    --ink:#E9EBE4; --ink2:#AFB5AA; --ink3:#7C8379;
    --rule:#2C312B; --rule2:#3C423A;
    --v5:#E4705B; --gnd:#9BA3AC; --sig:#4EB6B2; --nc:#575E54;
    --board:#17301F; --hole:#5C7466;
  }
}
:root[data-theme="dark"]{
  --paper:#131512; --surface:#1B1E1A; --sunk:#0E100D;
  --ink:#E9EBE4; --ink2:#AFB5AA; --ink3:#7C8379;
  --rule:#2C312B; --rule2:#3C423A;
  --v5:#E4705B; --gnd:#9BA3AC; --sig:#4EB6B2; --nc:#575E54;
  --board:#17301F; --hole:#5C7466;
}

*{box-sizing:border-box}
body{background:var(--paper); color:var(--ink);
  font-family:'Source Sans 3',system-ui,-apple-system,sans-serif;
  font-size:16px; line-height:1.62; margin:0;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:960px; margin:0 auto; padding:48px 24px 96px}

header.mast{border-bottom:2px solid var(--ink); padding-bottom:20px;
  display:flex; flex-wrap:wrap; align-items:flex-end; gap:18px 32px}
h1{font-family:Archivo,system-ui,sans-serif; font-weight:700; font-size:38px;
  line-height:1.08; letter-spacing:-.015em; margin:0; text-wrap:balance; flex:1 1 340px}
.sub{font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--ink3);
  letter-spacing:.06em; text-transform:uppercase; line-height:1.9}
.sub b{color:var(--ink2); font-weight:500}

.lede{margin:26px 0 0; font-size:18px; color:var(--ink2); max-width:64ch}
.lede strong{color:var(--ink); font-weight:600}

section{margin-top:56px; scroll-margin-top:24px}
.eyebrow{display:flex; align-items:baseline; gap:14px; margin-bottom:6px}
.num{font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:500;
  color:var(--sig); letter-spacing:.1em; border:1px solid var(--rule2);
  padding:2px 7px; border-radius:3px}
h2{font-family:Archivo,system-ui,sans-serif; font-weight:600; font-size:25px;
  letter-spacing:-.012em; margin:0; text-wrap:balance}
.dek{color:var(--ink2); margin:8px 0 22px; max-width:66ch}
p{margin:0 0 14px; max-width:68ch}
p:last-child{margin-bottom:0}
strong{font-weight:600}
code{font-family:'IBM Plex Mono',monospace; font-size:.9em;
  background:var(--sunk); padding:1px 5px; border-radius:3px}

.plate{background:var(--surface); border:1px solid var(--rule); border-radius:8px;
  padding:22px; overflow-x:auto}
.plate svg{display:block; width:100%; height:auto; min-width:560px}
.cap{font-family:'IBM Plex Mono',monospace; font-size:11.5px; color:var(--ink3);
  letter-spacing:.03em; margin-top:14px; padding-top:12px;
  border-top:1px solid var(--rule)}

/* 핀 그리드 */
.pin{stroke:var(--surface); stroke-width:1.5}
.pin.v5{fill:var(--v5)} .pin.sig{fill:var(--sig)}
.pin.gnd{fill:var(--gnd)} .pin.nc{fill:none; stroke:var(--nc); stroke-width:1.4}
.pn{font-family:'IBM Plex Mono',monospace; font-size:9.5px; fill:var(--surface)}
.pn.nc{fill:var(--nc)}
.chl{font-family:'IBM Plex Mono',monospace; font-size:9.5px; fill:var(--sig);
  letter-spacing:.02em}
.rl{font-family:'IBM Plex Mono',monospace; font-size:10px; fill:var(--ink3)}
.key{fill:var(--sunk); stroke:var(--rule2); stroke-width:1.2}
.kl{font-family:'IBM Plex Mono',monospace; font-size:10px; fill:var(--ink3)}

/* 만능기판 */
.board{fill:var(--board); opacity:.14; stroke:var(--rule2); stroke-width:1}
.hole{fill:var(--hole); opacity:.5}
.hdr{fill:var(--sunk); stroke:var(--ink2); stroke-width:1.3; opacity:.92}
.hl{font-family:Archivo,sans-serif; font-size:12.5px; font-weight:600; fill:var(--ink)}
.hs{font-family:'IBM Plex Mono',monospace; font-size:10.5px; fill:var(--ink3)}
.hp{stroke:none}
.hp.v5{fill:var(--v5)} .hp.sig{fill:var(--sig)}
.hp.gnd{fill:var(--gnd)} .hp.nc{fill:var(--nc); opacity:.55}
.w{fill:none; stroke-width:2.1; stroke-linecap:round; stroke-linejoin:round}
.w.v5{stroke:var(--v5)} .w.sig{stroke:var(--sig)} .w.gnd{stroke:var(--gnd)}
.note{font-family:'IBM Plex Mono',monospace; font-size:11px; fill:var(--ink3)}

/* 체인 */
.blk{fill:var(--surface); stroke-width:1.4}
.blk.afe{stroke:var(--rule2); stroke-dasharray:5 4}
.blk.adp{stroke:var(--sig)}
.blk.kit{stroke:var(--rule2)}
.bt{font-family:Archivo,sans-serif; font-size:17px; font-weight:700; fill:var(--ink)}
.bs{font-family:'IBM Plex Mono',monospace; font-size:10.5px; fill:var(--ink3);
  letter-spacing:.04em}
.bi{font-family:'Source Sans 3',sans-serif; font-size:13.5px; fill:var(--ink2)}
.lnk{stroke:var(--ink2); stroke-width:1.6; fill:none}
.ll{font-family:Archivo,sans-serif; font-size:13px; font-weight:600; fill:var(--ink)}
.lv{font-family:'IBM Plex Mono',monospace; font-size:10.5px; fill:var(--ink3)}
.flow{fill:none; stroke-width:1.6; stroke-dasharray:6 5}
.flow.sig{stroke:var(--sig)} .flow.v5{stroke:var(--v5)}
.fl{font-family:'IBM Plex Mono',monospace; font-size:11px}
.fl.sig{fill:var(--sig)} .fl.v5{fill:var(--v5)}

/* AFE 쪽 */
.cmp{fill:var(--sunk); stroke:var(--rule2); stroke-width:1.2}
.ct{font-family:'IBM Plex Mono',monospace; font-size:12px; fill:var(--ink)}
.pl{font-family:'IBM Plex Mono',monospace; font-size:11px; fill:var(--ink3)}
.ell{font-family:'IBM Plex Mono',monospace; font-size:13px; fill:var(--ink3)}
.hdr2{fill:none; stroke:var(--ink2); stroke-width:1.4; stroke-dasharray:4 3}
.hl2{font-family:Archivo,sans-serif; font-size:13px; font-weight:600; fill:var(--ink)}
.hr{font-family:'IBM Plex Mono',monospace; font-size:11px; fill:var(--ink3)}
.hv{font-family:'IBM Plex Mono',monospace; font-size:11.5px}
.hv.v5{fill:var(--v5)} .hv.sig{fill:var(--sig)} .hv.gnd{fill:var(--gnd)}

/* 표 */
.tbl{width:100%; border-collapse:collapse; font-size:14.5px;
  font-variant-numeric:tabular-nums}
.tbl th{font-family:'IBM Plex Mono',monospace; font-size:10.5px; font-weight:500;
  text-transform:uppercase; letter-spacing:.08em; color:var(--ink3);
  text-align:left; padding:0 12px 8px; border-bottom:1px solid var(--rule2)}
.tbl td{padding:7px 12px; border-bottom:1px solid var(--rule); color:var(--ink2)}
.tbl td.m{font-family:'IBM Plex Mono',monospace; color:var(--ink)}
.tbl td.q{color:var(--ink3); font-size:13.5px}
.tag{font-family:'IBM Plex Mono',monospace; font-size:11px; padding:2px 8px;
  border-radius:3px; color:var(--surface); font-weight:500}
.tag.v5{background:var(--v5)} .tag.sig{background:var(--sig)}
.tag.gnd{background:var(--gnd)}
.scroll{overflow-x:auto}

/* 부품 */
.bom{display:grid; gap:1px; background:var(--rule); border:1px solid var(--rule);
  border-radius:8px; overflow:hidden}
.bom > div{background:var(--surface); padding:16px 18px; display:grid;
  grid-template-columns:auto 1fr auto; gap:4px 16px; align-items:baseline}
.bom .pt{font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--sig);
  letter-spacing:.08em}
.bom .pnm{font-family:Archivo,sans-serif; font-weight:600; font-size:15.5px;
  color:var(--ink)}
.bom .pq{font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--ink3);
  text-align:right; white-space:nowrap}
.bom .pd{grid-column:2/4; color:var(--ink2); font-size:14px}

/* 경고 */
.warn{border-left:3px solid var(--v5); background:var(--surface);
  border-top:1px solid var(--rule); border-right:1px solid var(--rule);
  border-bottom:1px solid var(--rule); border-radius:0 7px 7px 0;
  padding:16px 20px; margin:20px 0 0}
.warn h3{font-family:Archivo,sans-serif; font-size:15px; font-weight:600;
  margin:0 0 6px; color:var(--v5)}
.warn p{font-size:14.5px; color:var(--ink2)}

/* 순서 */
.steps{list-style:none; padding:0; margin:0; display:grid; gap:2px}
.steps li{background:var(--surface); border:1px solid var(--rule);
  padding:14px 18px 14px 52px; position:relative; font-size:15px; color:var(--ink2)}
.steps li:first-child{border-radius:8px 8px 0 0}
.steps li:last-child{border-radius:0 0 8px 8px}
.steps li b{color:var(--ink); font-weight:600}
.steps li::before{content:counter(s); counter-increment:s;
  font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--sig);
  position:absolute; left:18px; top:15px; border:1px solid var(--rule2);
  border-radius:3px; padding:1px 6px}
.steps{counter-reset:s}

footer{margin-top:72px; padding-top:20px; border-top:1px solid var(--rule);
  font-family:'IBM Plex Mono',monospace; font-size:11.5px; color:var(--ink3);
  display:flex; flex-wrap:wrap; gap:8px 24px}

@media (max-width:640px){
  h1{font-size:29px} h2{font-size:21px} .wrap{padding:32px 16px 64px}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>

<div class="wrap">

<header class="mast">
  <h1>AFE 링크 조립도</h1>
  <div class="sub">
    비교기 <b>16</b>가닥 · 5V · GND<br>
    한백 XC7S75 킷 <b>J6</b> Exp. Port<br>
    2026-09-07
  </div>
</header>

<p class="lede">아날로그 기판에서 나오는 비교기 16가닥을 FPGA 킷의 확장 포트까지
가져오는 물리 설계다. 핵심은 <strong>중간에 어댑터를 하나 두는 것</strong> — 킷의
커넥터는 50핀이고 그 중 18개만 쓰는데, 그 8.3&nbsp;cm짜리 부품을 아날로그 기판에
올릴 이유가 없다.</p>

<section>
  <div class="eyebrow"><span class="num">체인</span><h2>세 덩어리와 두 케이블</h2></div>
  <p class="dek">신호는 왼쪽으로, 전원은 오른쪽으로 흐른다. FPGA 에서 아날로그로
  가는 제어선은 <strong>하나도 없다</strong> — sticky OR 은 펄스가 언제 왔는지를
  묻지 않기 때문에 프로토콜이 필요 없다.</p>
  <div class="plate">
    __CHAIN__
    <div class="cap">어댑터만 우리가 만든다. AFE 기판은 2×10 헤더 하나만 알면 되고,
    킷은 이미 있는 것을 쓴다.</div>
  </div>
</section>

<section>
  <div class="eyebrow"><span class="num">킷 쪽</span><h2>J6 의 50핀 중 18개만 쓴다</h2></div>
  <p class="dek">매뉴얼 99쪽 회로도에서 읽은 배치다. 핀 1·2 가 5V, 핀 49·50 이 GND,
  그 사이 46개가 EXT0–EXT45 다. 우리는 앞에서부터 16개만 쓴다.</p>
  <div class="plate">
    __GRID50__
    <div class="cap">핀 = ch + 3 · 회색 테두리(핀 19–48)는 연결하지 않는다 ·
    홀수 핀에 짝수 채널, 짝수 핀에 홀수 채널이 온다</div>
  </div>
  <div class="warn">
    <h3>빨간 줄이 ch00 이 아니다</h3>
    <p>리본의 빨간 줄은 <strong>1번 도선 = 핀 1 = 5V</strong> 다.
    ch00 은 <strong>셋째 도선</strong>이다. 16핀 커넥터를 쓰던 때와 다르므로
    옛 습관대로 세면 전 채널이 한 칸씩 밀린다.</p>
  </div>
</section>

<section>
  <div class="eyebrow"><span class="num">어댑터</span><h2>만능기판 위에 헤더 두 개</h2></div>
  <p class="dek">두 헤더의 1번 핀을 <strong>같은 열</strong>에 맞춰 꽂는 것이 설계의
  전부다. 그러면 핀 1–18 이 같은 열에 서고, 배선 20가닥 중 18가닥이 그냥
  <strong>수직 직선</strong>이 된다. 남는 2가닥은 반대쪽 끝의 GND 다.</p>
  <div class="plate">
    __PERF__
    <div class="cap">만능기판 90×70&nbsp;mm (35×27홀) 기준 · 50핀 헤더의 이젝터 래치는
    기판 밖으로 나가도 무방하다 (기판에 닿지 않는다)</div>
  </div>
  <p style="margin-top:20px">두 헤더 사이는 <strong>7홀(17.8&nbsp;mm)</strong> 이상
  띄운다. 하우징 깊이가 12&nbsp;mm 남짓이라 더 붙이면 몸통끼리 닿는다.
  배선은 기판 뒷면에서 하고, 앞면은 부품만 둔다.</p>
</section>

<section>
  <div class="eyebrow"><span class="num">배선표</span><h2>20가닥</h2></div>
  <p class="dek">두 커넥터가 같은 규칙(<code>핀 = ch + 3</code>)을 쓰기 때문에
  채널 배선이 전부 같은 번호끼리 만난다. 외울 것이 하나뿐이다.</p>
  <div class="scroll">
    <table class="tbl">
      <thead><tr><th>50핀 (킷)</th><th>20핀 (AFE)</th><th>신호</th><th>배선</th></tr></thead>
      <tbody>__WIRES__</tbody>
    </table>
  </div>
</section>

<section>
  <div class="eyebrow"><span class="num">AFE 쪽</span><h2>16가닥을 어떻게 뽑아내는가</h2></div>
  <p class="dek">아날로그 기판이 할 일은 <strong>비교기 출력 하나를 헤더 핀 하나에
  잇는 것</strong>뿐이다. 특별한 회로가 없다 — 16개의 배선과, 5V 를 받아 LDO 로
  넣는 것과, 접지를 공유하는 것이 전부다.</p>
  <div class="plate">
    __AFESIDE__
    <div class="cap">2×10 박스헤더 · 핀 필드 22.86&nbsp;mm 라 아날로그 기판에
    부담이 없다 · 키홈 필수</div>
  </div>
  <div class="warn">
    <h3>일반 핀헤더로 대체하지 말 것</h3>
    <p>키홈이 없으면 소켓이 180° 뒤집혀 꽂힌다. 그러면 핀 1(5V)이 핀 20(GND)
    자리로 가서 <strong>5V 가 그대로 단락</strong>된다. 몇백 원 차이다.</p>
  </div>
</section>

<section>
  <div class="eyebrow"><span class="num">부품</span><h2>사야 할 것</h2></div>
  <div class="bom">
    <div>
      <span class="pt">A</span><span class="pnm">NT-IDC 50핀 케이블 200 mm</span>
      <span class="pq">×1</span>
      <span class="pd">양끝 IDC 소켓 완제품, 1:1 스트레이트. 압착 공정이 없어진다.</span>
    </div>
    <div>
      <span class="pt">B</span><span class="pnm">박스헤더 2×25 (50P) 스트레이트</span>
      <span class="pq">×1 +여분</span>
      <span class="pd">DS1011-50-S-L-S1-B 계열. 핀 필드 60.96 mm, 래치 포함 82.80 mm.
      어댑터에 실장.</span>
    </div>
    <div>
      <span class="pt">C</span><span class="pnm">박스헤더 2×10 (20P) 스트레이트</span>
      <span class="pq">×2 +여분</span>
      <span class="pd">어댑터에 1개, AFE 기판에 1개. 핀 필드 22.86 mm.</span>
    </div>
    <div>
      <span class="pt">D</span><span class="pnm">IDC 20핀 케이블 (또는 리본+소켓)</span>
      <span class="pq">×1</span>
      <span class="pd">어댑터 ↔ AFE 기판. 짧을수록 좋다.</span>
    </div>
    <div>
      <span class="pt">E</span><span class="pnm">만능기판 90×70 mm</span>
      <span class="pq">×1</span>
      <span class="pd">35×27홀. 50핀 헤더가 25홀을 먹으므로 이보다 작으면 안 된다.</span>
    </div>
    <div>
      <span class="pt">F</span><span class="pnm">주석도금선 또는 UEW 0.3 mm</span>
      <span class="pq">—</span>
      <span class="pd">뒷면 배선 20가닥.</span>
    </div>
  </div>
</section>

<section>
  <div class="eyebrow"><span class="num">순서</span><h2>만들고 확인하기</h2></div>
  <ol class="steps">
    <li><b>어댑터를 먼저 만든다.</b> AFE 기판이 없어도 된다. 헤더 두 개를 꽂고
      모서리 핀만 먼저 납땜해 수평을 확인한 뒤 나머지를 채운다.</li>
    <li><b>도통으로 20가닥을 전부 확인한다.</b> 50핀 3번 ↔ 20핀 3번, … 18↔18,
      49↔19, 50↔20. 그리고 <b>인접 핀끼리 안 붙었는지</b>도 본다 — 납땜 브릿지가
      가장 흔한 고장이고 눈으로는 안 보인다.</li>
    <li><b>VCCO 를 잰다. 이때 AFE 는 물리지 않는다.</b> EXT0–15 을 High 로 구동하는
      비트스트림을 올리고 어댑터의 20핀 헤더에서 전압을 잰다. 그 값이 VCCO 이고,
      16핀이 전부 같은 값이면 핀 맵도 맞은 것이다.</li>
    <li><b>3.3 V 로 나오면 결정을 먼저 한다.</b> 비교기를 3.3 V 로 올릴지
      레벨 변환기를 넣을지 — 아날로그 쪽 판단이다. 그 전에 AFE 를 연결하지 않는다.</li>
    <li><b>AFE 를 물리고 주파수 스윕을 건다.</b> 125 Hz → 5 kHz 를 마이크에 들려주고
      활성 채널이 ch00 에서 ch15 로 <b>단조 이동</b>하는지 본다. 깨끗한 대각선이
      나오면 매핑이 맞은 것이고, 섞였으면 어떻게 섞였는지까지 보인다.</li>
  </ol>
  <div class="warn">
    <h3>3번을 AFE 물린 채로 하면 비교기가 죽는다</h3>
    <p>EXT 핀을 FPGA 출력으로 구동하는 순간 비교기 출력과 맞물린다. 둘 다 푸시풀이라
    출력끼리 싸우고, 킷의 직렬 33 Ω 만으로는 100 mA 가까이 흐른다.
    LPV7215 는 580 nA 마이크로파워 부품이라 그걸 못 견딘다.</p>
  </div>
</section>

<footer>
  <span>근거: docs/hanback_kit.md · docs/ICD.md</span>
  <span>킷 매뉴얼 CH2 94–99쪽</span>
  <span>생성: docs/artifacts/gen_afe_link.py</span>
</footer>

</div>
"""


def main() -> None:
    html = (HTML
            .replace("__CHAIN__", chain())
            .replace("__GRID50__", pin_grid(KIT_PINS))
            .replace("__PERF__", perfboard())
            .replace("__AFESIDE__", afe_side())
            .replace("__WIRES__", wire_table()))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({len(html):,} bytes)")
    # 배선이 20가닥이고 채널이 16개인지 -- 그림이 아니라 숫자로 확인한다
    w = wires()
    assert len(w) == 20, len(w)
    assert sum(1 for _, _, r in w if r == "sig") == N_CH
    assert all(s == d for s, d, _ in w if s <= 18)
    print("배선 20가닥 / 신호 16 / 핀1-18 직결 -- 확인")


if __name__ == "__main__":
    main()
