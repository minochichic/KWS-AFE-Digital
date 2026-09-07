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
# 킷 J6 의 배치를 AFE 기판이 그대로 받는다: 핀 1,2 = 5V / 핀 3..18 = ch00..ch15 /
# 핀 49,50 = GND. 규칙은 `핀 = ch + 3` 하나다.
#
# 2026-09-07 정정: 한때 여기 20핀 어댑터를 기본으로 두었다. 그럴 이유가 없었다 --
# 16채널 AFE 는 op-amp 만 ~48개짜리 큰 기판이라 82.8 mm 커넥터가 부담이 아니고,
# 어댑터는 손납땜 40군데를 새로 만든다. **직결이 기본**이고 어댑터는 조건부다.
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
    """전체 체인. 직결이 기본이고 어댑터는 조건부라는 것이 이 그림의 요지다."""
    return """
<svg viewBox="0 0 900 300" role="img" aria-label="AFE 기판과 FPGA 킷의 직결 연결">
  <rect class="blk afe" x="20" y="34" width="250" height="132" rx="7"/>
  <text class="bt" x="40" y="62">AFE 기판</text>
  <text class="bs" x="40" y="82">동료 담당</text>
  <text class="bi" x="40" y="110">비교기 ×16</text>
  <text class="bi" x="40" y="130">2×25 박스헤더 (키홈)</text>
  <text class="bi" x="40" y="150">LDO 5V → 1.8 V (+3.3 V)</text>

  <rect class="blk kit" x="630" y="34" width="250" height="132" rx="7"/>
  <text class="bt" x="650" y="62">FPGA 킷</text>
  <text class="bs" x="650" y="82">XC7S75 · 연구실</text>
  <text class="bi" x="650" y="110">J6 Exp. Port</text>
  <text class="bi" x="650" y="130">2×25, 키홈 · 5V·GND 제공</text>
  <text class="bi" x="650" y="150">직렬 33Ω 내장</text>

  <path class="lnk" d="M270 100 H630"/>
  <text class="ll" x="450" y="88" text-anchor="middle">50핀 리본 20cm · 1:1 스트레이트</text>
  <text class="lv" x="450" y="124" text-anchor="middle">사용 18 / 50 · 나머지 30핀은 NC</text>

  <path class="flow sig" d="M630 196 H270"/>
  <text class="fl sig" x="450" y="188" text-anchor="middle">비교기 16가닥 ─ 아날로그에서 FPGA 로</text>
  <path class="flow gndf" d="M270 222 H630"/>
  <text class="fl gndf" x="450" y="214" text-anchor="middle">GND ─ 기준 공유. 대안이 없다</text>
  <path class="flow v5" d="M270 248 H630"/>
  <text class="fl v5" x="450" y="240" text-anchor="middle">5V ─ 킷에서 AFE 로. 기판 LDO 입력 (필수)</text>

  <rect class="ghost" x="352" y="266" width="196" height="26" rx="5"/>
  <text class="gl" x="450" y="284" text-anchor="middle">어댑터는 여기 들어간다 — 조건부</text>
</svg>
"""


def afe_side() -> str:
    """AFE 기판 쪽: 비교기 16개가 어떻게 헤더 한 개로 모이는가.

    좌변의 블록과 우변의 헤더 핀을 **같은 y 에** 두어 배선이 수평 직선이 되게 한다.
    선이 꺾이면 보는 사람이 "왜 꺾였지" 를 먼저 묻는데, 여기엔 이유가 없다.
    """
    rows = [("1·2", "5V → LDO", "v5"), ("3", "ch00", "sig"),
            ("4", "ch01", "sig"), ("5–16", "ch02–ch13", "sig"), ("17", "ch14", "sig"),
            ("18", "ch15", "sig"), ("19–48", "연결 안 함 · 30핀", "nc"),
            ("49·50", "GND", "gnd")]
    step, y0 = 40, 54
    ys = [y0 + i * step for i in range(len(rows))]
    h = ys[-1] + 46

    o = [f'<svg viewBox="0 0 880 {h}" role="img" '
         f'aria-label="AFE 기판에서 비교기 16개를 2x10 헤더로 모으는 방법">']

    def block(y, label, cls="cmp", dashed=False):
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        o.append(f'<rect class="{cls}" x="18" y="{y - 15}" width="188" height="30" '
                 f'rx="4"{dash}/>')
        o.append(f'<text class="ct" x="32" y="{y + 4}">{label}</text>')

    block(ys[0], "LDO → 1.8 V (+3.3 V)")
    block(ys[1], "비교기 ch00")
    block(ys[2], "비교기 ch01")
    block(ys[3], "비교기 ch02 ~ ch13   (12개)", dashed=True)
    block(ys[4], "비교기 ch14")
    block(ys[5], "비교기 ch15")
    o.append(f'<text class="ell" x="112" y="{ys[6] + 4}" text-anchor="middle">'
             f'— 아무것도 잇지 않는다 —</text>')
    block(ys[7], "기판 접지")

    # 헤더
    top, bot = ys[0] - 26, ys[-1] + 26
    o.append(f'<rect class="hdr2" x="470" y="{top}" width="248" '
             f'height="{bot - top}" rx="6"/>')
    o.append(f'<text class="hl2" x="594" y="{top - 10}" text-anchor="middle">'
             f'2×25 박스헤더 → 리본 → 킷 J6</text>')

    for (p, lab, rl), y in zip(rows, ys):
        dash = ' stroke-dasharray="3 4"' if rl == "nc" else ""
        if rl != "nc":
            o.append(f'<path class="w {rl}" d="M206 {y} H500"{dash}/>')
        o.append(f'<circle class="hp {rl}" cx="500" cy="{y}" r="6"/>')
        o.append(f'<text class="hr" x="518" y="{y + 4}">핀 {p}</text>')
        o.append(f'<text class="hv {rl}" x="704" y="{y + 4}" '
                 f'text-anchor="end">{lab}</text>')
    o.append("</svg>")
    return "\n".join(o)


# --- 페이지 ---------------------------------------------------------------- #

def net_table() -> str:
    """AFE 기판이 50핀 헤더에 무엇을 잇는가. 50행이 아니라 4묶음이면 된다."""
    rows = [
        ("1, 2", "5V", "v5", "선택",
         "LDO 입력. 별도 전원을 쓰면 패드만 두고 비운다"),
        (f"3 – {2 + N_CH}", f"ch00 – ch{N_CH - 1:02d}", "sig", "필수",
         f"비교기 출력 {N_CH}개. 핀 = ch + 3"),
        (f"{3 + N_CH} – {KIT_PINS - 2}", "—", "nc", "—",
         f"연결하지 않는다 ({KIT_PINS - 2 - (2 + N_CH)}핀). GND 에도 잇지 말 것"),
        (f"{KIT_PINS - 1}, {KIT_PINS}", "GND", "gnd", "필수",
         "두 기판의 기준 전위. 이게 없으면 1.8 V 가 뜻이 없다"),
    ]
    return "\n".join(
        f'<tr><td class="m">{p}</td>'
        f'<td><span class="tag {rl}">{net}</span></td>'
        f'<td class="m">{need}</td><td class="q">{note}</td></tr>'
        for p, net, rl, need, note in rows)


def ch_chips() -> str:
    return "\n".join(
        f'<div class="chip"><span class="cc">ch{c:02d}</span>'
        f'<span class="cp">핀 {c + 3}</span></div>' for c in range(N_CH))


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
.tag.nc{background:none; color:var(--ink3); border:1px solid var(--nc)}
.scroll{overflow-x:auto}

/* ch <-> 핀 칩 */
.chips{display:grid; grid-template-columns:repeat(auto-fill,minmax(104px,1fr));
  gap:1px; background:var(--rule); border:1px solid var(--rule);
  border-radius:7px; overflow:hidden; margin-top:20px}
.chip{background:var(--surface); padding:9px 12px; display:flex;
  justify-content:space-between; align-items:baseline; gap:8px;
  font-family:'IBM Plex Mono',monospace; font-size:12.5px}
.chip .cc{color:var(--sig); font-weight:500}
.chip .cp{color:var(--ink3)}

.sub-h{font-family:Archivo,sans-serif; font-size:17px; font-weight:600;
  margin:30px 0 12px; color:var(--ink)}
.tree{font-family:'IBM Plex Mono',monospace; font-size:12.5px; line-height:1.75;
  background:var(--surface); border:1px solid var(--rule); border-radius:7px;
  padding:16px 18px; overflow-x:auto; color:var(--ink2); margin:0}

/* 조건부 표시 */
.ghost{fill:none; stroke:var(--ink3); stroke-width:1.2; stroke-dasharray:5 4}
.gl{font-family:'IBM Plex Mono',monospace; font-size:11px; fill:var(--ink3)}
.flow.gndf{stroke:var(--gnd)} .fl.gndf{fill:var(--gnd)}

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
가져오는 물리 설계다. <strong>기판 두 개, 케이블 한 개, 커넥터 두 개</strong>가
전부다 — AFE 기판에 킷과 똑같은 2×25 박스헤더를 하나 올리고 리본으로 잇는다.</p>

<section>
  <div class="eyebrow"><span class="num">체인</span><h2>직결한다</h2></div>
  <p class="dek">신호는 왼쪽으로만 흐른다. FPGA 에서 아날로그로 가는 선은
  <strong>하나도 없다</strong> — sticky OR 은 펄스가 언제 왔는지를 묻지 않아서
  프로토콜이 필요 없다. 그래서 이 경계에는 협상도, 클럭 공유도, 방향 제어도 없다.</p>
  <div class="plate">
    __CHAIN__
    <div class="cap">신호 16 + 5V + GND = 18가닥 · 나머지 30핀은 NC ·
    어댑터는 아래 조건에서만 끼운다</div>
  </div>
  <p style="margin-top:20px">50핀 중 18개만 쓰는 것이 낭비처럼 보이지만, 남는 30핀은
  <strong>패드 30개 값</strong>일 뿐이다. 그걸 줄이겠다고 중간에 변환 기판을 넣으면
  손납땜 40군데와 커넥터 접점 두 벌이 새로 생긴다 — 그쪽이 훨씬 비싸다.</p>
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
  <div class="eyebrow"><span class="num">AFE 쪽</span><h2>16가닥을 어떻게 뽑아내는가</h2></div>
  <p class="dek">아날로그 기판이 할 일은 <strong>비교기 출력 하나를 헤더 핀 하나에
  잇는 것</strong>뿐이다. 회로가 없다 — 배선 16개와 접지 공유가 전부이고,
  나머지 30핀은 아무 데도 안 잇는다.</p>
  <div class="plate">
    __AFESIDE__
    <div class="cap">2×25 박스헤더 · 핀 필드 60.96&nbsp;mm, 래치 포함 82.80&nbsp;mm ·
    점선은 선택 항목 · 키홈 필수</div>
  </div>
  <div class="chips">__CHIPS__</div>
  <div class="warn">
    <h3>일반 핀헤더로 대체하지 말 것</h3>
    <p>키홈이 없으면 소켓이 180° 뒤집혀 꽂힌다. 그러면 핀 1 이 핀 50 자리로 가서
    <strong>5V 와 GND 가 만난다</strong>. 몇백 원 차이다.</p>
  </div>
  <div class="warn">
    <h3>남는 30핀을 GND 에 잇지 말 것</h3>
    <p>접지를 촘촘히 넣으면 좋아 보이지만, 그 핀들은 FPGA I/O 에 물려 있다.
    도구가 미사용 핀을 어떻게 처리하느냐에 따라 <strong>출력이 접지를 때리는</strong>
    구성이 될 수 있다. 얻는 것도 없다 — 에지가 65~80&nbsp;ns 라 크로스토크가
    이 신호의 위험이 아니다. <strong>NC 로 둔다.</strong></p>
  </div>
</section>

<section>
  <div class="eyebrow"><span class="num">배선표</span><h2>네 묶음이면 끝난다</h2></div>
  <p class="dek">50행짜리 표가 필요 없다. 규칙이 <code>핀 = ch + 3</code> 하나이기
  때문이다.</p>
  <div class="scroll">
    <table class="tbl">
      <thead><tr><th>J6 핀</th><th>네트</th><th>필요</th><th>비고</th></tr></thead>
      <tbody>__NETS__</tbody>
    </table>
  </div>
</section>

<section>
  <div class="eyebrow"><span class="num">전원</span><h2>킷의 5V 에서 전부 만든다</h2></div>
  <p class="dek">2026-09-01 에 정해진 것이다 — 아날로그 전원은 FPGA 보드에서 받는다.
  보드가 KC705 에서 킷으로 바뀌면서 소스가 <strong>U103 의 3.3&nbsp;V 에서 J6 핀 1·2 의
  5&nbsp;V</strong> 로 바뀌었을 뿐이고, 헤드룸은 오히려 늘었다.</p>
  <pre class="tree">J6 핀 1·2  (VCC5V) ──리본──→ AFE 기판 ┬─ LDO → 1.8 V   아날로그 · 비교기 · 변환기 A측
                                       └─ LDO → 3.3 V   변환기 B측
J6 핀 49·50 (GND) ──리본──→ 기준 접지</pre>
  <p style="margin-top:16px">로컬 LDO 를 두는 이유는 킷의 5&nbsp;V 가 스위칭
  레귤레이터에서 오기 때문이다. 리플이 리본을 타고 오는데 LDO 한 겹이 PSRR 로 그걸
  깎고 디커플링을 부하 옆으로 가져온다. 고를 때 볼 것은 전류 용량이 아니라
  <strong>출력 잡음과 PSRR</strong> 이다 — 능동 소자 합이 150~280&nbsp;µA 뿐이고,
  가장 약한 ch15 의 스윙이 16&nbsp;mV 라 비교기 오프셋 대비 5.3배밖에 안 된다.</p>
  <p><strong>GND 는 성격이 다르다.</strong> 두 기판이 기준 전위를 공유하지 않으면
  비교기가 내는 "1.8&nbsp;V" 가 FPGA 입장에서 몇 볼트인지 정의되지 않는다.
  전원과 달리 <strong>대안이 없다.</strong></p>
</section>

<section>
  <div class="eyebrow"><span class="num">조건부</span><h2>어댑터가 값을 하는 한 경우</h2></div>
  <p class="dek">어댑터가 필요한 이유는 레벨 변환기 자체가 아니라
  <strong>그 칩을 놓을 자리가 없을 때</strong> 하나뿐이다. AFE 기판이 아직 설계
  중이면 거기 같이 올리면 되고 <strong>어댑터는 사라진다.</strong> 이미 제작된
  기판에 뒤늦게 변환기가 필요해졌을 때가 어댑터의 자리다.</p>
  <table class="tbl" style="margin-bottom:24px">
    <thead><tr><th>VCCO</th><th>선택</th><th>어댑터</th></tr></thead>
    <tbody>
      <tr><td class="m">1.8 V</td><td>그대로 직결. 여유 +460 mV</td>
        <td class="q">필요 없음</td></tr>
      <tr><td class="m">3.3 V</td><td>비교기를 3.3 V 로 올린다 (부품 0개)</td>
        <td class="q">필요 없음</td></tr>
      <tr><td class="m">3.3 V</td><td>'245 · AFE 기판이 아직 설계 중</td>
        <td class="q"><b>필요 없음</b> — 기판에 같이 올린다</td></tr>
      <tr><td class="m">3.3 V</td><td>'245 · AFE 기판이 이미 제작됨</td>
        <td class="q"><b>여기</b></td></tr>
    </tbody>
  </table>
  <p><strong>변환기를 AFE 기판에 올려도 된다.</strong> "아날로그 무손상" 이라는 이 선택의
  이유는 기판이 <em>이미 만들어졌을 때</em> 값을 한다. 아직 설계 중이면 AFE 기판에
  직접 올리는 편이 커넥터 한 벌과 기판 하나를 아낀다. 어댑터는
  <strong>기판이 이미 제작됐을 때의 구제책</strong>이다.</p>

  <h3 class="sub-h">A측은 비교기, B측은 FPGA</h3>
  <p>'245 는 양방향 트랜시버라 두 포트가 대칭이고, 어느 쪽을 A 로 쓸지는 우리가
  정한다. 정하는 기준은 두 가지다 — <strong>제어핀(DIR·OE)이 VCCA 기준</strong>이라
  A 를 1.8&nbsp;V 쪽에 두면 DIR 을 아날로그 기판의 1.8&nbsp;V 레일에 그냥 묶을 수 있고,
  <strong>버스홀드가 A측(데이터 입력)에</strong> 있으므로 A 가 받는 쪽이어야 한다.</p>
  <pre class="tree">비교기 16 ──1.8V──▶ A측 │ '245 │ B측 ──3.3V──▶ 헤더 핀 3–18 ──리본──▶ 킷 J6 ──▶ FPGA
                    VCCA = 1.8 V   VCCB = 3.3 V
                    DIR → 1.8 V (A→B)      OE → GND</pre>
  <h3 class="sub-h">'245 를 넣으면 무엇이 바뀌나</h3>
  <table class="tbl" style="margin-bottom:22px">
    <tbody>
      <tr><td class="m">아날로그 회로</td><td><b>안 바뀐다.</b> 비교기 1.8 V, R7/R8 그대로</td></tr>
      <tr><td class="m">전원</td><td>3.3 V 레일 하나 추가 — <b>디지털 전용</b>, 아날로그에 안 닿는다</td></tr>
      <tr><td class="m">기판</td><td>'245 1개 + 디커플링. DIR→1.8 V, OE→GND 4가닥</td></tr>
      <tr><td class="m">커넥터 · 핀맵</td><td><b>안 바뀐다</b></td></tr>
      <tr><td class="m">RTL · 가중치</td><td><b>안 바뀐다.</b> 경계는 여전히 비교기 16가닥뿐</td></tr>
    </tbody>
  </table>
  <table class="tbl" style="margin-bottom:22px">
    <thead><tr><th>데이터시트 SCES587D</th><th>값</th><th>우리 경우</th></tr></thead>
    <tbody>
      <tr><td class="m">A측 V_IH</td><td class="m">VCCA × 0.65 = 1.17 V</td>
        <td class="q">비교기 최악 VOH 1.63 V → 여유 +460 mV</td></tr>
      <tr><td class="m">B측 VOH @ 100 µA</td><td class="m">VCCB − 0.2 = 3.1 V</td>
        <td class="q">FPGA V_IH 2.0 V → 여유 +1.1 V</td></tr>
      <tr><td class="m">DIR / OE 기준</td><td class="m">VCCA</td>
        <td class="q">DIR→1.8 V, OE→GND · 뱅크 2개라 각 2개</td></tr>
      <tr><td class="m">버스홀드 과구동</td><td class="m">~200 µA @ 1.8 V</td>
        <td class="q">LPV7215 는 500 µA 보장 → 통과. 여유의 40%를 먹는다</td></tr>
      <tr><td class="m">양 포트 입력 내압</td><td class="m">4.6 V (절대최대)</td>
        <td class="q">5 V 레일이 스쳐도 안 죽는다</td></tr>
      <tr><td class="m">Ioff</td><td class="m">±5 µA</td>
        <td class="q">한쪽 VCC 가 GND 면 양 포트 Hi-Z</td></tr>
    </tbody>
  </table>
  <p>버스홀드는 부수 효과가 오히려 좋다. 뒤집으려면 과구동해야 하므로
  <strong>인터페이스에 히스테리시스가 생긴다</strong> — LPV7215 에 내부 히스테리시스가
  없다는 것이 ICD §7 의 9번 걱정이었는데 그 일부를 덮는다.
  다만 <strong>A측에 풀업·풀다운을 달면 안 된다</strong> (데이터시트 명시, 버스홀드와 싸운다).</p>

  <p style="margin-top:20px">어댑터를 만들게 되면 배치는 이렇다. 두 헤더의 1번 핀을
  같은 열에 맞추면 배선이 전부 수직 직선이 된다 (그림은 출력을 20핀으로 줄인
  변형이다 — 변환기를 넣으면 그 자리에 IC 가 들어간다).</p>
  <div class="plate">
    __PERF__
    <div class="cap">만능기판 90×70&nbsp;mm · 위 13홀만 쓴다 ·
    50핀 헤더의 이젝터 래치는 기판 밖으로 나가도 된다 (기판에 닿지 않는다) ·
    두 헤더 사이는 7홀 이상 (하우징 깊이 12&nbsp;mm)</div>
  </div>
</section>

<section>
  <div class="eyebrow"><span class="num">부품</span><h2>사야 할 것</h2></div>
  <p class="dek">직결이면 두 줄로 끝난다.</p>
  <div class="bom">
    <div>
      <span class="pt">A</span><span class="pnm">NT-IDC 50핀 케이블 200 mm</span>
      <span class="pq">×1</span>
      <span class="pd">양끝 IDC 소켓 완제품, 1:1 스트레이트. 압착 공정이 없어진다.
      빨간 줄이 양끝 같은 쪽인지만 확인한다.</span>
    </div>
    <div>
      <span class="pt">B</span><span class="pnm">박스헤더 2×25 (50P) 스트레이트</span>
      <span class="pq">×1 +여분</span>
      <span class="pd">DS1011-50-S-L-S1-B 계열. 핀 필드 60.96 mm, 래치 포함 82.80 mm,
      키홈 4.3 mm, 핀 0.635 각 → 드릴 1.0 mm. <b>AFE 기판에 실장.</b></span>
    </div>
  </div>
  <p style="margin-top:24px; color:var(--ink3); font-size:14.5px">
    아래는 <b>레벨 변환기 경로로 확정됐을 때만</b> 산다. 그 전에 사면 버릴 수 있다.</p>
  <div class="bom" style="opacity:.82">
    <div>
      <span class="pt">C</span><span class="pnm">만능기판 90×70 mm</span>
      <span class="pq">×1</span>
      <span class="pd">35×27홀. 50핀 헤더가 25홀을 먹으므로 이보다 작으면 안 된다.</span>
    </div>
    <div>
      <span class="pt">D</span><span class="pnm">박스헤더 2×25 추가</span>
      <span class="pq">×2</span>
      <span class="pd">어댑터의 입력·출력.</span>
    </div>
    <div>
      <span class="pt">E</span><span class="pnm">SN74AVCH16T245 (TSSOP-48)</span>
      <span class="pq">×1</span>
      <span class="pd">16채널 1.8 ↔ 3.3 V. DIR = H, OE = L, 뱅크 2개라 각 2개씩.</span>
    </div>
  </div>
</section>

<section>
  <div class="eyebrow"><span class="num">순서</span><h2>만들고 확인하기</h2></div>
  <ol class="steps">
    <li><b>VCCO 부터 잰다. AFE 는 아직 만들지도, 물리지도 않는다.</b>
      EXT0–15 를 High 로 구동하는 비트스트림을 올리고 리본 끝에서 전압을 잰다.
      그 값이 VCCO 다. 16가닥이 전부 같은 값이면 핀 맵도 같이 확인된 것이다.</li>
    <li><b>그 값으로 회로를 정한다.</b> 1.8 V 면 그대로. 3.3 V 면 비교기를 올릴지
      변환기를 넣을지 — 아날로그 쪽 판단이고, R7/R8 분압이 비교기 레일 기준인지에
      달렸다. 이 답이 나오기 전에는 부품을 더 사지 않는다.</li>
    <li><b>AFE 기판에 헤더를 실장한다.</b> 모서리 핀 두 개만 먼저 납땜해 수평과
      키홈 방향을 확인한 뒤 나머지 48개를 채운다. 50개를 다 하고 나면 못 고친다.</li>
    <li><b>도통으로 18가닥을 확인한다.</b> 핀 3 ↔ ch00 … 핀 18 ↔ ch15, 그리고
      49·50 ↔ 접지. <b>인접 핀끼리 안 붙었는지</b>도 같이 본다 — 납땜 브릿지가 가장
      흔한 고장이고 눈으로는 안 보인다.</li>
    <li><b>물리고 주파수 스윕을 건다.</b> 125 Hz → 5 kHz 를 마이크에 들려주고 활성
      채널이 ch00 에서 ch15 로 <b>단조 이동</b>하는지 본다. 깨끗한 대각선이 나오면
      매핑이 맞은 것이고, 섞였으면 어떻게 섞였는지까지 보인다. 건너뛰면 안 되는
      이유: 채널이 섞였을 때 증상이 "정확도가 좀 낮다" 뿐이다.</li>
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
            .replace("__NETS__", net_table())
            .replace("__CHIPS__", ch_chips()))
    assert "__" not in html.split("<style>")[0] + html.split("</style>")[-1], \
        "치환 안 된 자리표시자가 남았다"
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({len(html):,} bytes)")

    # 그림이 아니라 숫자로 확인한다.
    roles = [role(p, KIT_PINS) for p in range(1, KIT_PINS + 1)]
    assert roles.count("sig") == N_CH, roles.count("sig")
    assert roles.count("v5") == 2 and roles.count("gnd") == 2
    assert roles.count("nc") == KIT_PINS - N_CH - 4
    assert [ch_of(c + 3) for c in range(N_CH)] == list(range(N_CH))
    print(f"신호 {N_CH} / 5V 2 / GND 2 / NC {roles.count('nc')} "
          f"= {KIT_PINS} · 핀 = ch + 3 -- 확인")


if __name__ == "__main__":
    main()
