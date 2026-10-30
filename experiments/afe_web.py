"""Browser view of the board's AFE captures, live (wake plan step 4).

A small local server polls the STM32 (-D WAKE_AFELOG build) over SWD, like
tools/read_stats.ps1 -Afe, runs best.pt on every new capture, and serves a page
at http://127.0.0.1:8765 that redraws once a second:

    the 16x100 image the STM32 sent to the FPGA (channel k = PBk = bit k,
    labelled with the filter's centre frequency), the FPGA's answer and
    best.pt's answer on the same 100 frames, per-channel firing rate against
    the software AFE (--ref), a tally, recent captures, the board counters

The board keeps only BOARD_MAX_CAPS (16) captures; once all are read the
server clears the log (board_log_reset) so it keeps going. Every capture is
appended to out/afe/live/*.jsonl.

    python -m experiments.afe_web --run runs/bd_base --ref out/afe/54aecbd5_nohash_1/ref.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from experiments.afe_clip import bits_of, classify, load

ELF = "firmware/stm32_wake/kws_wake/Debug/kws_wake.elf"
MAX_CAPS = 16
CAP_WORDS = 52                      # board_cap_t: end, got | bg_q << 16, clip[100] as 50 words
KEEP = 32                           # captures the page can show


def find_tool(name: str) -> str:
    hits = glob.glob(f"C:/ST/**/{name}", recursive=True)
    if not hits:
        raise SystemExit(f"{name} not found under C:/ST -- pass it explicitly")
    return hits[0]


class Board:
    """board_stats / board_caps / board_quiet over SWD (STM32_Programmer_CLI, hot-plug)."""

    def __init__(self, elf: str, cli: Optional[str], nm: Optional[str]) -> None:
        self.cli = cli or find_tool("STM32_Programmer_CLI.exe")
        out = subprocess.run([nm or find_tool("arm-none-eabi-nm.exe"), "-S", elf],
                             capture_output=True, text=True, check=True).stdout
        self.sym: Dict[str, tuple] = {}
        for line in out.splitlines():
            f = line.split()
            if len(f) == 4:
                self.sym[f[3]] = (int(f[0], 16), int(f[1], 16))
        for need in ("board_stats", "board_cap_n", "board_caps", "board_quiet", "board_log_reset"):
            if need not in self.sym:
                raise SystemExit(f"{need} not in {elf} -- is this a WAKE_AFELOG build?")

    def read(self, addr: int, nbytes: int, tries: int = 20) -> Optional[List[int]]:
        for _ in range(tries):
            r = subprocess.run([self.cli, "-c", "port=SWD", "mode=HOTPLUG", "-r32", hex(addr), str(nbytes)],
                               capture_output=True, text=True)
            words = [int(x, 16) for l in r.stdout.splitlines() if re.match(r"^0x2000", l)
                     for x in l.split(":", 1)[1].split()]
            if len(words) == nbytes // 4:
                return words
        return None

    def stats(self) -> Optional[List[int]]:
        a, n = self.sym["board_stats"]
        return self.read(a, n)

    def cap_n(self) -> Optional[int]:
        w = self.read(self.sym["board_cap_n"][0], 4)
        return None if w is None else w[0]

    def caps(self, lo: int, hi: int) -> List[dict]:
        v = self.read(self.sym["board_caps"][0] + lo * CAP_WORDS * 4, (hi - lo) * CAP_WORDS * 4)
        out = []
        for c in range(hi - lo if v else 0):
            b = c * CAP_WORDS
            clip = [x for w in v[b + 2:b + CAP_WORDS] for x in (w & 0xFFFF, w >> 16)]
            out.append({"end": v[b], "got": v[b + 1] & 0xFF, "bg_q": v[b + 1] >> 16, "clip": clip})
        return out

    def quiet(self) -> Optional[List[int]]:
        return self.read(self.sym["board_quiet"][0], 68)

    def reset(self) -> None:
        subprocess.run([self.cli, "-c", "port=SWD", "mode=HOTPLUG", "-w32",
                        hex(self.sym["board_log_reset"][0]), "0x1"], capture_output=True, text=True)


class Playing:
    """Which reference clip is being played, as told by the page (the person running the AWG).

    Picking the clip from the board image itself does not work: against the 50 ARB refs the
    real AFE's image overlaps the right one about as well as several wrong ones (2026-10-30,
    rig00034 was matched to other clips), so the clip is named by hand. Captures within
    SETTLE seconds of a change may still be the previous waveform and are not scored."""

    SETTLE = 2.0

    def __init__(self, paths: List[Path], names: List[str]) -> None:
        self.refs: Dict[str, dict] = {}
        for p in paths:
            ref = json.loads(p.read_text())
            caps = ref["gains"]["0"]["captures"]
            if caps:
                self.refs[p.parent.name] = {"ref": p.parent.name, "label": names[ref["label"]],
                                            "sim_pred": names[caps[0]["pred"]], "sim": caps[0]["window"]}
        self.current: Optional[str] = next(iter(self.refs)) if len(self.refs) == 1 else None
        self.since = 0.0

    def set(self, name: Optional[str]) -> None:
        self.current = name if name in self.refs else None
        self.since = time.time()

    def tag(self) -> dict:
        if self.current is None:
            return {}
        r = self.refs[self.current]
        settled = time.time() - self.since >= self.SETTLE
        return {"ref": r["ref"] if settled else None, "label": r["label"], "sim_pred": r["sim_pred"],
                "sim": r["sim"], "playing": r["ref"]}


class Poller(threading.Thread):
    def __init__(self, board: Board, model, names: List[str], period: float, log_path: Path,
                 playing: Playing) -> None:
        super().__init__(daemon=True)
        self.board, self.model, self.names, self.period, self.log_path = board, model, names, period, log_path
        self.playing = playing
        self.lock = threading.Lock()
        self.caps: List[dict] = []
        self.tally: Counter = Counter()
        self.per_ref: Dict[str, dict] = {}
        self.total = self.agree = 0
        self.status: dict = {"ok": False, "msg": "connecting..."}

    def run(self) -> None:
        seen = 0
        while True:
            try:
                seen = self.poll(seen)
            except Exception as e:                  # keep serving; show the error on the page
                with self.lock:
                    self.status = {"ok": False, "msg": f"{type(e).__name__}: {e}"}
            time.sleep(self.period)

    def poll(self, seen: int) -> int:
        st = self.board.stats()
        n = self.board.cap_n()
        if st is None or n is None:
            with self.lock:
                self.status = {"ok": False, "msg": "SWD read failed (board off? CubeIDE debug session open?)"}
            return seen
        if n < seen:                                # someone else cleared the log
            seen = 0
        new = self.board.caps(seen, n) if n > seen else []
        if new:
            bits = np.stack([bits_of(c["clip"]) for c in new])
            pred, prob = classify(self.model, bits)
            tag = self.playing.tag()
            with self.lock, self.log_path.open("a") as f:
                for c, p, q in zip(new, pred, prob):
                    m = tag
                    g = c["got"]
                    fpga = self.names[g] if g < len(self.names) else ("timeout" if g == 0xFF else "bad")
                    self.total += 1
                    self.agree += int(g == p)
                    self.tally[fpga] += 1
                    rec = {**c, **m, "n": self.total, "fpga": fpga, "best_pt": self.names[int(p)],
                           "prob": round(float(q), 3), "time": time.strftime("%H:%M:%S")}
                    self.caps = (self.caps + [rec])[-KEEP:]
                    f.write(json.dumps({k: v for k, v in rec.items() if k != "sim"}) + "\n")
                    if m.get("ref"):
                        pr = self.per_ref.setdefault(m["ref"], {"ref": m["ref"], "label": m["label"],
                                                                "sim_pred": m["sim_pred"], "n": 0,
                                                                "answered": 0, "fpga": Counter()})
                        pr["n"] += 1
                        if fpga in self.names:
                            pr["answered"] += 1
                            pr["fpga"][fpga] += 1
            seen = n
        if n >= MAX_CAPS and seen >= MAX_CAPS:      # the board stops logging at 16; start over
            self.board.reset()
            seen = 0
        q = self.board.quiet()
        with self.lock:
            self.status = {"ok": True, "msg": "", "ticks": st[0], "clips": st[2], "ok_n": st[3],
                           "timeouts": st[4], "bad": st[5], "bg": int(np.int32(np.uint32(st[10]))) / 32,
                           "quiet_s": (q[0] / 100) if q else 0,
                           "quiet_rate": [o / q[0] for o in q[1:17]] if q and q[0] else None,
                           "updated": time.strftime("%H:%M:%S")}
        return seen

    def snapshot(self) -> dict:
        with self.lock:
            per = [{**v, "fpga": dict(v["fpga"]),
                    "correct": v["fpga"].get(v["label"], 0)} for _, v in sorted(self.per_ref.items())]
            return {"status": self.status, "caps": self.caps, "tally": dict(self.tally),
                    "total": self.total, "agree": self.agree, "per_ref": per,
                    "playing": self.playing.current}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/bd_base", help="dir with best.pt + config.yaml")
    ap.add_argument("--ref", default=None, help="ref.json from experiments.afe_clip prep (sim overlay)")
    ap.add_argument("--refs", default="out/afe", help="dir of <clip>/ref.json: the page lists them "
                                                     "so you can say which one is playing")
    ap.add_argument("--elf", default=ELF)
    ap.add_argument("--cli", default=None, help="STM32_Programmer_CLI.exe (default: search C:/ST)")
    ap.add_argument("--nm", default=None, help="arm-none-eabi-nm.exe (default: search C:/ST)")
    ap.add_argument("--period", type=float, default=1.0, help="seconds between polls")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--save", default="out/afe/live", help="every capture is appended here as JSON lines")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    try:                                        # already running: two pollers would fight over SWD
        import urllib.request
        urllib.request.urlopen(url + "/state", timeout=1).read()
        print(f"a viewer is already running at {url} -- opening it")
        if args.open:
            import webbrowser
            webbrowser.open(url)
        return
    except OSError:
        pass

    from data.speech_commands import KEYWORDS

    names = KEYWORDS + ["silence", "unknown"]
    model, afe = load(Path(args.run))
    # channel k's centre frequency, from the filter bank the software AFE was trained with
    fb = afe.spice_fbank.numpy() if hasattr(afe, "spice_fbank") else None
    fc = [round(float(i) * 8000 / (fb.shape[1] - 1)) for i in fb.argmax(1)] if fb is not None else None
    save = Path(args.save)
    save.mkdir(parents=True, exist_ok=True)
    ref_paths = [Path(args.ref)] if args.ref else sorted(Path(args.refs).glob("*/ref.json"))
    playing = Playing(ref_paths, names)
    print(f"{len(playing.refs)} reference clip(s)", flush=True)
    page_cfg = {"names": names, "fc": fc, "refs": list(playing.refs.values())}
    poller = Poller(Board(args.elf, args.cli, args.nm), model, names, args.period,
                    save / f"live_{time.strftime('%Y%m%d_%H%M%S')}.jsonl", playing)
    poller.start()
    page = PAGE.replace("/*CFG*/null", json.dumps(page_cfg))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/state"):
                body, ctype = json.dumps(poller.snapshot()).encode(), "application/json"
            elif self.path.startswith("/play"):              # /play?ref=<clip>, empty = nothing
                from urllib.parse import parse_qs, urlparse
                playing.set((parse_qs(urlparse(self.path).query).get("ref") or [None])[0])
                body, ctype = json.dumps({"playing": playing.current}).encode(), "application/json"
            elif self.path in ("/", "/index.html"):
                body, ctype = page.encode(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a) -> None:
            pass

    ThreadingHTTPServer.allow_reuse_address = False     # Windows would let a second copy bind too
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"open {url}  (Ctrl+C to stop; captures -> {poller.log_path})", flush=True)
    if args.open:
        import webbrowser
        webbrowser.open(url)
    srv.serve_forever()


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AFE 실시간 보기</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--mute:#6b7280;--line:#e3e6eb;--on:#1f5fbf;--sim:#9aa1ab;--ok:#1a7f37;--bad:#c2410c}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1d2128;--ink:#e6e8eb;--mute:#9aa1ab;--line:#2c313a;--on:#5b9cff;--sim:#5f6672;--ok:#3fb950;--bad:#f0883e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,"Malgun Gothic",sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:8px 20px;align-items:baseline}
h1{font-size:17px;margin:0}.mute{color:var(--mute)}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;background:var(--bad)}
.dot.ok{background:var(--ok)}main{padding:16px 20px;display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card h2{font-size:13px;margin:0 0 10px;color:var(--mute);font-weight:600}
.ans{display:flex;gap:28px;flex-wrap:wrap;margin-bottom:10px}.ans b{font-size:26px}.ans .k{font-size:12px;color:var(--mute)}
canvas{width:100%;display:block}.row{display:flex;justify-content:space-between;margin:3px 0}
.bar{height:16px;background:var(--on);border-radius:3px}.tl{display:flex;align-items:center;gap:8px;margin:4px 0}
.tl span{width:62px;text-align:right}.tl .n{width:auto;color:var(--mute)}
#thumbs{display:flex;flex-wrap:wrap;gap:8px}.th{cursor:pointer;border:2px solid transparent;border-radius:6px;padding:3px;background:var(--bg)}
.th.sel{border-color:var(--on)}.th div{font-size:11px;text-align:center}.ok{color:var(--ok)}.badc{color:var(--bad)}
.legend span{display:inline-block;width:10px;height:10px;margin:0 4px 0 12px;vertical-align:-1px}
.full{grid-column:1/-1}table.res{border-collapse:collapse;width:100%;font-size:13px;color:var(--ink)}.res th,.res td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}.res th{color:var(--mute);font-weight:600}select{font:inherit;padding:2px 4px}@media (max-width:860px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>AFE → STM32 → FPGA 실시간</h1><span id="conn" class="mute"><span class="dot"></span>연결 중</span>
<label>재생 중인 ARB <select id="play"><option value="">(고르세요)</option></select></label>
<span id="cnt" class="mute"></span></header>
<main>
<section class="card"><div class="ans">
<div><div class="k">캡처</div><b id="cn">—</b></div>
<div><div class="k">FPGA 답</div><b id="fp">—</b></div>
<div><div class="k">best.pt (같은 100칸)</div><b id="bp">—</b></div>
<div><div class="k">정답</div><b id="lb">—</b></div>
<div><div class="k">소프트웨어 AFE 예측</div><b id="sp">—</b></div></div>
<h2 id="imgt">STM32 가 FPGA 로 보낸 16×100 (진할수록 1, 아래가 저역)</h2><canvas id="img" height="300"></canvas></section>
<section class="card"><h2>FPGA 답 누계 (전체)</h2><div id="tally" class="mute">아직 없음</div>
<h2 style="margin-top:16px">FPGA = best.pt</h2><div id="agree" style="font-size:22px">—</div>
<h2 style="margin-top:16px">보드</h2><div id="board" class="mute"></div></section>
<section class="card"><h2>채널별 발화율 <span class="legend"><span style="background:var(--on)"></span>보드 (이 캡처)<span style="background:var(--sim)"></span>소프트웨어 AFE</span></h2><canvas id="rate" height="190"></canvas></section>
<section class="card"><h2 id="simt">소프트웨어 AFE 기준</h2><canvas id="sim" height="190"></canvas></section>
<section class="card full"><h2>최근 캡처 (누르면 위에 크게)</h2><div id="thumbs"></div></section>
<section class="card full"><h2 id="sumt">ARB 별 결과</h2><div id="sum" class="mute">재생 중인 ARB 를 고르면 그 뒤 캡처가 여기 쌓인다 (바꾼 직후 2 초는 빼고)</div></section>
</main>
<script>
const CFG=/*CFG*/null;let S=null,sel=null;
const REF=Object.fromEntries(CFG.refs.map(r=>[r.ref,r]));
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const bits=w=>w.map(x=>Array.from({length:16},(_,k)=>(x>>k)&1));
function chLabel(k){return CFG.fc?`ch${k} ${CFG.fc[k]>=1000?(CFG.fc[k]/1000).toFixed(1)+'k':CFG.fc[k]}`:`ch${k}`}
function setup(cv){const d=devicePixelRatio||1,H=+(cv.dataset.h||(cv.dataset.h=cv.getAttribute('height'))),W=cv.clientWidth;
 cv.style.height=H+'px';cv.width=W*d;cv.height=H*d;const g=cv.getContext('2d');g.setTransform(d,0,0,d,0,0);return [g,W,H]}
function drawImg(cv,words,col,labels=true){const [g,W,H]=setup(cv);const L=labels?66:0,B=labels?18:0,cw=(W-L)/100,ch=(H-B)/16;
 g.fillStyle=css('--bg');g.fillRect(L,0,W-L,H-B);if(!words)return;const b=bits(words);g.fillStyle=col;
 for(let t=0;t<100;t++)for(let k=0;k<16;k++)if(b[t][k])g.fillRect(L+t*cw,(15-k)*ch,Math.ceil(cw),Math.ceil(ch));
 if(!labels)return;g.fillStyle=css('--mute');g.font='11px system-ui';g.textAlign='right';g.textBaseline='middle';
 for(let k=0;k<16;k++)g.fillText(chLabel(k),L-6,(15-k+.5)*ch);g.textAlign='center';g.textBaseline='top';
 for(let t=0;t<=100;t+=20)g.fillText(t*10+' ms',Math.min(L+t*cw,W-18),H-B+3)}
const rates=w=>{const b=bits(w);return Array.from({length:16},(_,k)=>b.reduce((s,f)=>s+f[k],0)/100)};
function drawRate(cv,words,simw){const [g,W,H]=setup(cv);const B=20,bw=W/16,max=.6;
 const bo=words?rates(words):null,si=simw?rates(simw):null;g.font='11px system-ui';g.textAlign='center';
 for(let k=0;k<16;k++){const x=k*bw;if(si){g.fillStyle=css('--sim');const h=(H-B)*Math.min(si[k],max)/max;g.fillRect(x+bw*.12,H-B-h,bw*.76,h)}
  if(bo){g.fillStyle=css('--on');const h=(H-B)*Math.min(bo[k],max)/max;g.fillRect(x+bw*.3,H-B-h,bw*.4,h)}
  g.fillStyle=css('--mute');g.fillText(k,x+bw/2,H-B+4)}}
const playSel=document.getElementById('play');
CFG.refs.forEach(r=>{const o=document.createElement('option');o.value=r.ref;o.textContent=`${r.ref} — ${r.label} (시뮬: ${r.sim_pred})`;playSel.appendChild(o)});
playSel.onchange=async()=>{await fetch('/play?ref='+encodeURIComponent(playSel.value));sel=null;tick()};
const esc=s=>String(s??'—');
function render(){if(!S)return;const st=S.status;const c=document.getElementById('conn');
 c.innerHTML=`<span class="dot ${st.ok?'ok':''}"></span>`+(st.ok?`연결됨 · ${st.updated}`:st.msg);
 if(document.activeElement!==playSel)playSel.value=S.playing||'';
 if(st.ok){document.getElementById('cnt').textContent=`보낸 클립 ${st.clips} · 응답 ${st.ok_n} · 타임아웃 ${st.timeouts} · 배경 ${st.bg.toFixed(2)}채널`;
  document.getElementById('board').innerHTML=`무음 ${st.quiet_s.toFixed(1)} s 동안 발화율<br>`+(st.quiet_rate?st.quiet_rate.map((r,k)=>`ch${k} ${r.toFixed(2)}`).join(' · '):'—')}
 const caps=S.caps,cur=caps.find(x=>x.n===sel)||caps[caps.length-1],pr=S.playing?REF[S.playing]:null;
 const label=cur?cur.label:(pr&&pr.label),simw=cur&&cur.sim?cur.sim:(pr&&pr.sim),simRef=cur&&cur.playing?cur.playing:(pr&&pr.ref);
 if(cur){const good=label&&cur.fpga===label;document.getElementById('cn').textContent='#'+cur.n;
  const fp=document.getElementById('fp');fp.textContent=cur.fpga;fp.className=label?(good?'ok':'badc'):'';
  document.getElementById('bp').textContent=`${cur.best_pt} ${cur.prob.toFixed(2)}`;
  document.getElementById('imgt').textContent=`캡처 #${cur.n} (${cur.time})${cur.playing?' · '+cur.playing+(cur.ref?'':' (바꾼 직후, 집계 제외)'):''} — STM32 가 FPGA 로 보낸 16×100, 아래가 저역`;
  drawImg(document.getElementById('img'),cur.clip,css('--on'))}
 document.getElementById('lb').textContent=esc(label);document.getElementById('sp').textContent=esc(cur&&cur.sim_pred||(pr&&pr.sim_pred));
 drawRate(document.getElementById('rate'),cur?cur.clip:null,simw);
 document.getElementById('simt').textContent=simRef?`소프트웨어 AFE 기준 — ${simRef} (보드와 같은 음량 추정)`:'소프트웨어 AFE 기준 — 재생 중인 ARB 를 고르세요';
 drawImg(document.getElementById('sim'),simw,css('--ink'));
 const T=S.tally,ks=Object.keys(T).sort((a,b)=>T[b]-T[a]),mx=Math.max(1,...Object.values(T));
 document.getElementById('tally').innerHTML=ks.length?ks.map(k=>`<div class="tl"><span>${k}</span><div class="bar" style="width:${150*T[k]/mx}px"></div><span class="n">${T[k]}</span></div>`).join(''):'아직 없음';
 document.getElementById('agree').textContent=S.total?`${S.agree} / ${S.total}`:'—';
 const th=document.getElementById('thumbs');th.innerHTML='';[...caps].reverse().slice(0,16).forEach(x=>{const d=document.createElement('div');
  d.className='th'+(cur&&x.n===cur.n?' sel':'');const cv=document.createElement('canvas');cv.style.width='120px';cv.height=40;d.appendChild(cv);
  const t=document.createElement('div');t.innerHTML=`#${x.n} ${x.playing||''} <span class="${x.label?(x.fpga===x.label?'ok':'badc'):''}">${x.fpga}</span>`;d.appendChild(t);
  d.onclick=()=>{sel=(sel===x.n?null:x.n);render()};th.appendChild(d);drawImg(cv,x.clip,css('--on'),false)});
 const P=S.per_ref;if(P.length){let ans=0,cor=0,maj=0,simok=0,simmaj=0;
  const rows=P.map(r=>{const top=Object.entries(r.fpga).sort((a,b)=>b[1]-a[1]);ans+=r.answered;cor+=r.correct;
   const m=top.length&&top[0][0]===r.label;maj+=m;simok+=r.sim_pred===r.label;simmaj+=(r.sim_pred===r.label)===m;
   return `<tr><td>${r.ref}</td><td>${r.label}</td><td class="${r.sim_pred===r.label?'ok':'badc'}">${r.sim_pred}</td><td>${r.n}</td><td>${r.answered}</td>`+
    `<td>${top.slice(0,3).map(([k,v])=>`<span class="${k===r.label?'ok':''}">${k} ${v}</span>`).join(', ')||'—'}</td><td class="${m?'ok':'badc'}">${r.answered?Math.round(100*r.correct/r.answered)+'%':'—'}</td></tr>`}).join('');
  document.getElementById('sumt').textContent=`ARB 별 결과 — ${P.length}개 · 최다 답이 정답 ${maj}/${P.length} · 캡처 단위 정답 ${cor}/${ans}${ans?' ('+Math.round(100*cor/ans)+'%)':''} · 시뮬 예측과 맞고 틀림이 같음 ${simmaj}/${P.length}`;
  document.getElementById('sum').innerHTML=`<table class="res"><tr><th>ARB</th><th>정답</th><th>시뮬 예측</th><th>캡처</th><th>FPGA 응답</th><th>FPGA 답</th><th>정답률</th></tr>${rows}</table>`}}
async function tick(){try{S=await (await fetch('/state')).json();render()}catch(e){document.getElementById('conn').innerHTML='<span class="dot"></span>서버 꺼짐'}}
setInterval(tick,1000);tick();addEventListener('resize',render);</script></body></html>
"""

if __name__ == "__main__":
    main()
