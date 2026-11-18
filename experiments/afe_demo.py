"""Two presentation screens for the board demo (video, dual monitor), on top of afe_web.

    python -m experiments.afe_web --run runs/bd_setC_leak2 --refs out/afe/grid_leak2 --vthr ...   # board, :8765
    python -m experiments.afe_demo --refs out/afe/grid_leak2 --open                               # screens, :8780

  http://127.0.0.1:8780/     screen 1: this capture's FPGA answer vs the truth, the 16x100 the
                             FPGA got, and the last 10 captures in green (right) / red (wrong).
                             Channel rates, threshold tuning and slot results sit below the fold.
  http://127.0.0.1:8780/cm   screen 2: label x FPGA answer for the whole session, filling in live.

Why a second server instead of another page in afe_web: afe_web owns the SWD poller and the
records, and stays the analysis tool as it is. This one only relays its /state and controls
(choose ARB, V_thr, pause), so both screens and afe_web's own page show the same session, and
closing this changes nothing that is recorded.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from experiments.afe_record import load_ref

NAMES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",   # data/speech_commands.py
         "silence", "unknown"]
RELAY = ("/state", "/play", "/vthr", "/pause", "/target", "/save_record", "/save_png")


def channel_fc(design: Path):
    try:
        return [int(round(v)) for v in np.loadtxt(design, delimiter=",", skiprows=1)[:, 2]]
    except OSError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viewer", default="http://127.0.0.1:8765", help="the running experiments.afe_web")
    ap.add_argument("--refs", default="out/afe/grid_leak2", help="same --refs as afe_web (ARB list)")
    ap.add_argument("--design", default="analog/AFE/artifacts/filterbank_design_board.csv",
                    help="channel centre frequencies for the labels")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--open", action="store_true", help="open both screens in the browser")
    args = ap.parse_args()

    refs = [r for r in (load_ref(p, NAMES) for p in sorted(Path(args.refs).glob("*/ref.json"))) if r]
    cfg = json.dumps({"names": NAMES, "fc": channel_fc(Path(args.design)),
                      "refs": [{k: r[k] for k in ("ref", "kind", "label", "sim_pred")} for r in refs]})
    pages = {"/": LIVE.replace("/*CFG*/null", cfg), "/cm": CM.replace("/*CFG*/null", cfg)}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path in pages:
                body, ctype, code = pages[path].encode(), "text/html; charset=utf-8", 200
            elif path.startswith(RELAY):
                try:
                    with urllib.request.urlopen(args.viewer + self.path, timeout=5) as r:
                        body, ctype, code = r.read(), r.headers.get("Content-Type", "application/json"), 200
                except (urllib.error.URLError, OSError) as e:
                    body = json.dumps({"error": f"afe_web ({args.viewer}) 응답 없음: {e}"}).encode()
                    ctype, code = "application/json", 502
            else:
                self.send_error(404)
                return
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a) -> None:
            pass

    ThreadingHTTPServer.allow_reuse_address = False
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    base = f"http://127.0.0.1:{args.port}"
    print(f"{len(refs)} ARB(s) from {args.refs}; relaying {args.viewer}\n"
          f"  screen 1: {base}/\n  screen 2: {base}/cm\n(Ctrl+C to stop)", flush=True)
    if args.open:
        import webbrowser
        webbrowser.open_new(base + "/")
        webbrowser.open_new(base + "/cm")
    srv.serve_forever()


STYLE = r"""
:root{--bg:#f4f5f7;--card:#fff;--ink:#1d2330;--mute:#6b7280;--line:#e3e6eb;--on:#1f5fbf;--sim:#9aa1ab;
 --ok:#1a7f37;--bad:#d1242f;--warn:#9a6700}
@media (prefers-color-scheme:dark){:root{--bg:#121519;--card:#1c2027;--ink:#e6e8eb;--mute:#9aa1ab;--line:#2c313a;
 --on:#5b9cff;--sim:#5f6672;--ok:#3fb950;--bad:#f85149;--warn:#d29922}}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.4 system-ui,"Malgun Gothic",sans-serif}
.mute{color:var(--mute)}.ok{color:var(--ok)}.bad{color:var(--bad)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;background:var(--bad)}.dot.ok{background:var(--ok)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 18px}
.card h2{font-size:14px;margin:0 0 10px;color:var(--mute);font-weight:600}
button,select,input{font:inherit;padding:3px 8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
button{cursor:pointer}.pausebtn.on{background:var(--bad);border-color:var(--bad);color:#fff}
"""

LIVE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KWS 보드 실시간</title><style>""" + STYLE + r"""
header{padding:10px 20px;display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;border-bottom:1px solid var(--line)}
header h1{font-size:17px;margin:0}
#hero{min-height:calc(100vh - 54px);padding:14px 20px;display:flex;flex-direction:column;gap:14px}
#now{transition:box-shadow .2s}#now.flash{animation:fl 1s ease-out}
@keyframes fl{from{box-shadow:0 0 0 6px color-mix(in srgb,var(--on) 45%,transparent)}to{box-shadow:0 0 0 0 transparent}}
.ans{display:flex;gap:64px;align-items:flex-end;flex-wrap:wrap;margin-bottom:8px}
.ans .k{font-size:16px;color:var(--mute)}.ans b{font-size:64px;line-height:1.05}
#verdict{font-size:64px;line-height:1.05;font-weight:800}
canvas{width:100%;display:block}
#recent{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:10px}
.tile{border:3px solid var(--line);border-radius:10px;padding:6px 6px 8px;background:var(--card);cursor:pointer;text-align:center;min-width:0}
.tile.ok{border-color:var(--ok);background:color-mix(in srgb,var(--ok) 16%,var(--card))}
.tile.bad{border-color:var(--bad);background:color-mix(in srgb,var(--bad) 16%,var(--card))}
.tile.sel{outline:3px solid var(--on);outline-offset:2px}.tile.new{animation:pop .8s ease-out}
@keyframes pop{from{transform:scale(1.08)}to{transform:scale(1)}}
.tile .mk{font-size:30px;font-weight:800;line-height:1}.tile.ok .mk{color:var(--ok)}.tile.bad .mk{color:var(--bad)}
.tile .w{font-size:17px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile .s{font-size:12px;color:var(--mute);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile canvas{height:34px;margin:4px 0}
.below{padding:0 20px 24px;display:flex;flex-direction:column;gap:14px}
.legend span{display:inline-block;width:10px;height:10px;margin:0 4px 0 12px;vertical-align:-1px}
table.res{border-collapse:collapse;width:100%;font-size:13px}.res th,.res td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}
.res th{color:var(--mute);font-weight:600}
@media (max-width:900px){#recent{grid-template-columns:repeat(5,minmax(0,1fr))}.ans b,#verdict{font-size:40px}}
</style></head><body>
<header><h1>AFE → STM32 → FPGA 키워드 인식</h1><span id="conn" class="mute"><span class="dot"></span>연결 중</span>
<label class="mute">재생 중인 ARB <select id="play"><option value="">(고르세요)</option></select></label>
<label class="mute">V_thr (mV) <input id="vthr" size="30"></label>
<button class="pausebtn" id="pause"></button><span id="rect" class="mute"></span></header>
<section id="hero">
<div class="card" id="now"><div class="ans">
<div><div class="k">FPGA 답</div><b id="fp">—</b></div>
<div><div class="k">정답</div><b id="lb">—</b></div>
<div><div class="k">&nbsp;</div><span id="verdict"></span></div></div>
<h2 id="imgt">FPGA 로 들어간 16채널 × 1초 (아래가 저역)</h2><canvas id="img" height="280"></canvas></div>
<div class="card"><h2 id="rct">최근 캡처 10개 (왼쪽이 최신, 누르면 위에 크게)</h2><div id="recent"><span class="mute">아직 없음</span></div></div>
</section>
<section class="below">
<div class="card"><h2>채널별 발화율 <span class="legend"><span style="background:var(--on)"></span>보드 (이 캡처)<span style="background:var(--sim)"></span>소프트웨어 AFE</span></h2><canvas id="rate" height="190"></canvas></div>
<div class="card"><h2 id="tunet">threshold 조정 안내</h2><div id="tunei" class="mute" style="margin-bottom:6px"></div>
<div class="legend mute" style="margin-bottom:4px"><span style="background:var(--sim)"></span>목표 (소프트웨어 AFE)<span style="background:var(--on)"></span>보드 (합산)<span style="background:var(--bad)"></span>무음 발화</div>
<canvas id="tunec" height="190"></canvas><div id="tunetab" style="margin-top:10px"></div></div>
<div class="card"><h2 id="slott">슬롯별 결과</h2><div id="slots" class="mute">ARB 를 고르면 나온다</div></div>
</section>
<script>
const CFG=/*CFG*/null;let S=null,sel=null,lastKey=null;const SEEN=new Map();
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const bits=w=>w.map(x=>Array.from({length:16},(_,k)=>(x>>k)&1));
const chLabel=k=>CFG.fc?`ch${k} ${CFG.fc[k]>=1000?(CFG.fc[k]/1000).toFixed(1)+'k':CFG.fc[k]}`:`ch${k}`;
function setup(cv){const d=devicePixelRatio||1,H=+(cv.dataset.h||(cv.dataset.h=cv.getAttribute('height'))),W=cv.clientWidth;
 cv.style.height=H+'px';cv.width=W*d;cv.height=H*d;const g=cv.getContext('2d');g.setTransform(d,0,0,d,0,0);return [g,W,H]}
function drawImg(cv,words,col,labels=true){const [g,W,H]=setup(cv);const L=labels?70:0,B=labels?18:0,cw=(W-L)/100,ch=(H-B)/16;
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
const playSel=document.getElementById('play'),vthrIn=document.getElementById('vthr'),pauseB=document.getElementById('pause');
CFG.refs.forEach(r=>{const o=document.createElement('option');o.value=r.ref;o.textContent=`${r.ref} — ${r.label}`;playSel.appendChild(o)});
playSel.onchange=async()=>{await fetch('/play?ref='+encodeURIComponent(playSel.value));sel=null;tick()};
vthrIn.onchange=async()=>{await fetch('/vthr?v='+encodeURIComponent(vthrIn.value));tick()};
pauseB.onclick=async()=>{await fetch('/pause?on='+(S&&S.paused?0:1));tick()};
const cls=c=>c.label?(c.fpga===c.label?'ok':'bad'):'';
let recIdx=null;
function remember(){const idx=S.rec&&S.rec.idx||0;
 // a new record (another ARB, or "new record" on the same one) starts the strip empty
 if(recIdx!==null&&idx!==recIdx){SEEN.clear();sel=null;lastKey=null}recIdx=idx;
 for(const c of S.caps){const key=idx+':'+c.n;SEEN.set(key,{...c,key,order:idx*1e6+c.n})}
 if(SEEN.size>60){const ks=[...SEEN.values()].sort((a,b)=>a.order-b.order).slice(0,SEEN.size-60);ks.forEach(x=>SEEN.delete(x.key))}}
function render(){if(!S)return;const st=S.status||{},c=document.getElementById('conn');
 c.innerHTML=`<span class="dot ${st.ok?'ok':''}"></span>`+(S.error?S.error:st.ok?`연결됨 · ${st.updated}`:(st.msg||'연결 중'));
 if(S.error)return;
 if(document.activeElement!==playSel)playSel.value=S.playing||'';
 if(document.activeElement!==vthrIn)vthrIn.value=S.vthr||'';
 pauseB.textContent=S.paused?'▶ 집계 재개':'⏸ 집계 일시 중지';pauseB.className='pausebtn'+(S.paused?' on':'');
 const R=S.rec;document.getElementById('rect').textContent=S.playing&&R?`기록 ${String(R.idx).padStart(2,'0')} · 채점 캡처 ${R.result.n}${R.target?' / '+R.target:''}${R.done?' · 목표 도달':''}`:'';
 remember();const recent=[...SEEN.values()].sort((a,b)=>b.order-a.order).slice(0,10);
 const cur=recent.find(x=>x.key===sel)||recent[0];
 if(cur){const fp=document.getElementById('fp'),lb=document.getElementById('lb'),vd=document.getElementById('verdict'),k=cls(cur);
  fp.textContent=cur.fpga;fp.className=k;lb.textContent=cur.label||'—';
  vd.textContent=cur.label?(k==='ok'?'✓ 정답':'✗ 오답'):'(집계 제외)';vd.className=k||'mute';vd.style.fontSize=cur.label?'':'24px';
  document.getElementById('imgt').textContent=`캡처 #${cur.n} (${cur.time})${cur.playing?' · '+cur.playing:''} — FPGA 로 들어간 16채널 × 1초 (아래가 저역)`;
  drawImg(document.getElementById('img'),cur.clip,css('--on'));
  if(cur.key!==lastKey){lastKey=cur.key;const n=document.getElementById('now');n.classList.remove('flash');void n.offsetWidth;n.classList.add('flash')}
  drawRate(document.getElementById('rate'),cur.clip,cur.sim)}
 else{['fp','lb'].forEach(id=>{const e=document.getElementById(id);e.textContent='—';e.className=''});
  const vd=document.getElementById('verdict');vd.textContent='';vd.className='';
  document.getElementById('imgt').textContent=S.playing?`${S.playing} — 재생을 기다리는 중`:'재생 중인 ARB 를 고르세요';
  drawImg(document.getElementById('img'),null,css('--on'));drawRate(document.getElementById('rate'),null,null)}
 const box=document.getElementById('recent');
 if(!recent.length){box.innerHTML='<span class="mute">아직 없음 — 캡처가 들어오면 여기부터 채워진다</span>';delete box.dataset.first;
  document.getElementById('rct').textContent='최근 캡처 10개 (왼쪽이 최신, 누르면 위에 크게)'}
 if(recent.length){box.innerHTML='';
  recent.forEach((x,i)=>{const d=document.createElement('div'),k=cls(x);d.className='tile '+k+(cur&&x.key===cur.key&&sel?' sel':'')+(i===0&&x.key!==box.dataset.first?' new':'');
   d.innerHTML=`<div class="mk">${x.label?(k==='ok'?'✓':'✗'):'·'}</div><div class="w">${x.fpga}</div><div class="s">정답 ${x.label||'—'}</div>`;
   const cv=document.createElement('canvas');cv.height=34;d.insertBefore(cv,d.children[1]);
   d.onclick=()=>{sel=(sel===x.key?null:x.key);render()};box.appendChild(d);drawImg(cv,x.clip,css('--on'),false)});
  box.dataset.first=recent[0].key;
  const sc=recent.filter(x=>x.label),ok=sc.filter(x=>x.fpga===x.label).length;
  document.getElementById('rct').textContent=`최근 캡처 ${recent.length}개 — 맞음 ${ok} / ${sc.length} (왼쪽이 최신, 누르면 위에 크게)`}
 renderTune();renderSlots()}
const TOL=.03,QMAX=.02,MIN_N=5;
function verdict(n,b,s,q){const d=b-s;
 if(n<MIN_N)return ['mute',`캡처 더 모으기 (${n}/${MIN_N})`];
 if(q>QMAX&&d<-TOL)return ['bad','무음에서 켜지는데 음성은 부족 — threshold 로는 안 됨, 이득·배선 확인'];
 if(q>QMAX)return ['bad','무음에서 켜짐 — Vthr 올리기'];
 if(b<.005&&s>=TOL)return ['bad','꺼져 있음 — Vthr 를 조금씩 내리며 무음 발화 확인'];
 if(d>TOL)return ['',`발화 ${d.toFixed(2)} 줄이기 → Vthr 올리기 ↑`];
 if(d<-TOL)return ['',`발화 ${(-d).toFixed(2)} 늘리기 → Vthr 내리기 ↓`];
 return ['ok','맞음']}
function renderTune(){const T=S.rec&&S.rec.tune;if(!T)return;const n=T.n,qn=T.quiet_n||0;
 const b=T.board.map(x=>n?x/(100*n):0),s=T.sim.map(x=>n?x/(100*n):0),q=(T.quiet||[]).map(x=>qn?x/qn:0);
 document.getElementById('tunei').textContent=S.playing?`기록 ${String(S.rec.idx).padStart(2,'0')} (${S.playing}) · 채점 캡처 ${n}개 · 무음 ${(qn/100).toFixed(1)} s`:'ARB 를 고르면 기록이 시작된다';
 const [g,W,H]=setup(document.getElementById('tunec'));const B=20,bw=W/16,max=Math.max(.3,...b,...s,...q)*1.1;
 g.font='11px system-ui';g.textAlign='center';
 for(let k=0;k<16;k++){const x=k*bw,y=v=>(H-B)*(1-Math.min(v,max)/max);
  g.fillStyle=css('--sim');g.fillRect(x+bw*.12,y(s[k]),bw*.76,H-B-y(s[k]));
  g.fillStyle=css('--on');g.fillRect(x+bw*.3,y(b[k]),bw*.4,H-B-y(b[k]));
  if(qn){g.fillStyle=css('--bad');g.fillRect(x+bw*.08,y(q[k])-1.5,bw*.84,3)}
  g.fillStyle=css('--mute');g.fillText(k,x+bw/2,H-B+4)}
 document.getElementById('tunetab').innerHTML=`<table class="res"><tr><th>채널</th><th>목표</th><th>보드</th><th>보드−목표</th><th>무음 발화</th><th>조정</th></tr>`+
  Array.from({length:16},(_,k)=>{const [cl,txt]=verdict(n,b[k],s[k],q[k]),d=b[k]-s[k];
   return `<tr><td>${chLabel(k)}</td><td>${n?s[k].toFixed(2):'—'}</td><td>${n?b[k].toFixed(2):'—'}</td><td class="${Math.abs(d)>TOL?'bad':''}">${n?(d>=0?'+':'')+d.toFixed(2):'—'}</td><td class="${q[k]>QMAX?'bad':''}">${qn?q[k].toFixed(3):'—'}</td><td class="${cl}">${txt}</td></tr>`}).join('')+'</table>'}
function renderSlots(){const R=S.rec;if(!S.playing||!R||R.kind!=='sequence'||!R.slots)return;
 document.getElementById('slott').textContent=`슬롯별 결과 — ${R.ref}`;
 document.getElementById('slots').innerHTML=`<table class="res"><tr><th>슬롯</th><th>클립</th><th>정답</th><th>보드 캡처</th><th>FPGA 답</th></tr>`+
  R.slots.map(s=>`<tr><td>${s.slot}</td><td>${s.src}</td><td>${s.label}</td><td>${s.n}</td><td>${Object.entries(s.fpga).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`<span class="${k===s.label?'ok':'bad'}">${k} ${v}</span>`).join(', ')||'—'}</td></tr>`).join('')+'</table>'}
async function tick(){try{const r=await fetch('/state');S=await r.json();render()}catch(e){document.getElementById('conn').innerHTML='<span class="dot"></span>서버 꺼짐'}}
setInterval(tick,1000);tick();addEventListener('resize',render);
</script></body></html>
"""

CM = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KWS 혼동행렬</title><style>""" + STYLE + r"""
body{height:100vh;display:flex;flex-direction:column;padding:18px 24px;gap:12px}
.top{display:flex;align-items:baseline;gap:28px;flex-wrap:wrap}
.top h1{font-size:24px;margin:0}#acc{font-size:56px;font-weight:800;line-height:1}#accn{font-size:20px}
#last{font-size:22px;margin-left:auto}#reset{font-size:14px;opacity:.7}#reset:hover{opacity:1}
.wrap{flex:1;min-height:0}
table{border-collapse:separate;border-spacing:4px;width:100%;height:100%;table-layout:fixed}
th{font-size:18px;color:var(--mute);font-weight:600}th.r{text-align:right;padding-right:10px;width:9%}
thead th{height:34px;vertical-align:bottom}
td{border-radius:8px;text-align:center;font-size:28px;font-weight:700;background:color-mix(in srgb,var(--line) 80%,var(--card))}
td.z{color:transparent}td.diag{outline:2px solid color-mix(in srgb,var(--ok) 55%,transparent);outline-offset:-2px}
td.sum{background:none;font-size:20px;font-weight:600;color:var(--mute)}td.pct{background:none;font-size:22px}
td.pop{animation:pop 1.2s ease-out}
@keyframes pop{0%{transform:scale(1.18);box-shadow:0 0 0 6px color-mix(in srgb,var(--on) 55%,transparent)}100%{transform:scale(1);box-shadow:0 0 0 0 transparent}}
tr.hit th{color:var(--ink)}
</style></head><body>
<div class="top"><h1>정답 × FPGA 답</h1><div id="acc" class="ok">—</div><div id="accn" class="mute"></div>
<span id="conn" class="mute"><span class="dot"></span>연결 중</span><div id="last" class="mute"></div>
<button id="reset" title="이 화면만 0 부터 다시 센다. 기록·저장 파일과 afe_web 의 집계는 그대로">↺ 초기화</button></div>
<div class="wrap"><table id="t"></table></div>
<script>
const CFG=/*CFG*/null;let prev=null,lastKey=null,lastS=null;
const ROWS=CFG.names.filter(n=>n!=='silence');                 // ARBs are keywords + unknown
// Reset = remember the session's counts at that moment and show only what came after.
// Kept in localStorage so a refresh keeps it; a new afe_web session (another folder) drops it.
let BASE=null;try{BASE=JSON.parse(localStorage.getItem('kws_cm_base')||'null')}catch(e){}
const capOrder=(S,c)=>((S.rec&&S.rec.idx)||0)*1e6+c.n;
document.getElementById('reset').onclick=()=>{if(!lastS||!confirm('혼동행렬을 0 부터 다시 채울까요?\n(기록·저장 파일은 그대로)'))return;
 const m=(lastS.confusion||{}).m||{},base={};for(const l in m)for(const c in m[l])base[l+'|'+c]=m[l][c];
 const caps=lastS.caps||[],c=caps[caps.length-1];
 BASE={session:lastS.session,base,order:c?capOrder(lastS,c):-1,time:new Date().toTimeString().slice(0,5),dropped:{}};
 try{localStorage.setItem('kws_cm_base',JSON.stringify(BASE))}catch(e){}prev=null;draw(lastS)};
// Shift+click a cell: take out what it holds now (captures recorded under the wrong ARB).
// Later captures in that cell count again; the records and files are not touched.
document.getElementById('t').onclick=e=>{const td=e.target.closest('td[data-k]');if(!td||!e.shiftKey||!lastS)return;
 const v=+td.textContent||0,[l,c]=td.dataset.k.split('|');if(!v)return;
 if(!confirm(`정답 ${l} → FPGA ${c} 의 ${v}개를 혼동행렬에서 뺄까요?\n(기록·저장 파일은 그대로, 이후 캡처는 다시 센다)`))return;
 if(!BASE||BASE.session!==lastS.session)BASE={session:lastS.session,base:{},order:-1,time:null,dropped:{}};
 BASE.base[td.dataset.k]=(BASE.base[td.dataset.k]||0)+v;BASE.dropped=BASE.dropped||{};BASE.dropped[td.dataset.k]=(BASE.dropped[td.dataset.k]||0)+v;
 try{localStorage.setItem('kws_cm_base',JSON.stringify(BASE))}catch(e){}prev=null;draw(lastS)};
function draw(S){lastS=S;const raw=(S.confusion||{m:{}}).m||{},B=BASE&&BASE.session===S.session?BASE:null,m={};
 for(const l in raw){m[l]={};for(const c in raw[l]){const v=raw[l][c]-(B?B.base[l+'|'+c]||0:0);if(v>0)m[l][c]=v}}
 const extra=['timeout','bad'].filter(e=>ROWS.some(l=>(m[l]||{})[e]));const cols=[...CFG.names,...extra];
 let tot=0,ok=0;const now={};
 let h=`<thead><tr><th class="r">정답 ↓ FPGA →</th>${cols.map(c=>`<th>${c}</th>`).join('')}<th>합</th><th>정답률</th></tr></thead><tbody>`;
 for(const l of ROWS){const row=m[l]||{},n=Object.values(row).reduce((a,b)=>a+b,0);tot+=n;ok+=row[l]||0;
  h+=`<tr class="${n?'hit':''}"><th class="r">${l}</th>`;
  for(const c of cols){const v=row[c]||0,key=l+'|'+c;now[key]=v;const a=n?v/n:0,good=c===l;
   const bg=v?`background:color-mix(in srgb,var(${good?'--ok':'--bad'}) ${Math.round(18+62*a)}%,var(--card));color:${a>.45?'#fff':'var(--ink)'}`:'';
   const pop=prev&&v>(prev[key]||0)?' pop':'';
   h+=`<td data-k="${key}" class="${v?'':'z'}${good?' diag':''}${pop}" style="${bg}">${v||0}</td>`}
  h+=`<td class="sum">${n||''}</td><td class="pct ${n?((row[l]||0)/n>=.7?'ok':'bad'):''}">${n?Math.round(100*(row[l]||0)/n)+'%':''}</td></tr>`}
 document.getElementById('t').innerHTML=h+'</tbody>';prev=now;
 document.getElementById('acc').textContent=tot?(Math.round(1000*ok/tot)/10)+'%':'—';
 document.getElementById('accn').textContent=(tot?`${ok} / ${tot} 캡처 맞음`:'아직 채점된 캡처 없음')+(B&&B.time?` · ${B.time} 초기화 이후`:'');
 // what was taken out stays findable on hover over the reset button, not on screen
 const dr=B&&B.dropped?Object.entries(B.dropped).map(([k,v])=>`${k.replace('|','→')} ${v}`).join(', '):'';
 document.getElementById('reset').title='이 화면만 0 부터 다시 센다. 기록·저장 파일과 afe_web 의 집계는 그대로'+(dr?`\n제외한 칸: ${dr}`:'');
 const caps=(S.caps||[]).filter(c=>c.label&&(!B||capOrder(S,c)>B.order)),c=caps[caps.length-1];
 const last=document.getElementById('last');
 if(c){const key=(S.rec&&S.rec.idx)+':'+c.n;const good=c.fpga===c.label;
  last.innerHTML=`방금: 정답 <b>${c.label}</b> → FPGA <b class="${good?'ok':'bad'}">${c.fpga} ${good?'✓':'✗'}</b>`;lastKey=key}
 else last.innerHTML=''}
async function tick(){try{const S=await (await fetch('/state')).json();const st=S.status||{};
  document.getElementById('conn').innerHTML=`<span class="dot ${st.ok?'ok':''}"></span>`+(S.error?S.error:st.ok?'연결됨':(st.msg||'연결 중'));
  if(!S.error)draw(S)}catch(e){document.getElementById('conn').innerHTML='<span class="dot"></span>서버 꺼짐'}}
setInterval(tick,1000);tick();
</script></body></html>
"""

if __name__ == "__main__":
    main()
