"""Browser view of the board's AFE captures, live (wake plan step 4).

A small local server polls the STM32 (-D WAKE_AFELOG build) over SWD, like
tools/read_stats.ps1 -Afe, runs best.pt on every new capture, and serves a page
at http://127.0.0.1:8765 that redraws once a second:

    the 16x100 image the STM32 sent to the FPGA (channel k = PBk = bit k,
    labelled with the filter's centre frequency), the FPGA's answer and
    best.pt's answer on the same 100 frames, per-channel firing rate against
    the software AFE (--ref), a tally, recent captures, the board counters,
    and a threshold table: per channel, board vs software on-frames summed over
    the current record plus the board's quiet (IDLE) firing, with which way to
    move Vthr

Picking the playing ARB on the page starts a record for it (experiments.afe_record):
the previous one is written to out/afe/sessions/<run>/<nn>_<arb>/ with the session's
records.csv and confusion matrix. For a sequence ARB (experiments.afe_seq) each capture
is matched to its slot by its frame tick. The board keeps only BOARD_MAX_CAPS (16)
captures; once all are read the server clears the log (board_log_reset) so it keeps
going. Every capture is also appended to out/afe/live/*.jsonl, and each saved record
writes its own captures.jsonl, with the comparator reference voltages typed into the
page (V_thr) -- experiments/score_live.py and thr_calib.py read either. Changing V_thr
starts a new record of the same ARB.

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
from urllib.parse import parse_qs, urlparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from experiments.afe_clip import bits_of, classify, load
from experiments.afe_record import Record, confusion, load_ref, write_record, write_session

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


def parse_vthr(text: str) -> Optional[List[float]]:
    """'948' -> 16 x 948.0; '900,916,...' (16 values) -> those; anything else -> None."""
    vals = [v for v in re.split(r"[,\s]+", text.strip()) if v]
    try:
        mv = [float(v) for v in vals]
    except ValueError:
        return None
    if len(mv) == 1:
        mv = mv * 16
    return mv if len(mv) == 16 else None


class VThr:
    """The comparator reference voltages, as typed on the page by whoever set them.

    Like the ARB name: the board cannot report them, so they are told. Captures within
    SETTLE seconds of a change may predate it and carry no voltages."""

    SETTLE = 2.0

    def __init__(self, text: str = "") -> None:
        self.set(text)

    def set(self, text: str) -> None:
        self.text = text or ""
        self.mv = parse_vthr(self.text) if self.text else None
        self.since = time.time()

    def tag(self) -> dict:
        settled = time.time() - self.since >= self.SETTLE
        return {"vthr": self.text, "vthr_mv": self.mv if (self.mv and settled) else None}


class Poller(threading.Thread):
    """Polls the board and keeps one Record per ARB selection (experiments.afe_record).

    Picking the ARB from the board image itself does not work: against the 50 ARB refs the
    real AFE's image overlaps the right one about as well as several wrong ones (2026-10-30,
    rig00034 was matched to other clips), so the person at the AWG names it on the page.
    Selecting an ARB closes the current record (saved under the session folder) and starts a
    new one; captures in its first SETTLE seconds may still be the previous waveform."""

    def __init__(self, board: Board, model, names: List[str], period: float, log_path: Path,
                 refs: Dict[str, dict], session: Path, fc: Optional[List[int]], vthr: VThr) -> None:
        super().__init__(daemon=True)
        self.board, self.model, self.names, self.period, self.log_path = board, model, names, period, log_path
        self.refs, self.session, self.fc, self.vthr = refs, session, fc, vthr
        self.lock = threading.Lock()
        self.save_lock = threading.Lock()
        self.tally: Counter = Counter()
        self.total = self.agree = 0
        self.status: dict = {"ok": False, "msg": "connecting..."}
        self.want_reset = False
        self.paused = False
        self.q_prev = [0] * 17                      # board_quiet at the last read, to take deltas
        self.target_override: Optional[int] = None
        self.n_rec = 0
        self.records: Dict[int, dict] = {}          # idx -> last view of a closed (or saved) record
        only = next(iter(refs)) if len(refs) == 1 else None
        self.rec = Record(0, None, None)
        if only:
            self.select(only)

    # --- page actions -------------------------------------------------------------------
    def select(self, name: Optional[str], again: bool = False) -> None:
        """ARB picked on the page: close the current record, start a new one (again: same ARB)."""
        with self.lock:
            if not again and name == (self.rec.ref or {}).get("ref"):
                return
            old = self.rec
            ref = self.refs.get(name) if name else None
            if ref:
                self.n_rec += 1
            self.rec = Record(self.n_rec if ref else 0, ref, self.target_override)
            self.paused = False
            self.want_reset = True                  # the poll thread owns SWD; it clears the board log
        self.close(old)

    def set_vthr(self, text: str) -> None:
        """New jumper setting: the same ARB starts a new record, so no record mixes voltages."""
        if text == self.vthr.text:
            return
        self.vthr.set(text)
        name = (self.rec.ref or {}).get("ref")
        if name and self.rec.caps:
            self.select(name, again=True)

    def set_paused(self, paused: bool) -> None:
        """Paused: captures still show and are logged, but are not scored (e.g. while a
        trimmer is being turned)."""
        with self.lock:
            self.paused = paused

    def set_target(self, n: Optional[int]) -> None:
        with self.lock:
            self.target_override = n
            r = self.rec
            r.target = n or (r.ref["default_target"] if r.ref else None)
            r.done = False

    def save_now(self) -> Optional[str]:
        return self.save(self.rec)

    # --- records ------------------------------------------------------------------------
    def close(self, rec: Record) -> None:
        if rec.ref and rec.caps:
            self.save(rec)

    def save(self, rec: Record) -> Optional[str]:
        if not rec.ref:
            return None
        with self.lock:
            view = rec.view(self.names)
        folder = self.session / f"{rec.idx:02d}_{rec.name}"
        view["saved"] = str(folder.resolve())
        with self.save_lock:
            write_record(folder, view, self.fc, self.names)
            with self.lock:
                rec.saved, rec.dirty = view["saved"], False
                self.records[rec.idx] = view
                views = [self.records[k] for k in sorted(self.records)]
            write_session(self.session, views, self.names)
        return view["saved"]

    # --- polling ------------------------------------------------------------------------
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
        new = [] if self.want_reset else (self.board.caps(seen, n) if n > seen else [])
        finished = None
        if new:
            bits = np.stack([bits_of(c["clip"]) for c in new])
            pred, prob = classify(self.model, bits)
            with self.lock, self.log_path.open("a") as f:
                rec = self.rec
                for c, p, q in zip(new, pred, prob):
                    g = c["got"]
                    fpga = self.names[g] if g < len(self.names) else ("timeout" if g == 0xFF else "bad")
                    self.total += 1
                    self.agree += int(g == p)
                    if not self.paused:
                        self.tally[fpga] += 1
                    cap = {**c, "n": self.total, "fpga": fpga, "best_pt": self.names[int(p)],
                           "prob": round(float(q), 3), "time": time.strftime("%H:%M:%S"), "paused": self.paused,
                           **self.vthr.tag()}
                    rec.add(cap)
                # ref/label as the record sees them now (a sequence ARB's first captures are
                # still aligning); the record's own captures.jsonl has the final ones
                vs = rec.view(self.names)["caps"][-len(new):]
                for v in vs:
                    f.write(json.dumps({**{k: x for k, x in v.items() if k != "sim"},
                                        "record": rec.idx, "arb": rec.name}) + "\n")
                if rec.target and not rec.done and rec.scored(self.names) >= rec.target:
                    rec.done = True                 # enough: stop scoring until the next ARB
                    self.paused = True
                    finished = rec
            seen = n
        if finished:
            self.save(finished)
        q = self.board.quiet()                      # before any reset: the reset zeroes it too
        with self.lock:
            if q:
                d = [a - b for a, b in zip(q, self.q_prev)] if q[0] >= self.q_prev[0] else q
                self.q_prev = q
                if not self.paused and not self.want_reset and self.rec.ref:
                    self.rec.add_quiet(d)
            reset = self.want_reset or (n >= MAX_CAPS and seen >= MAX_CAPS)
            self.want_reset = False
        if reset:                                   # the board stops logging at 16; start over
            self.board.reset()
            seen = 0
            with self.lock:
                self.q_prev = [0] * 17
        with self.lock:
            self.status = {"ok": True, "msg": "", "ticks": st[0], "clips": st[2], "ok_n": st[3],
                           "timeouts": st[4], "bad": st[5], "bg": int(np.int32(np.uint32(st[10]))) / 32,
                           "quiet_s": (q[0] / 100) if q else 0,
                           "quiet_rate": [o / q[0] for o in q[1:17]] if q and q[0] else None,
                           "updated": time.strftime("%H:%M:%S")}
        return seen

    def snapshot(self) -> dict:
        with self.lock:
            cur = self.rec.view(self.names)
            cur["saved"] = self.rec.saved
            views = {**self.records}
            if self.rec.ref:
                views[self.rec.idx] = cur
            vs = [views[k] for k in sorted(views)]
            summary = [{k: v[k] for k in ("idx", "ref", "kind", "label", "sim_pred", "started", "target",
                                          "done", "saved", "result")} for v in vs]
            return {"status": self.status, "caps": cur["caps"][-KEEP:], "tally": dict(self.tally),
                    "total": self.total, "agree": self.agree, "records": summary,
                    "playing": cur["ref"] if self.rec.ref else None, "paused": self.paused,
                    "rec": {k: v for k, v in cur.items() if k != "caps"},
                    "confusion": confusion(vs, self.names), "session": str(self.session.resolve()),
                    "target_override": self.target_override,
                    "vthr": self.vthr.text, "vthr_ok": self.vthr.mv is not None}


def save_png(rec: dict, fc: Optional[List[int]], out: Path) -> Path:
    """One capture as a PNG: the board's 16x100, the software AFE's for the same ARB, the
    per-channel firing rates, and the answers in the title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = [f"ch{k} {fc[k]}" if fc else f"ch{k}" for k in range(16)]
    b = bits_of(rec["clip"]).T
    s = bits_of(rec["sim"]).T if rec.get("sim") else None
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [5, 5, 2.4]})
    for a, img, t in ((ax[0], b, "보드 (STM32 → FPGA)"), (ax[1], s, f"소프트웨어 AFE ({rec.get('playing') or '—'})")):
        if img is None:
            a.axis("off")
            continue
        a.imshow(img, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1.4, interpolation="nearest")
        a.set_yticks(range(16), labels, fontsize=8)
        a.set_xticks(range(0, 101, 20), [f"{t * 10}" for t in range(0, 101, 20)])
        a.set_xlabel("ms")
        a.set_title(t)
    y = np.arange(16)
    ax[2].barh(y + 0.2, b.mean(1), 0.4, color="#1f5fbf", label="보드")
    if s is not None:
        ax[2].barh(y - 0.2, s.mean(1), 0.4, color="#9aa1ab", label="소프트웨어")
    ax[2].set_yticks(y, [f"ch{k}" for k in range(16)], fontsize=8)
    ax[2].set_ylim(-0.6, 15.6)
    ax[2].set_xlabel("발화율")
    ax[2].legend(fontsize=8, loc="lower right")
    good = rec.get("label") and rec["fpga"] == rec["label"]
    fig.suptitle(f"캡처 #{rec['n']} ({rec['time']}) · 정답 {rec.get('label') or '—'} · FPGA {rec['fpga']}"
                 f"{' ○' if good else (' ×' if rec.get('label') else '')} · best.pt {rec['best_pt']} {rec['prob']:.2f}"
                 f" · 시뮬 예측 {rec.get('sim_pred') or '—'}", fontsize=12)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"cap_{time.strftime('%Y%m%d_%H%M%S')}_n{rec['n']}_{rec.get('playing') or 'none'}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


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
    ap.add_argument("--sessions", default="out/afe/sessions",
                    help="one folder per viewer run, one sub-folder per ARB record (CSV, PNG)")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    ap.add_argument("--vthr", default="", help="comparator references [mV]: one value for all "
                                               "16, or 16 comma-separated (can be set on the page)")
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
    stamp = time.strftime("%Y%m%d_%H%M%S")
    session = Path(args.sessions) / stamp
    ref_paths = [Path(args.ref)] if args.ref else sorted(Path(args.refs).glob("*/ref.json"))
    refs = {}
    for p in ref_paths:
        r = load_ref(p, names)
        if r:
            refs[r["ref"]] = r
    print(f"{len(refs)} reference(s), {sum(r['kind'] == 'sequence' for r in refs.values())} of them sequences; "
          f"records -> {session}", flush=True)
    page_cfg = {"names": names, "fc": fc,
                "refs": [{k: r[k] for k in ("ref", "kind", "label", "sim_pred", "slots", "default_target")}
                         | {"sim": r["sim"][0]["window"]} for r in refs.values()]}
    poller = Poller(Board(args.elf, args.cli, args.nm), model, names, args.period,
                    save / f"live_{stamp}.jsonl", refs, session, fc, VThr(args.vthr))
    poller.start()
    page = PAGE.replace("/*CFG*/null", json.dumps(page_cfg))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/state"):
                body, ctype = json.dumps(poller.snapshot()).encode(), "application/json"
            elif self.path.startswith("/play"):              # /play?ref=<arb>[&again=1]: new record
                qs = parse_qs(urlparse(self.path).query)
                poller.select((qs.get("ref") or [None])[0], again=(qs.get("again") or ["0"])[0] == "1")
                body, ctype = b"{}", "application/json"
            elif self.path.startswith("/vthr"):              # /vthr?v=948 | /vthr?v=900,916,...
                poller.set_vthr((parse_qs(urlparse(self.path).query).get("v") or [""])[0].strip())
                body = json.dumps({"vthr": poller.vthr.text, "ok": poller.vthr.mv is not None}).encode()
                ctype = "application/json"
            elif self.path.startswith("/target"):            # /target?n=<captures>, empty = default
                v = (parse_qs(urlparse(self.path).query).get("n") or [""])[0]
                poller.set_target(int(v) if v.isdigit() and int(v) > 0 else None)
                body, ctype = b"{}", "application/json"
            elif self.path.startswith("/save_record"):
                path = poller.save_now()
                res = {"saved": [path]} if path else {"error": "ARB 를 고른 기록이 없음"}
                body, ctype = json.dumps(res).encode(), "application/json"
            elif self.path.startswith("/pause"):             # /pause?on=1 | /pause?on=0
                poller.set_paused((parse_qs(urlparse(self.path).query).get("on") or ["1"])[0] == "1")
                body, ctype = json.dumps({"paused": poller.paused}).encode(), "application/json"
            elif self.path.startswith("/save_png"):          # /save_png?n=<capture number>
                want = (parse_qs(urlparse(self.path).query).get("n") or [""])[0]
                caps = poller.snapshot()["caps"]
                rec = next((c for c in caps if str(c["n"]) == want), caps[-1] if caps else None)
                r = poller.rec
                where = session / (f"{r.idx:02d}_{r.name}" if r.ref else "no_arb")
                res = {"saved": [str(save_png(rec, fc, where).resolve())]} if rec else {"error": "no capture yet"}
                body, ctype = json.dumps(res).encode(), "application/json"
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
.full{grid-column:1/-1}table.res{border-collapse:collapse;width:100%;font-size:13px;color:var(--ink)}.res th,.res td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}.res th{color:var(--mute);font-weight:600}select{font:inherit;padding:2px 4px}button{font:inherit;padding:3px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}.pausebtn.on{background:var(--bad);border-color:var(--bad);color:#fff}@media (max-width:860px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>AFE → STM32 → FPGA 실시간</h1><span id="conn" class="mute"><span class="dot"></span>연결 중</span>
<label>재생 중인 ARB <select id="play"><option value="">(고르세요)</option></select></label>
<label title="비교기 기준전압. 하나만 쓰면 16채널 모두, 아니면 ch0..15 순서로 16개를 쉼표로. 바꾸면 같은 ARB 의 새 기록이 시작된다">V_thr (mV) <input id="vthr" size="34" placeholder="948  또는  900,916,...(16개)"></label><span id="vthrok" class="mute"></span>
<button class="pausebtn"></button><span id="cnt" class="mute"></span></header>
<main>
<section class="card full" id="recbar"><div style="display:flex;gap:10px 18px;align-items:center;flex-wrap:wrap">
<b id="rect">기록 없음 — 재생 중인 ARB 를 고르면 그 ARB 의 기록이 새로 시작된다</b>
<label class="mute">목표 캡처 <input id="target" type="number" min="1" style="width:70px;font:inherit" placeholder="자동"></label>
<span id="recp" class="mute"></span>
<button id="newrec">새 기록 (같은 ARB 다시 — threshold 바꾼 뒤)</button><button id="saverec">지금 저장</button>
</div><div id="recs" class="mute" style="font-size:12px;margin-top:6px"></div></section>
<section class="card"><div class="ans">
<div><div class="k">캡처</div><b id="cn">—</b></div>
<div><div class="k">FPGA 답</div><b id="fp">—</b></div>
<div><div class="k">best.pt (같은 100칸)</div><b id="bp">—</b></div>
<div><div class="k">정답</div><b id="lb">—</b></div>
<div><div class="k">소프트웨어 AFE 예측</div><b id="sp">—</b></div></div>
<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
<button id="savepng">이 캡처 PNG 저장</button><button id="savecsv">이 기록 저장 (CSV·PNG)</button>
<span id="saved" class="mute" style="font-size:12px"></span></div>
<h2 id="imgt">STM32 가 FPGA 로 보낸 16×100 (진할수록 1, 아래가 저역)</h2><canvas id="img" height="300"></canvas></section>
<section class="card"><h2>FPGA 답 누계 (전체)</h2><div id="tally" class="mute">아직 없음</div>
<h2 style="margin-top:16px">FPGA = best.pt</h2><div id="agree" style="font-size:22px">—</div>
<h2 style="margin-top:16px">보드</h2><div id="board" class="mute"></div></section>
<section class="card"><h2>채널별 발화율 <span class="legend"><span style="background:var(--on)"></span>보드 (이 캡처)<span style="background:var(--sim)"></span>소프트웨어 AFE</span></h2><canvas id="rate" height="190"></canvas></section>
<section class="card"><h2 id="simt">소프트웨어 AFE 기준</h2><canvas id="sim" height="190"></canvas></section>
<section class="card full"><h2 id="tunet">threshold 조정 안내</h2>
<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px"><span id="tunei" class="mute"></span>
<button class="pausebtn"></button></div>
<div class="legend mute" style="margin-bottom:4px"><span style="background:var(--sim)"></span>목표 (소프트웨어 AFE, 같은 ARB)<span style="background:var(--on)"></span>보드 (합산)<span style="background:var(--bad)"></span>무음 발화</div>
<canvas id="tunec" height="190"></canvas><div id="tunetab" style="margin-top:10px"></div>
<div class="mute" style="font-size:12px;margin-top:8px">합산 = 지금 기록(이 ARB 를 고른 뒤)의 채점된 캡처 전부 (고른 직후 2 초, 일시 중지 중, 슬롯에 안 맞는 캡처 제외). 목표는 캡처마다 그 슬롯의 소프트웨어 AFE 캡처 창이며,
ARB 100 mVpp ≈ PCM 0.29 레벨 보정을 쓴다 — 레벨 기준이 바뀌면 목표도 바뀐다. 무음 발화 = 보드가 트리거 대기(IDLE) 중인 프레임에서 켜진 비율 (ARB 앞뒤 무음 포함).
Vthr 를 올리면 발화가 준다. 발화율은 Vthr 에 대해 넓게 평평하고 잡음 바닥 근처에서 급하게 변하므로, 한 번에 조금씩 바꾸고 "새 기록" 으로 다시 모은다.</div></section>
<section class="card full" id="slotcard" style="display:none"><h2 id="slott">슬롯별 결과</h2><div id="slots"></div></section>
<section class="card full"><h2>최근 캡처 (누르면 위에 크게)</h2><div id="thumbs"></div></section>
<section class="card full"><h2 id="sumt">기록별 결과 (이번 세션)</h2><div id="sum" class="mute">재생 중인 ARB 를 고르면 기록이 시작되고, 다른 ARB 를 고르면 저장된 뒤 여기 한 줄이 된다</div></section>
<section class="card full"><h2 id="cmt">정답 × FPGA 답 (이번 세션, 채점된 캡처)</h2><div id="cm" class="mute">아직 없음</div></section>
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
const vthrIn=document.getElementById('vthr');
vthrIn.onchange=async()=>{await fetch('/vthr?v='+encodeURIComponent(vthrIn.value));sel=null;tick()};
const esc=s=>String(s??'—');
function render(){if(!S)return;const st=S.status;const c=document.getElementById('conn');
 c.innerHTML=`<span class="dot ${st.ok?'ok':''}"></span>`+(st.ok?`연결됨 · ${st.updated}`:st.msg);
 if(document.activeElement!==playSel)playSel.value=S.playing||'';
 if(document.activeElement!==vthrIn)vthrIn.value=S.vthr||'';
 document.getElementById('vthrok').textContent=S.vthr?(S.vthr_ok?'✓ 기록 중':'✗ 형식 (1개 또는 16개)'):'(보정·채점에 쓰려면 입력)';
 if(st.ok){document.getElementById('cnt').textContent=`보낸 클립 ${st.clips} · 응답 ${st.ok_n} · 타임아웃 ${st.timeouts} · 배경 ${st.bg.toFixed(2)}채널`;
  document.getElementById('board').innerHTML=`무음 ${st.quiet_s.toFixed(1)} s 동안 발화율<br>`+(st.quiet_rate?st.quiet_rate.map((r,k)=>`ch${k} ${r.toFixed(2)}`).join(' · '):'—')}
 renderRec();
 const caps=S.caps,cur=caps.find(x=>x.n===sel)||caps[caps.length-1],pr=S.playing?REF[S.playing]:null;
 const label=cur?cur.label:(pr&&pr.label),simw=cur&&cur.sim?cur.sim:(pr&&pr.sim),simRef=cur&&cur.playing?cur.playing:(pr&&pr.ref);
 if(cur){const good=label&&cur.fpga===label;document.getElementById('cn').textContent='#'+cur.n;
  const fp=document.getElementById('fp');fp.textContent=cur.fpga;fp.className=label?(good?'ok':'badc'):'';
  document.getElementById('bp').textContent=`${cur.best_pt} ${cur.prob.toFixed(2)}`;
  document.getElementById('imgt').textContent=`캡처 #${cur.n} (${cur.time})${cur.playing?' · '+cur.playing+(isSeq(cur)?' 슬롯 '+cur.slot:'')+(cur.ref?'':' ('+(WHY[cur.why]||cur.why)+', 집계 제외)'):''} — STM32 가 FPGA 로 보낸 16×100, 아래가 저역`;
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
  const t=document.createElement('div');t.innerHTML=`#${x.n} ${isSeq(x)?'s'+x.slot:''}${x.ref?'':' ·제외'} <span class="${x.label?(x.fpga===x.label?'ok':'badc'):''}">${x.fpga}</span>`;d.appendChild(t);
  d.onclick=()=>{sel=(sel===x.n?null:x.n);render()};th.appendChild(d);drawImg(cv,x.clip,css('--on'),false)});
 renderTune();renderPause();
 const P=S.records;if(P.length){let ans=0,cor=0,maj=0;
  const rows=P.map(r=>{const R=r.result,top=Object.entries(R.fpga).filter(([k])=>CFG.names.includes(k)).sort((a,b)=>b[1]-a[1]);ans+=R.answered;cor+=R.correct;
   const m=top.length&&top[0][0]===r.label;maj+=m;const sv=r.saved?`<span title="${r.saved}">저장됨</span>`:'—';
   return `<tr><td>${String(r.idx).padStart(2,'0')}</td><td>${r.ref}</td><td>${r.label}</td><td>${r.sim_pred}</td><td>${r.started}</td><td>${R.n}${r.target?' / '+r.target:''}${r.done?' ○':''}</td><td>${R.answered}</td>`+
    `<td>${top.slice(0,3).map(([k,v])=>`<span class="${k===r.label?'ok':''}">${k} ${v}</span>`).join(', ')||'—'}</td><td class="${m?'ok':'badc'}">${R.answered?Math.round(100*R.correct/R.answered)+'%':'—'}</td><td>${R.unmatched||0}</td><td>${sv}</td></tr>`}).join('');
  document.getElementById('sumt').textContent=`기록별 결과 (이번 세션) — ${P.length}개 · 최다 답이 정답 ${maj}/${P.length} · 캡처 단위 정답 ${cor}/${ans}${ans?' ('+Math.round(100*cor/ans)+'%)':''}`;
  document.getElementById('sum').innerHTML=`<table class="res"><tr><th>#</th><th>ARB</th><th>정답</th><th>시뮬</th><th>시작</th><th>채점 캡처</th><th>FPGA 응답</th><th>FPGA 답</th><th>정답률</th><th>미정렬</th><th>파일</th></tr>${rows}</table>`}
 renderSlots();renderCM()}
const WHY={paused:'일시 중지',settle:'고른 직후',aligning:'슬롯 정렬 중',unmatched:'슬롯에 안 맞음','no ARB':'ARB 안 고름'};
const isSeq=x=>x.slot!=null&&x.playing&&REF[x.playing]&&REF[x.playing].kind==='sequence';
function renderRec(){const R=S.rec,ti=document.getElementById('target');
 if(document.activeElement!==ti)ti.value=S.target_override||'';
 if(!S.playing){document.getElementById('rect').textContent='기록 없음 — 재생 중인 ARB 를 고르면 그 ARB 의 기록이 새로 시작된다';document.getElementById('recp').textContent='';
  document.getElementById('recs').textContent='세션 폴더: '+S.session;return}
 const r=R.result,A=R.align;
 document.getElementById('rect').textContent=`기록 ${String(R.idx).padStart(2,'0')} · ${R.ref} (${R.label}) · ${R.started} 부터`;
 document.getElementById('recp').innerHTML=`채점 캡처 <b>${r.n}</b>${R.target?' / '+R.target:''}${R.done?' — <span class="ok">목표 도달, 자동 일시 중지</span>':''}`+
  ` · 제외: 일시중지 ${r.paused}, 고른 직후 ${r.settling}${R.kind==='sequence'?`, 슬롯에 안 맞음 ${r.unmatched}`:''}`+
  (R.kind==='sequence'?` · 슬롯 정렬: ${A.state==='aligned'?`맞음 (${A.matched}개 일치, 오프셋 ${A.offset})`:'캡처 3개부터'}`:'');
 document.getElementById('recs').textContent=(R.saved?'저장: '+R.saved:'아직 저장 안 함 (ARB 를 바꾸거나 목표에 닿으면 자동 저장)')+' · 세션 폴더: '+S.session}
function renderSlots(){const R=S.rec,card=document.getElementById('slotcard');
 if(!S.playing||R.kind!=='sequence'){card.style.display='none';return}card.style.display='';
 document.getElementById('slott').textContent=`슬롯별 결과 — ${R.ref}`;
 document.getElementById('slots').innerHTML=`<table class="res"><tr><th>슬롯</th><th>클립</th><th>정답</th><th>시뮬 (클립)</th><th>시뮬 (캡처)</th><th>보드 캡처</th><th>FPGA 답</th></tr>`+
  R.slots.map(s=>`<tr><td>${s.slot}</td><td>${s.src}</td><td>${s.label}</td><td class="${s.clip_pred===s.label?'ok':'badc'}">${s.clip_pred}</td><td class="${s.sim_pred===s.label?'ok':'badc'}">${s.sim_pred||'(트리거 안 됨)'}</td><td>${s.n}</td>`+
   `<td>${Object.entries(s.fpga).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`<span class="${k===s.label?'ok':'badc'}">${k} ${v}</span>`).join(', ')||'—'}</td></tr>`).join('')+'</table>'}
function renderCM(){const C=S.confusion;if(!C.rows.length)return;
 const cols=C.cols.filter(c=>C.rows.includes(c)||C.rows.some(l=>C.m[l][c]));let tot=0,ok=0;
 const body=C.rows.map(l=>{const row=C.m[l],n=Object.values(row).reduce((a,b)=>a+b,0);tot+=n;ok+=row[l]||0;
  return `<tr><th>${l}</th>`+cols.map(c=>{const v=row[c]||0;return `<td style="text-align:center;${v?`background:color-mix(in srgb,var(--on) ${Math.round(70*v/n)}%,transparent)`:''}" class="${v&&c===l?'ok':''}">${v||''}</td>`}).join('')+`<td>${n}</td><td>${n?Math.round(100*(row[l]||0)/n)+'%':''}</td></tr>`}).join('');
 document.getElementById('cmt').textContent=`정답 × FPGA 답 (이번 세션, 채점된 캡처) — 정확도 ${tot?Math.round(1000*ok/tot)/10:0}% (${ok}/${tot})`;
 document.getElementById('cm').innerHTML=`<table class="res"><tr><th>정답 \\ FPGA</th>${cols.map(c=>`<th>${c}</th>`).join('')}<th>합</th><th>정답률</th></tr>${body}</table>`}
const TOL=.03,QMAX=.02,MIN_N=5;
function verdict(n,b,s,q){const d=b-s;
 if(n<MIN_N)return ['mute',`캡처 더 모으기 (${n}/${MIN_N})`];
 if(q>QMAX&&d<-TOL)return ['badc','무음에서 켜지는데 음성은 부족 — threshold 로는 안 됨, 이득·배선 확인'];
 if(q>QMAX)return ['badc','무음에서 켜짐 — Vthr 올리기 (잡음 바닥에 걸림)'];
 if(b<.005&&s>=TOL)return ['badc','꺼져 있음 — Vthr 를 조금씩 내리며 무음 발화 확인 (무음이 먼저 켜지면 이득·배선)'];
 if(d>TOL)return ['',`발화 ${d.toFixed(2)} 줄이기 (클립당 ${Math.round(d*100)}칸) → Vthr 올리기 ↑`];
 if(d<-TOL)return ['',`발화 ${(-d).toFixed(2)} 늘리기 (클립당 ${Math.round(-d*100)}칸) → Vthr 내리기 ↓`];
 return ['ok','맞음']}
function renderTune(){const T=S.rec&&S.rec.tune;if(!T)return;const n=T.n,qn=T.quiet_n||0;
 const b=T.board.map(x=>n?x/(100*n):0),s=T.sim.map(x=>n?x/(100*n):0),q=(T.quiet||[]).map(x=>qn?x/qn:0);
 document.getElementById('tunei').textContent=(S.playing?`기록 ${String(S.rec.idx).padStart(2,'0')} (${S.playing}) · ${T.since} 부터 · 채점 캡처 ${n}개 · 무음 ${(qn/100).toFixed(1)} s`:'ARB 를 고르면 기록이 시작된다')+(S.paused?' · 일시 중지됨 (캡처는 보이지만 더하지 않음)':'');
 const [g,W,H]=setup(document.getElementById('tunec'));const B=20,bw=W/16,max=Math.max(.3,...b,...s,...q)*1.1;
 g.font='11px system-ui';g.textAlign='center';
 for(let k=0;k<16;k++){const x=k*bw,y=v=>(H-B)*(1-Math.min(v,max)/max);
  g.fillStyle=css('--sim');g.fillRect(x+bw*.12,y(s[k]),bw*.76,H-B-y(s[k]));
  g.fillStyle=css('--on');g.fillRect(x+bw*.3,y(b[k]),bw*.4,H-B-y(b[k]));
  if(qn){g.fillStyle=css('--bad');g.fillRect(x+bw*.08,y(q[k])-1.5,bw*.84,3)}
  g.fillStyle=css('--mute');g.fillText(k,x+bw/2,H-B+4)}
 const rows=Array.from({length:16},(_,k)=>{const [cl,txt]=verdict(n,b[k],s[k],q[k]),d=b[k]-s[k];
  return `<tr><td>${chLabel(k)}</td><td>${n?s[k].toFixed(2):'—'}</td><td>${n?b[k].toFixed(2):'—'}</td>`+
   `<td class="${Math.abs(d)>TOL?'badc':''}">${n?(d>=0?'+':'')+d.toFixed(2):'—'}</td><td class="${q[k]>QMAX?'badc':''}">${qn?q[k].toFixed(3):'—'}</td><td class="${cl}">${txt}</td></tr>`}).join('');
 document.getElementById('tunetab').innerHTML=`<table class="res"><tr><th>채널</th><th>목표</th><th>보드</th><th>보드−목표</th><th>무음 발화</th><th>조정</th></tr>${rows}</table>`}
document.getElementById('newrec').onclick=async()=>{if(!S||!S.playing)return;await fetch('/play?again=1&ref='+encodeURIComponent(S.playing));sel=null;tick()};
document.getElementById('saverec').onclick=()=>saveTo('/save_record');
document.getElementById('target').onchange=async e=>{await fetch('/target?n='+encodeURIComponent(e.target.value));tick()};
async function saveTo(url){const el=document.getElementById('saved');el.textContent='저장 중...';
 try{const r=await (await fetch(url)).json();el.textContent=r.error?r.error:'저장됨: '+r.saved.join(' , ')}catch(e){el.textContent='저장 실패 (서버 꺼짐?)'}}
document.getElementById('savepng').onclick=()=>{const caps=S?S.caps:[],cur=caps.find(x=>x.n===sel)||caps[caps.length-1];
 saveTo('/save_png'+(cur?'?n='+cur.n:''))};
document.getElementById('savecsv').onclick=()=>saveTo('/save_record');
const pauseBtns=[...document.querySelectorAll('.pausebtn')];
pauseBtns.forEach(b=>b.onclick=async()=>{await fetch('/pause?on='+(S&&S.paused?0:1));tick()});
function renderPause(){const p=S&&S.paused;pauseBtns.forEach(b=>{b.textContent=p?'▶ 집계 재개':'⏸ 집계 일시 중지';b.className='pausebtn'+(p?' on':'')})}
async function tick(){try{S=await (await fetch('/state')).json();render()}catch(e){document.getElementById('conn').innerHTML='<span class="dot"></span>서버 꺼짐'}}
setInterval(tick,1000);tick();addEventListener('resize',render);</script></body></html>
"""

if __name__ == "__main__":
    main()
