"""Replay report -- what the FPGA answered for each golden clip, as one HTML page.

    python -I replay_report.py got.txt out.html [--vectors DIR] [--seconds S]

got.txt   one number per line, clip i's class from the FPGA as read off the
          STM32 by read_stats.ps1 -Replay (255 = timeout, 254 = sync/CRC error)
--vectors the golden directory the replay header was cut from
          (predictions_fixed.txt = what the hardware must say,
           labels.txt = what the clip really is)

Two different questions, kept apart on the page:
  hardware == Python integer model?   must be every clip (the pass criterion)
  model == true label?                 the model's accuracy on these clips only

Standard library only -- it runs on the Python that ships with Vivado.
"""
from __future__ import annotations

import argparse
import html
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
         "silence", "unknown"]           # data/speech_commands.py class order
TIMEOUT, BAD = 255, 254


def read_ints(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().split()]


def name(c: int) -> str:
    if c == TIMEOUT:
        return "timeout"
    if c == BAD:
        return "CRC error"
    return NAMES[c] if 0 <= c < len(NAMES) else f"#{c}"


def render(got: list[int], exp: list[int], lab: list[int], seconds: float | None) -> str:
    n = len(got)
    hw_match = sum(g == e for g, e in zip(got, exp))
    timeouts = got.count(TIMEOUT)
    bad = got.count(BAD)
    acc = sum(g == t for g, t in zip(got, lab)) / n
    passed = hw_match == n
    classes = sorted(set(lab) | {g for g in got if g < len(NAMES)})

    # 600-cell strip: one square per clip, in send order
    cells = []
    for i, (g, e, t) in enumerate(zip(got, exp, lab)):
        if g != e:
            kind = "miss" if g < len(NAMES) else "lost"
        else:
            kind = "ok" if g == t else "okwrong"
        tip = f"#{i}: FPGA {name(g)}, expected {name(e)}, true {name(t)}"
        cells.append(f'<span class="c {kind}" title="{html.escape(tip)}"></span>')

    # confusion: true label (rows) x FPGA answer (cols)
    conf = Counter(zip(lab, got))
    head = "".join(f"<th>{name(c)}</th>" for c in classes)
    rows = []
    for t in classes:
        tot = sum(conf[(t, c)] for c in classes)
        if tot == 0:
            continue
        tds = []
        for c in classes:
            v = conf[(t, c)]
            cls = "diag" if c == t and v else ("off" if v else "zero")
            tds.append(f'<td class="{cls}">{v or ""}</td>')
        rows.append(f"<tr><th>{name(t)}</th>{''.join(tds)}<td class='tot'>{tot}</td></tr>")

    per_class = []
    for t in classes:
        idx = [i for i in range(n) if lab[i] == t]
        if idx:
            ok = sum(got[i] == t for i in idx)
            per_class.append(f"<li><b>{name(t)}</b> {ok}/{len(idx)} "
                             f"({ok / len(idx):.1%})</li>")

    verdict = "PASS" if passed else "FAIL"
    speed = (f"<div class='k'><span>{seconds:.1f} s</span>"
             f"<small>{seconds / n * 1000:.0f} ms per clip</small></div>") if seconds else ""
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Replay report</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --mute:#6b6b70; --line:#e3e3e6;
        --ok:#2f8f5b; --okwrong:#9fd3b4; --miss:#d1453b; --lost:#8a8a90; --acc:#2a6fdb; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --bg:#17171a; --fg:#ececf0; --mute:#9a9aa2; --line:#2e2e34;
        --ok:#3fae72; --okwrong:#2c5e43; --miss:#e5584e; --lost:#6c6c73; --acc:#5b93f0; }} }}
body {{ background:var(--bg); color:var(--fg); margin:0;
       font:14px/1.5 system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif; }}
main {{ max-width:880px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:20px; margin:0 0 4px; }} h2 {{ font-size:15px; margin:28px 0 8px; }}
.sub {{ color:var(--mute); margin:0 0 20px; }}
.kpi {{ display:flex; flex-wrap:wrap; gap:12px; }}
.k {{ border:1px solid var(--line); border-radius:8px; padding:10px 14px; min-width:120px; }}
.k span {{ display:block; font-size:20px; font-weight:600; }}
.k small {{ color:var(--mute); }}
.pass span {{ color:var(--ok); }} .fail span {{ color:var(--miss); }}
.strip {{ display:grid; grid-template-columns:repeat(25, 1fr); gap:2px; max-width:520px; }}
.c {{ aspect-ratio:1; border-radius:2px; }}
.ok {{ background:var(--ok); }} .okwrong {{ background:var(--okwrong); }}
.miss {{ background:var(--miss); }} .lost {{ background:var(--lost); }}
.legend {{ color:var(--mute); font-size:12.5px; margin-top:8px; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin:0 4px 0 12px; vertical-align:-1px; }}
.legend i:first-child {{ margin-left:0; }}
table {{ border-collapse:collapse; font-variant-numeric:tabular-nums; }}
th, td {{ border:1px solid var(--line); padding:4px 10px; text-align:right; }}
th {{ font-weight:600; color:var(--mute); }}
td.diag {{ color:var(--ok); font-weight:600; }} td.off {{ color:var(--miss); }}
td.tot {{ color:var(--mute); }}
.note {{ color:var(--mute); font-size:13px; }}
ul {{ padding-left:18px; }}
.wrap {{ overflow-x:auto; }}
</style></head><body><main>
<h1>Golden replay: STM32 → UART → FPGA</h1>
<p class="sub">{n} clips from <code>rtl/gen/bd_base/selftest</code> (GSC v2 test, software AFE).
FPGA <code>kws_wake_top</code> + <code>bd_base</code>.</p>

<div class="kpi">
  <div class="k {'pass' if passed else 'fail'}"><span>{verdict} {hw_match}/{n}</span>
    <small>FPGA = Python integer model</small></div>
  <div class="k"><span>{timeouts + bad}</span><small>timeouts {timeouts}, CRC {bad}</small></div>
  <div class="k"><span>{acc:.1%}</span><small>vs true label (these {n} clips)</small></div>
  {speed}
</div>

<h2>Every clip, in send order</h2>
<div class="strip">{''.join(cells)}</div>
<p class="legend"><i style="background:var(--ok)"></i>hardware = model, model right
<i style="background:var(--okwrong)"></i>hardware = model, model wrong
<i style="background:var(--miss)"></i>hardware ≠ model
<i style="background:var(--lost)"></i>no answer</p>
<p class="note">Pass/fail is the first two colours together: the hardware has to give the model's
answer, right or wrong. The pale cells are the model's own mistakes, not the link's.</p>

<h2>True label × FPGA answer</h2>
<div class="wrap"><table><tr><th>true \\ FPGA</th>{head}<th>n</th></tr>{''.join(rows)}</table></div>
<ul>{''.join(per_class)}</ul>
<p class="note">The golden set holds only the classes shown here (the slice is not class-balanced),
so this is not the model's 12-class test accuracy.</p>
</main></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("got", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--vectors", type=Path, default=ROOT / "rtl" / "gen" / "bd_base" / "selftest")
    ap.add_argument("--seconds", type=float, default=None)
    a = ap.parse_args()

    got = read_ints(a.got)
    n = len(got)
    exp = read_ints(a.vectors / "predictions_fixed.txt")[:n]
    lab = read_ints(a.vectors / "labels.txt")[:n]
    if len(exp) < n or len(lab) < n:
        raise SystemExit(f"{a.vectors} has fewer than {n} clips")
    a.out.write_text(render(got, exp, lab, a.seconds), encoding="utf-8")
    m = sum(g == e for g, e in zip(got, exp))
    print(f"report: {a.out}  ({m}/{n} hardware = model)")


if __name__ == "__main__":
    main()
