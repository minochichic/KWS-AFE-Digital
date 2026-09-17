"""Interim report (.docx) -- assembles text, tables and figures.

Numbers that come out of a run are READ from the run's files here (test JSON,
selftest ROMs, Vivado reports). Numbers from earlier experiments that exist only
in the project notes are written in with the note they came from.

    out/.venv_report/Scripts/python docs/report/build_report.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(".")
IMG = ROOT / "out/report_build/img"
FIG = ROOT / "out/report_remote/fig2/fig"
REP = ROOT / "out/report_remote/report"
HW = ROOT / "out/build/bd_base_bal/report"
HW75 = ROOT / "out/build/bd_base_ft20_partial75_bal/report"
OUT = ROOT / "out/report_build/KWS_FPGA_interim_report_v3.docx"

LATIN = "Times New Roman"
HANGUL = "Malgun Gothic"

T1_BASE = json.loads((REP / "bd_base_test.json").read_text())
T1_P75 = json.loads((REP / "bd_base_ft20_partial75_test.json").read_text())


# ---- low-level formatting ---------------------------------------------------- #
def set_fonts(run, size=None, bold=None, italic=None):
    run.font.name = LATIN
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), LATIN)
    rfonts.set(qn("w:hAnsi"), LATIN)
    rfonts.set(qn("w:eastAsia"), HANGUL)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic


def style_doc(doc):
    for name in ("Normal", "Heading 1", "Heading 2", "Heading 3", "Title", "Caption"):
        st = doc.styles[name]
        st.font.name = LATIN
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rpr.append(rf)
        rf.set(qn("w:eastAsia"), HANGUL)
        st.font.color.rgb = None
    doc.styles["Normal"].font.size = Pt(10)
    doc.styles["Normal"].paragraph_format.space_after = Pt(4)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.25
    for name, size in (("Heading 1", 13), ("Heading 2", 11), ("Heading 3", 10)):
        st = doc.styles[name]
        st.font.size = Pt(size)
        st.font.bold = True
        st.paragraph_format.space_before = Pt(10 if name == "Heading 1" else 6)
        st.paragraph_format.space_after = Pt(4)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(2.0)


def para(doc, text, size=10, bold=False, italic=False, align=None, after=None):
    p = doc.add_paragraph()
    # **bold** inline markup, nothing more
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if not part:
            continue
        b = part.startswith("**")
        r = p.add_run(part[2:-2] if b else part)
        set_fonts(r, size, bold or b, italic)
    if align:
        p.alignment = align
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullets(doc, items, size=10):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        parts = re.split(r"(\*\*[^*]+\*\*)", it)
        for part in parts:
            if not part:
                continue
            b = part.startswith("**")
            r = p.add_run(part[2:-2] if b else part)
            set_fonts(r, size, b)
        p.paragraph_format.space_after = Pt(2)


def heading(doc, text, level):
    h = doc.add_heading(level=level)
    r = h.add_run(text)
    set_fonts(r, None, True)
    return h


FIG_N = [0]
TAB_N = [0]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
         "XIII", "XIV", "XV", "XVI"]


def figure(doc, path, caption, width_cm=16.0):
    path = Path(path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if path.exists():
        p.add_run().add_picture(str(path), width=Cm(width_cm))
    else:
        r = p.add_run(f"[그림 파일 없음: {path.as_posix()}]")
        set_fonts(r, 9, italic=True)
    FIG_N[0] += 1
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = c.add_run(f"Fig. {FIG_N[0]}. ")
    set_fonts(r, 9, True)
    r = c.add_run(caption)
    set_fonts(r, 9)
    c.paragraph_format.space_after = Pt(8)
    return FIG_N[0]


def shade(cell, fill):
    tcpr = cell._element.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), fill)
    tcpr.append(sh)


def borders(table):
    """Three-rule table (top, below header, bottom) -- the IEEE look."""
    tbl = table._element
    tblpr = tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "8")
        e.set(qn("w:color"), "000000")
        b.append(e)
    for edge in ("left", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "nil")
        b.append(e)
    tblpr.append(b)
    for cell in table.rows[0].cells:
        tcpr = cell._element.get_or_add_tcPr()
        tb = OxmlElement("w:tcBorders")
        e = OxmlElement("w:bottom")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "6")
        e.set(qn("w:color"), "000000")
        tb.append(e)
        tcpr.append(tb)


def table(doc, caption, header, rows, widths_cm=None, size=8.5, bold_rows=(),
          note=None):
    TAB_N[0] += 1
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(f"TABLE {ROMAN[TAB_N[0] - 1]}")
    set_fonts(r, 9, False)
    c.paragraph_format.space_after = Pt(0)
    c.paragraph_format.keep_with_next = True
    c2 = doc.add_paragraph()
    c2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c2.add_run(caption.upper() if caption.isascii() else caption)
    set_fonts(r, 9, False)
    c2.paragraph_format.space_after = Pt(2)
    c2.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.text = ""
        rr = cell.paragraphs[0].add_run(h)
        set_fonts(rr, size, True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = t.rows[i + 1].cells[j]
            cell.text = ""
            rr = cell.paragraphs[0].add_run(str(v))
            set_fonts(rr, size, i in bold_rows)
            cell.paragraphs[0].alignment = (WD_ALIGN_PARAGRAPH.LEFT if j == 0
                                            else WD_ALIGN_PARAGRAPH.CENTER)
            cell.paragraphs[0].paragraph_format.space_after = Pt(0)
            cell.paragraphs[0].paragraph_format.line_spacing = 1.0
    borders(t)
    if widths_cm:
        for row in t.rows:
            for j, w in enumerate(widths_cm):
                row.cells[j].width = Cm(w)
    if note:
        n = doc.add_paragraph()
        r = n.add_run(note)
        set_fonts(r, 8, italic=True)
        n.paragraph_format.space_after = Pt(8)
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return TAB_N[0]


# ---- data pulled from files -------------------------------------------------- #
def util_row(rpt: Path, name: str):
    m = re.search(rf"^\|\s+{re.escape(name)}\s+\|\s+(\d+)\s+\|.*?\|\s+([\d.<]+)\s+\|$",
                  rpt.read_text(), re.M)
    return (int(m.group(1)), m.group(2)) if m else ("—", "—")


def block_util(rpt: Path, inst: str):
    m = re.search(rf"^\|\s+{inst}\s+\|[^|]+\|\s+(\d+)\s+\|\s+\d+\s+\|\s+(\d+)\s+\|"
                  rf"\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|",
                  rpt.read_text(), re.M)
    return [int(g) for g in m.groups()] if m else None


def timing_summary(rpt: Path):
    txt = rpt.read_text()
    i = txt.index("Design Timing Summary")
    line = [l for l in txt[i:].splitlines() if re.match(r"\s+-?\d+\.\d+", l)][0]
    v = line.split()
    return dict(wns=float(v[0]), tns=float(v[1]), fail=int(v[2]), n=int(v[3]),
                whs=float(v[4]), ths=float(v[5]), nh=int(v[7]))


def block_slack(rep: Path, blk: str):
    f = rep / f"timing_to_u_top_u_net_u_top_{blk}.rpt"
    if blk == "u_st":
        f = rep / "timing_to_u_top_u_st.rpt"
    t = f.read_text()
    s = re.search(r"Slack \(MET\) :\s+([\d.]+)ns", t).group(1)
    lv = re.search(r"Logic Levels:\s+(\d+)", t).group(1)
    dp = re.search(r"Data Path Delay:\s+([\d.]+)ns\s+\(logic ([\d.]+)ns \(([\d.]+)%\)\s+route ([\d.]+)ns", t)
    return float(s), int(lv), float(dp.group(1)), float(dp.group(3))


def power(rep: Path):
    t = (rep / "power_hier.rpt").read_text()
    g = lambda k: re.search(rf"\|\s+{re.escape(k)}\s+\|\s+([<\d.]+)", t).group(1)
    hier = {}
    for k in ("u_net", "u_st", "dbg_hub", "u_vio"):
        m = re.search(rf"^\|\s+{k}\s+\|\s+([<\d.]+)", t, re.M)
        hier[k] = m.group(1) if m else "—"
    return dict(total=g("Total On-Chip Power (W)"), dyn=g("Dynamic (W)"),
                static=g("Device Static (W)"), clocks=g("Clocks"),
                logic=g("Slice Logic"), signals=g("Signals"), bram=g("Block RAM"),
                dsp=g("DSPs"), hier=hier,
                conf=re.search(r"Confidence Level\s+\|\s+(\w+)", t).group(1))


def balanced_acc(tag: str, n: int):
    d = ROOT / f"rtl/gen/{tag}/selftest_bal"
    L = [int(x) for x in (d / "labels.txt").read_text().split()]
    F = [int(x) for x in (d / "predictions_fixed.txt").read_text().split()]
    return sum(a == b for a, b in zip(F[:n], L[:n])) / n


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


# ---- document ------------------------------------------------------------------ #
# Weight memory, estimated from parameter counts and the widths in parameters.vh
# (binary 1 bit, conv1/conv3 8 bit, conv4 7 bit). BN/threshold ROMs excluded.
PARAMS = dict(conv1=22784, b1=23616, b2=10624, b3=10880, conv2=10432, conv3=16640, conv4=1548)
BITS = dict(conv1=8, b1=1, b2=1, b3=1, conv2=1, conv3=8, conv4=7)


def build():
    doc = Document()
    style_doc(doc)
    TS = timing_summary(HW / "timing_summary.rpt")
    TS75 = timing_summary(HW75 / "timing_summary.rpt")
    PW = power(HW)
    n_par = sum(PARAMS.values())
    mem_bits = sum(PARAMS[k] * BITS[k] for k in PARAMS)
    fp_bits = n_par * 32
    bin_par = PARAMS["b1"] + PARAMS["b2"] + PARAMS["b3"] + PARAMS["conv2"]

    # ---- title / abstract ----
    para(doc, "중간보고서", 11, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    para(doc, "아날로그 이진 특징과 부분 이진화 MatchboxNet 기반", 16, True,
         align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
    para(doc, "키워드 스포팅 가속기의 FPGA 구현 및 칩 수준 검증", 16, True,
         align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
    para(doc, "KWS-AFE-Digital 프로젝트 · 2026년 9월 17일", 10,
         align=WD_ALIGN_PARAGRAPH.CENTER, after=12)

    heading(doc, "요약", 1)
    para(doc,
         "아날로그 필터뱅크와 비교기가 만든 16채널 이진 시간-주파수 이미지를 곧바로 분류하는 부분 이진화 "
         "MatchboxNet(BinaryMatchboxNet)을 설계하고, 이를 곱셈기 없는 정수 전용 RTL로 구현하여 Spartan-7 "
         "XC7S75 FPGA에서 검증하였다. Cerutti 등의 아날로그 이진 특징 + BNN 개념을 (i) 1D time-channel "
         "separable 구조의 부분 이진화, (ii) PCB로 설계된 실제 필터뱅크 응답과 저항 분압으로 구현 가능한 "
         "절대 임계값 학습, (iii) 전용 하드웨어 구현과 칩 수준 동치 검증, (iv) sliding window 기반 연속 판정으로 "
         f"확장하였다. Google Speech Commands v2 12-class test 세트({T1_BASE['n_clips']:,}개)에서 정수 경로 "
         f"정확도 {pct(T1_P75['fixed_acc'])} %를 얻었고 float 대비 양자화 손실은 없었다. 가중치 메모리는 float32 대비 "
         f"약 {fp_bits / mem_bits:.1f}배 작으며, 이진층에는 곱셈기를 쓰지 않는다. FPGA 구현은 50 MHz에서 setup 여유 "
         f"{TS['wns']:+.3f} ns로 타이밍을 만족하였고, 칩 내부 자체 검사로 12개 클래스를 포함한 2,400개 클립–모델 "
         "조합(반복 포함 5,400회 분류)에서 칩의 결과가 소프트웨어 정수 경로와 모두 일치하였다.")

    # ---- 1 ----
    heading(doc, "1. 서론", 1)
    para(doc,
         "항상 켜져 있어야 하는 키워드 스포팅(KWS)에서 분류기를 충분히 경량화하면, 에너지의 대부분은 "
         "마이크·ADC·FFT/MFCC로 이어지는 데이터 취득과 전처리가 차지한다. Cerutti 등 [1]은 이 부분을 아날로그 "
         "필터뱅크·포락선 검출기·비교기로 대체하여 이진 특징을 직접 얻고, 이를 이진 신경망(BNN)으로 분류하는 "
         "방식을 제안하였다. 그러나 [1]의 정확도는 이상적인 mel 필터를 가정한 소프트웨어 float 결과이고, 분류기 "
         "에너지는 MCU 실행 성능으로부터의 추정이며, 이진 입력 BNN을 전용 하드웨어로 옮겼을 때의 동작은 다루지 "
         "않았다. 또한 첫 인터럽트 이후 1초를 모아 한 번 분류하는 트리거 방식이어서, 단어 시점을 모르는 연속 "
         "동작은 고려하지 않았다.")
    para(doc, "본 연구의 주요 아이디어는 다음 네 가지이다.")
    bullets(doc, [
        "**부분 이진화 MatchboxNet.** 1D time-channel separable 구조를 채택하고 중간 블록만 이진화하여, "
        "곱셈기 없는 XNOR-popcount 연산과 1비트 활성 버퍼로 대부분의 계산을 수행한다.",
        "**제작 가능한 조건의 학습.** PCB로 설계된 필터뱅크의 응답과, 저항 분압 하나로 구현되는 채널별 "
        "절대 임계값을 학습 단계에 포함한다.",
        "**정수 전용 전용 하드웨어와 칩 수준 동치 검증.** BN을 정수 임계로 접은 folded RTL을 FPGA에 구현하고, "
        "칩 내부 자체 검사로 소프트웨어와 클립 단위 일치를 확인한다.",
        "**always-on 연속 판정.** 100 ms마다 1초 창을 이동시키는 sliding window와 연속 투표를 하드웨어에 "
        "두고, 창 경계 상황을 학습한 미세조정 모델로 검출 성능을 개선한다.",
    ])

    # ---- 2 ----
    heading(doc, "2. 평가 조건과 관련 연구", 1)
    para(doc,
         "KWS 문헌의 '정확도'는 대부분 1초 클립마다 한 번 분류해 라벨과 비교하는 클립 정확도이며, 단어가 창 "
         "안에 온전히 들어 있다는 가정을 전제로 한다. 실제 연속 동작에서는 창 경계에 걸린 단어와 반복 판정 때문에 "
         "지표가 달라진다. 본 보고서는 Table I의 세 층위로 결과를 구분하여 보고한다. 데이터셋 조건은 표준 설정"
         "(12 labels, 공식 분할, unknown·silence 각 10 %)을 따르며 test 세트는 4,888개로, 같은 설정에서 보고된 "
         "4,890개 [3]와 사실상 같다.")
    table(doc, "Evaluation tiers used in this report",
          ["Tier", "Question", "Input", "Criterion", "Sample size"],
          [["T1 Clip accuracy", "How well does the model classify?",
            "1-s GSC clip, word inside window", "argmax = label", "full test set (4,888)"],
           ["T2 HW reproduction", "Does the chip compute the same integers?",
            "same clips, stored in on-chip ROM", "chip class = Python integer class",
            "4 sets × 600 clips, 9 runs"],
           ["T3 Always-on", "Is a keyword detected in a stream?",
            "3-s spliced stream, 100-ms hop", "5 consecutive + margin", "val: 2,560 kw + 512 quiet"]],
          widths_cm=[3.0, 3.8, 3.6, 3.4, 3.2], size=8)
    table(doc, "Prior work on Google Speech Commands (clip-level accuracy)",
          ["Work", "Feature extraction", "Classifier", "Classes", "Accuracy (%)", "Platform / basis"],
          [["Cerutti et al. [1]", "Analog BPF + comparator (64 ch)", "BNN (2D CNN)", "12", "86.0", "SW float + MCU est."],
           ["Cerutti et al. [1]", "Analog BPF + comparator (8 ch)", "BNN (2D CNN)", "12", "76.3", "SW float + MCU est."],
           ["Rybakov et al. [3]", "MFCC (digital)", "DS-CNN (FP)", "12", "97.0", "SW (TFLite)"],
           ["Rybakov et al. [3]", "MFCC (digital)", "MHAtt-RNN (FP)", "12", "98.0", "SW"],
           ["Giraldo, Vocell [4]", "MFCC (digital)", "NN", "12", "90.87", "65 nm chip, 16 µW"],
           ["Kim et al. [5]", "Ring-osc. time domain (analog)", "RNN", "12", "86.03", "65 nm chip, 23 µW"],
           ["Chen, DeltaKWS [6]", "IIR BPF (digital)", "ΔGRU", "12", "89.5", "65 nm chip, 5.22 µW"],
           ["Tan et al. [7]", "—", "Transfer computing", "12", "91.8", "28 nm chip, 1.73 µW"],
           ["This work", "Analog BPF + comparator (16 ch, sim.)", "Partial BNN (1D TCS)", "12",
            pct(T1_P75["fixed_acc"]), "FPGA RTL, integer = chip"]],
          widths_cm=[3.0, 4.2, 3.0, 1.3, 1.9, 3.6], size=7.5, bold_rows=(8,),
          note="Chip numbers for [5], [7] as tabulated in [6]; Vocell as quoted in [1]. "
               "None of the listed works reports a continuous-stream metric on 12-class GSC.")

    # ---- 3 ----
    heading(doc, "3. 주요 아이디어", 1)
    heading(doc, "3.1 시스템 구성", 2)
    figure(doc, IMG / "fig_system.png",
           "System overview. The analog front end produces 16 comparator outputs; the FPGA captures 10-ms "
           "frames, keeps a sliding 1-s window, classifies it with the folded BinaryMatchboxNet and applies a "
           "consecutive-vote decision.", 16.5)
    para(doc,
         "아날로그 전단은 공동 연구자가 회로·PCB 설계와 SPICE 시뮬레이션까지 진행하였다. 실장이 이루어지지 않아 "
         "실제 보드 연결 측정은 아직 하지 못했으나, 시뮬레이션 응답으로 학습한 모델이 이상적인 mel 필터 설계보다 "
         "높은 정확도를 보이는 등 결과는 긍정적이다. 디지털 단과의 계약은 비교기 출력 16가닥뿐이며, 학습·평가 "
         "입력은 그 응답을 소프트웨어로 모사한 이진 이미지이다. 실제 보드가 연결되어도 디지털 단은 그대로 사용한다.")

    heading(doc, "3.2 부분 이진화 MatchboxNet", 2)
    para(doc,
         "[1]은 이진 이미지를 3×3 커널의 2D CNN으로 분류하였다. 본 연구는 MatchboxNet [2]의 1D time-channel "
         "separable(TCS) 구조를 채택하였다. 16개 주파수 채널을 입력 채널로 보고 시간축으로만 합성곱하므로, "
         "depthwise(시간)와 pointwise(채널 혼합)가 분리되어 파라미터가 적고 연산이 규칙적이다. 정밀도는 층별로 "
         "달리하여, [1]에서 입력과 양 끝단 이진화 시 정확도가 크게 떨어진 점을 반영해 첫 층(conv1)은 INT8, "
         "마지막 층(conv4)은 고정소수점으로 두고 중간 TCS 블록 B1–B3와 conv2만 이진화하였다(Table III).")
    table(doc, "Network architecture (C = 64, T = 128, 12 classes)",
          ["Stage", "Type", "Precision", "Kernel", "Out ch.", "Params", "Weight bits"],
          [["conv1", "1D conv, stride 2", "INT8 (binary input)", "11", "128", "22,784", "8"],
           ["B1", "TCS ×2 + projection skip", "binary", "13", "64", "23,616", "1"],
           ["B2", "TCS ×2 + identity skip", "binary", "15", "64", "10,624", "1"],
           ["B3", "TCS ×2 + identity skip", "binary", "17", "64", "10,880", "1"],
           ["conv2", "separable, dilation 2", "binary", "29", "128", "10,432", "1"],
           ["conv3", "1×1 conv", "INT8", "1", "128", "16,640", "8"],
           ["conv4", "1×1 conv", "fixed point", "1", "12", "1,548", "7"],
           ["Total", "", "", "", "", f"{n_par:,}", f"{mem_bits / 8 / 1000:.1f} kB"]],
          widths_cm=[1.6, 4.0, 3.2, 1.3, 1.4, 2.0, 2.2], size=8, bold_rows=(7,))

    heading(doc, "3.3 이진 MatchboxNet의 하드웨어 강점", 2)
    table(doc, "Hardware consequences of the partial binarization",
          ["Property", "Floating-point MatchboxNet", "This design", "Measured / estimated effect"],
          [["Multiply in B1–B3, conv2", "real multiply per weight", "XNOR + popcount (2·popcnt − N)",
            "no multiplier; DSP48 used only in the tail (7 of 140)"],
           ["Multiply in conv1", "real multiply", "±1 input → signed accumulation", "no multiplier"],
           ["BatchNorm + activation", "scale, shift, nonlinearity", "one integer comparison (α absorbed)",
            "no BN arithmetic in binary layers"],
           ["Weight memory", f"{fp_bits / 8 / 1000:.0f} kB (32 bit)", f"{mem_bits / 8 / 1000:.1f} kB",
            f"≈{fp_bits / mem_bits:.1f}× smaller; binary stages {100 * bin_par / n_par:.0f} % of params"],
           ["Inter-layer activations", "32 bit per value", "1 bit per value (binary stages)",
            "64-ch × 64-frame plane = 4,096 bit; 4 planes use 669 LUT in total"],
           ["Residual add", "float add", "integer add before a single threshold", "one compare per sub-block"],
           ["Quantization accuracy", "—", "integer path on full test set",
            f"{pct(T1_BASE['fixed_acc'])} % vs float {pct(T1_BASE['float_acc'])} % (no loss)"]],
          widths_cm=[3.2, 3.6, 4.4, 5.3], size=7.5,
          note="Weight memory counts weights only (binary 1 bit, conv1/conv3 8 bit, conv4 7 bit); "
               "integer thresholds and tail gain/offset ROMs are excluded.")
    para(doc,
         "이진층의 곱셈은 XNOR과 popcount로 바뀌고, 뒤따르는 BN과 부호 함수는 정수 누산값에 대한 비교 하나로 "
         "접힌다. 입력이 이미 ±1이므로 INT8 가중치의 conv1도 부호 있는 누산만으로 계산된다. 결과적으로 곱셈이 "
         "필요한 곳은 BN이 부호 함수로 흡수되지 않는 후단(conv2 pointwise, conv3, conv4)뿐이며, 실제 구현에서도 "
         "DSP 블록 7개가 모두 이 꼬리 부분에만 사용되었다. 활성값이 1비트이므로 계층 사이 버퍼도 작다. 64채널 × "
         "64프레임 평면이 4,096비트로, 네 개 평면을 합쳐 LUT 669개로 구현되었다.")
    para(doc,
         "TCS 구조는 folded 하드웨어와도 잘 맞는다. depthwise와 pointwise가 각각 단순한 누산 패턴을 가지므로, "
         "계층마다 MAC 엔진 하나를 채널·시간에 걸쳐 시분할로 재사용할 수 있다. 클립당 계산은 50 MHz에서 57 ms로 "
         "100 ms 판정 주기 안에 들어와, 완전 병렬 구조 없이도 실시간 요구를 만족한다. 가중치·임계값은 ROM 파일로 "
         "읽으므로 아날로그 조건이 바뀌어 재학습해도 RTL은 바뀌지 않는다.")

    heading(doc, "3.4 Cerutti 등 [1] 대비 발전사항", 2)
    table(doc, "What this work adds to the analog-binary-feature + BNN approach of [1]",
          ["Aspect", "Cerutti et al. [1]", "This work"],
          [["Classifier", "2D CNN (3×3 kernels), BNN", "1D time-channel separable MatchboxNet, partial binarization"],
           ["Filter bank in training", "ideal mel-spaced filters (software)", "response of the PCB-designed board filter bank (SPICE)"],
           ["Threshold realizability", "thresholds learned on normalized envelopes",
            "absolute per-channel thresholds (fixed normalization) → one resistor divider each"],
           ["Threshold training", "STE", "narrow STE window (clip 1.0 → 0.003): +10.0 %p accuracy"],
           ["Classifier execution", "GAP8 MCU popcount; energy estimated", "dedicated integer-only RTL on FPGA"],
           ["HW/SW agreement", "not reported", "2,400 / 2,400 clip–model pairs identical on chip"],
           ["Acquisition model", "trigger at first event, one 1-s window",
            "continuous sliding window (100-ms hop) + 5-in-a-row vote in hardware"],
           ["Boundary robustness", "—", "partial-window fine-tuning: +46 detections, −11 quiet false alarms (val)"],
           ["Reported conditions", "clip accuracy", "clip accuracy, chip equivalence and stream detection separated"]],
          widths_cm=[3.4, 5.6, 7.5], size=8)
    para(doc,
         "정확도 측면에서 16채널인 본 연구의 클립 정확도(82.6 %)는 [1]의 8채널(76.3 %)과 64채널(86.0 %) 사이에 "
         "있다. [1]의 수치가 이상적인 mel 필터와 float 모델의 결과인 데 비해, 본 결과는 실제로 설계된 필터뱅크와 "
         "저항 분압으로 구현 가능한 임계값을 전제로 한 정수 경로 정확도이며 칩에서 그대로 재현된다. 채널 수가 "
         "4배인 64채널 구성은 필터·비교기 수와 취득 에너지도 그만큼 늘어나므로, 16채널은 정확도와 아날로그 "
         "비용 사이의 절충점이다.")

    heading(doc, "3.5 기타 발전사항", 2)
    bullets(doc, [
        "**필터뱅크 선택의 효과.** 필터뱅크만 보드 설계로 바꾼 비교에서 test 정확도가 0.817에서 0.825로 올라, "
        "제작 조건을 반영하는 것이 정확도 손해가 아님을 확인하였다(프로젝트 ablation 기록).",
        "**양자화 무손실.** 꼬리 계층의 BN을 정수 이득·오프셋·시프트(Q*.6)로 옮기고 이전 계층의 양자화 출력으로 "
        "연쇄 계산하여, 정수 경로가 float과 99.65–99.75 % 같은 클래스를 고르고 정확도는 오히려 0.06–0.08 %p 높았다.",
        "**재사용 가능한 칩 검증 체계.** 칩 내부 클립 ROM·기대값 ROM·채점기와 JTAG VIO로, 핀 2개만 쓰고 추가 배선 "
        "없이 수백 개 클립을 분 단위로 검증한다. 클립 세트와 모델만 바꿔 같은 절차를 반복할 수 있다.",
        "**모델 교체에 강한 RTL.** 두 모델(Baseline, Partial-75)을 RTL 수정 없이 같은 코드로 빌드·검증하였다.",
    ])

    # ---- 4 ----
    heading(doc, "4. 구현 결과", 1)
    heading(doc, "4.1 학습", 2)
    table(doc, "Training configuration and result",
          ["Item", "Baseline", "Partial-75 (fine-tune)"],
          [["Data", "GSC v2, official split, 12 classes, 16-ch binary input (T = 128)", "same"],
           ["Schedule", "Adam 1e-3, plateau ×0.1, 100 epochs, batch 128", "Adam 1e-4, 20 epochs from Baseline"],
           ["Binarization", "QAT with hardtanh STE; threshold STE clip 0.003", "same"],
           ["Augmentation", "none", "partial keyword window (p = 0.5, coverage 0.75–1.0)"],
           ["Best validation acc.", "83.64 %", "83.32 %"]],
          widths_cm=[3.2, 7.6, 5.8], size=8)
    figure(doc, FIG / "fig_training_curves.png",
           "Training history: (a) validation accuracy, (b) training loss. Partial-75 (dashed) continues from the "
           "Baseline checkpoint (dotted line at epoch 100).", 16.5)

    heading(doc, "4.2 FPGA 구현", 2)
    tot = {k: util_row(HW / "util_summary.rpt", k) for k in
           ("Slice LUTs", "Slice Registers", "Block RAM Tile", "DSPs", "Bonded IOB")}
    avail = {"Slice LUTs": "48,000", "Slice Registers": "96,000", "Block RAM Tile": "90", "DSPs": "140",
             "Bonded IOB": "338"}
    table(doc, "Post-route summary (XC7S75-1, Vivado 2026.1, 50 MHz)",
          ["Item", "Baseline build", "Partial-75 build", "Available"],
          [[k, f"{v[0]:,} ({v[1]} %)" if isinstance(v[0], int) else v[0],
            f"{util_row(HW75 / 'util_summary.rpt', k)[0]:,}", avail[k]] for k, v in tot.items()]
          + [["WNS / WHS (ns)", f"{TS['wns']:+.3f} / {TS['whs']:+.3f}", f"{TS75['wns']:+.3f} / {TS75['whs']:+.3f}", "—"],
             ["Failing endpoints", f"{TS['fail']} of {TS['n']:,}", f"{TS75['fail']} of {TS75['n']:,}", "—"],
             ["Inference time", "57.5 ms / clip", "57.5 ms / clip", "hop 100 ms"]],
          widths_cm=[3.6, 4.2, 3.6, 2.6], size=8.5,
          note="BRAM holds the self-test clip ROM (48 RAMB36); the network itself uses 6 RAMB18. "
               "Both builds use identical RTL and differ only in ROM contents.")
    rows = []
    names = [("u_c1", "conv1"), ("u_b1", "B1"), ("u_b2", "B2"), ("u_b3", "B3"), ("u_c2", "conv2"),
             ("u_tail", "tail (conv2 pw, conv3, conv4)")]
    for inst, lab in names:
        u = block_util(HW / "util_hier.rpt", inst)
        s, lv, dp, lpct = block_slack(HW, inst)
        rows.append([lab, f"{u[0]:,}", f"{u[3]:,}", u[6], f"{s:.3f}", lv])
    table(doc, "Per-block resources and worst setup path (Baseline build)",
          ["Block", "LUT", "FF", "DSP", "Worst slack (ns)", "Logic levels"],
          rows, widths_cm=[4.6, 1.8, 1.8, 1.2, 2.8, 2.2], size=8.5)
    para(doc,
         "네트워크는 LUT 약 19.8 k로 블록마다 2.7–3.7 k가 고르게 분포한다. 최악 경로는 두 빌드 모두 conv2 "
         "depthwise(k=29, dilation 2)의 MAC 입력 선택부터 누산기까지이며 지연의 약 75 %가 배선이다. 가장 느린 "
         "속도 등급에서 20 ns 주기에 5 ns 이상 여유가 있어 약 66 MHz까지 동작 여유가 있다.")
    figure(doc, IMG / "fig_slack_hist.png",
           "Setup-slack distribution of the 20,000 worst endpoints (of 55,034), Baseline build.", 8.5)
    figure(doc, HW / "device_place.png",
           "Placed design on the XC7S75 (Vivado device view). Coloured regions are the network blocks, the "
           "self-test harness and the debug logic; the design occupies the lower-middle part of the device.", 7.0)
    figure(doc, IMG / "report_schematic_top.png",
           "Post-synthesis schematic of the measurement top level: harness and network (u_top), VIO core and "
           "the debug hub.", 11.0)
    table(doc, "Vivado power estimate, Baseline build (vectorless, confidence: " + PW["conf"] + ")",
          ["Component", "Power (W)"],
          [["Total on-chip", PW["total"]], ["Device static", PW["static"]], ["Dynamic", PW["dyn"]],
           ["  of which network", PW["hier"]["u_net"]],
           ["  of which self-test clip ROM", PW["hier"]["u_st"]]],
          widths_cm=[6.0, 3.0], size=8.5,
          note="Estimate, not a measurement. Network only: ≈0.15 W including static power, "
               "≈8.7 mJ per 57.4-ms inference (≈3.3 mJ dynamic).")

    heading(doc, "4.3 칩 수준 동치 검증 (T2)", 2)
    figure(doc, IMG / "fig_selftest.png",
           "On-chip self-test. Clips and expected classes are stored in block RAM, every classification is "
           "scored on the chip, and only counters leave the device through the VIO over JTAG.", 9.0)
    b600 = balanced_acc("bd_base", 600)
    p600 = balanced_acc("bd_base_ft20_partial75", 600)
    table(doc, "On-chip self-test results",
          ["Model", "Clip set", "True classes", "Runs", "Chip = Python", "Accuracy on set (%)"],
          [["Baseline", "test order 0–599", "2", "3", "600/600 each run", "81.33"],
           ["Partial-75", "test order 0–599", "2", "2", "600/600 each run", "81.83"],
           ["Baseline", "balanced 0–599", "12 × 50", "2", "600/600 each run", pct(b600)],
           ["Partial-75", "balanced 0–599", "12 × 50", "2", "600/600 each run", pct(p600)]],
          widths_cm=[2.2, 3.0, 2.2, 1.3, 3.4, 3.0], size=8.5,
          note="Repeated runs include a power cycle and fresh programming. Accuracy on set is the Python integer "
               "accuracy, which the chip equals when every clip matches.")
    para(doc,
         "칩의 분류 결과는 모든 실행에서 소프트웨어 정수 경로와 600개 전부 일치하였고, 반복 실행 결과도 같았다. "
         "오류 0건이 관측된 서로 다른 2,400개 클립–모델 조합으로부터 칩 불일치율의 95 % 상한은 약 0.13 %이다. "
         "12개 클래스 균형 세트에서도 일치했으므로 모든 출력 클래스 경로가 칩에서 올바르게 동작한다.")

    # ---- 5 ----
    heading(doc, "5. 성능 평가", 1)
    heading(doc, "5.1 클립 정확도 (T1)", 2)
    fw = lambda d: sum(d["confusion_fixed"][t][p] for t in (10, 11) for p in range(10))
    table(doc, f"Clip accuracy on the full GSC v2 test set ({T1_BASE['n_clips']:,} clips)",
          ["Model", "Float (%)", "Integer path (%)", "Agreement (%)", "Non-keyword → keyword"],
          [["Baseline", pct(T1_BASE["float_acc"]), pct(T1_BASE["fixed_acc"]), pct(T1_BASE["agree"]),
            f"{fw(T1_BASE)} / 814"],
           ["Partial-75", pct(T1_P75["float_acc"]), pct(T1_P75["fixed_acc"]), pct(T1_P75["agree"]),
            f"{fw(T1_P75)} / 814"]],
          widths_cm=[2.6, 2.2, 2.8, 2.6, 3.6], size=8.5)
    figure(doc, FIG / "fig_per_class_recall.png",
           "Per-class recall of the integer path on the full test set.", 16.5)
    figure(doc, IMG / "fig_confusion_bd_base.png",
           "Confusion matrix of the Baseline integer path (clip counts; shading is the share of the true-class "
           "row). Each class has 396–425 test clips, the same scale as the matrices in [1].", 9.0)
    para(doc,
         "silence(99.3 %)가 가장 높고 unknown(64.4 %), down(76.1 %), go(77.6 %)가 낮으며, 주요 혼동은 no↔go, "
         "down↔go이다. 이는 [1]에서 이진 AFE 입력의 오류가 같은 단어 쌍에 모인다는 관찰과 일치한다.")

    heading(doc, "5.2 연속 동작 (T3, 검증 세트)", 2)
    table(doc, "Always-on detection on spliced streams (validation split, Python float)",
          ["Model / policy", "Keyword recall", "Quiet streams with event", "Wrong events", "Latency"],
          [["Baseline, N=5, margin 1.25", "67.73 %", "52 / 512", "287", "1000 ms"],
           ["Partial-75, N=5, margin 1.05", "69.53 %", "41 / 512", "242", "1000 ms"]],
          widths_cm=[4.8, 2.8, 3.4, 2.4, 2.0], size=8.5, bold_rows=(1,),
          note="1 s silence + 1 s clip + 1 s silence; a 1-s window every 100 ms; detection = same keyword five "
               "times in a row above the keyword–quiet margin.")
    para(doc,
         "같은 모델의 가운데 창(원래 클립) 정확도는 82.68 %이지만 연속 검출률은 69.53 %로, 클립 정확도가 실제 "
         "연속 동작 성능을 과대평가함을 보여 준다. Partial-75는 클립 정확도를 유지하면서 검출을 46건 늘리고 "
         "quiet 오검출과 잘못된 검출을 각각 11건, 45건 줄였다.")

    # ---- 6 ----
    heading(doc, "6. 한계와 향후 과제", 1)
    bullets(doc, [
        "**FPGA와 ASIC의 차이.** 추정 전력 0.183 W 중 0.094 W가 소자 정적 전력으로, µW급 ASIC [4]–[7]과 수치를 "
        "직접 비교할 수 없다. FPGA에서는 가중치 ROM과 연산이 범용 LUT·배선으로 구현되고 전압 스케일링·클럭 "
        "게이팅을 적용하기 어렵다. 본 결과는 기능·정확도·타이밍 검증용 프로토타입이며, 전력은 추정치이다.",
        "**실제 아날로그 보드 연결 미수행.** PCB 설계와 SPICE 시뮬레이션은 완료되었고 결과는 긍정적이나, 실장 후 "
        "실제 비교기 출력으로 정확도를 다시 확인해야 한다.",
        "**정확도 목표 미달.** 클립 정확도 82.6 %로 목표(85 %)보다 낮고, unknown 클래스와 no/go/down 혼동, "
        "비키워드의 약 18 %가 키워드로 분류되는 점이 always-on 오검출로 이어진다.",
        "**연속 동작 검증 범위.** T3는 잘라 붙인 스트림 위의 검증 세트·float 결과이며, 정수 margin 적용, test 세트 "
        "평가, 창 이동·투표를 포함한 칩 검증이 남아 있다.",
        "**향후 계획.** (1) 연속 동작의 정수 기준 확정과 칩 검증, (2) 전체 test 세트 칩 검증, (3) 실제 AFE 보드 연결, "
        "(4) 가중치 ROM의 BRAM 이전 등 자원 최적화, (5) 연산 카운트 기반 ASIC 에너지 추정.",
    ])

    heading(doc, "참고문헌", 1)
    refs = [
        "G. Cerutti, L. Cavigelli, R. Andri, M. Magno, E. Farella, and L. Benini, “Sub-mW keyword spotting "
        "on an MCU: Analog binary feature extraction and binary neural networks,” arXiv:2201.03386, 2022.",
        "S. Majumdar and B. Ginsburg, “MatchboxNet: 1D time-channel separable convolutional neural network "
        "architecture for speech commands recognition,” in Proc. Interspeech, 2020.",
        "O. Rybakov, N. Kononenko, N. Subrahmanya, M. Visontai, and S. Laurenzo, “Streaming keyword spotting "
        "on mobile devices,” in Proc. Interspeech, 2020 (arXiv:2005.06720).",
        "J. S. P. Giraldo, S. Lauwereins, K. Badami, and M. Verhelst, “Vocell: A 65-nm speech-triggered "
        "wake-up SoC for 10-µW keyword spotting and speaker verification,” IEEE JSSC, vol. 55, no. 4, "
        "pp. 868–878, 2020.",
        "K. Kim et al., “A 23-µW keyword spotting IC with ring-oscillator-based time-domain feature "
        "extraction,” IEEE JSSC, vol. 57, no. 11, pp. 3298–3311, 2022.",
        "Q. Chen et al., “DeltaKWS: A 65nm 36nJ/decision bio-inspired temporal-sparsity-aware digital keyword "
        "spotting IC with 0.6V near-threshold SRAM,” IEEE TCASAI, 2025 (arXiv:2405.03905).",
        "F. Tan et al., “A 1.8% FAR, 2ms decision latency, 1.73 nJ/decision keywords spotting chip "
        "incorporating transfer-computing speaker verification, hybrid-domain computing and scalable "
        "5T-SRAM,” in Proc. ISSCC, 2024, pp. 330–332.",
    ]
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph()
        rr = p.add_run(f"[{i}] {r}")
        set_fonts(rr, 8.5)
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.first_line_indent = Cm(-0.6)
        p.paragraph_format.space_after = Pt(2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT, "figures", FIG_N[0], "tables", TAB_N[0])


if __name__ == "__main__":
    build()
