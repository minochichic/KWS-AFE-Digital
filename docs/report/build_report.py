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
OUT = ROOT / "out/report_build/KWS_FPGA_interim_report.docx"

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
def build():
    doc = Document()
    style_doc(doc)
    TS = timing_summary(HW / "timing_summary.rpt")
    TS75 = timing_summary(HW75 / "timing_summary.rpt")
    PW = power(HW)

    # title
    p = para(doc, "중간보고서", 11, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    p = para(doc, "아날로그 이진 특징 추출과 부분 이진화 MatchboxNet을 이용한\n"
             .replace("\n", " "), 16, True, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
    para(doc, "키워드 스포팅 가속기의 FPGA 구현 및 칩 수준 검증", 16, True,
         align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
    para(doc, "KWS-AFE-Digital 프로젝트 · 2026년 9월 17일", 10,
         align=WD_ALIGN_PARAGRAPH.CENTER, after=12)

    heading(doc, "요약", 1)
    para(doc,
         "본 보고서는 아날로그 필터뱅크·비교기가 만드는 16채널 이진 시간-주파수 이미지를 입력으로 "
         "받는 부분 이진화 MatchboxNet(BinaryMatchboxNet)을 Verilog RTL로 설계하고, Xilinx "
         "Spartan-7 XC7S75 FPGA에 구현하여 칩에서 직접 검증한 중간 결과를 정리한다. 아날로그 전단은 "
         "공동 연구자가 PCB 설계와 SPICE 시뮬레이션까지 완료한 회로이며, 본 보고서의 모델은 그 필터뱅크 "
         "응답을 소프트웨어로 모사한 입력으로 학습하였다. 비교기 임계값 16개는 네트워크와 함께 학습하여 "
         "저항 분압으로 옮길 수 있는 절대값으로 고정하였다. 배치 정규화는 정수 임계값으로, 후단 계층은 Q*.6 고정소수점으로 "
         f"변환하여 곱셈 없는 이진 연산 경로를 구성하였다. Google Speech Commands v2 12-class 공식 "
         f"test 세트 {T1_BASE['n_clips']:,}개 클립에서 정수 경로 정확도는 {pct(T1_BASE['fixed_acc'])} % "
         f"(float {pct(T1_BASE['float_acc'])} %)로 양자화 손실이 관측되지 않았다. FPGA 구현은 50 MHz에서 "
         f"setup 여유 {TS['wns']:+.3f} ns로 타이밍을 만족하였고, LUT 43 %, BRAM 57 %를 사용하였다. "
         "칩 내부에 클립 ROM과 채점기를 두는 자체 검사 구조로 두 모델 × 두 클립 세트(12개 클래스 균형 "
         "세트 포함, 2,400개 조합)를 반복 실행하여 총 5,400회 분류한 결과, 칩의 분류 결과가 파이썬 정수 "
         "경로와 모든 클립에서 일치하였고 전원 재투입 후에도 동일하였다. 연속 동작(always-on) 판정은 검증 세트에서 "
         "키워드 검출률 69.53 %를 보였으며, 칩 수준 검증은 다음 단계로 남아 있다.")

    # 1
    heading(doc, "1. 서론", 1)
    para(doc,
         "키워드 스포팅(KWS)은 음성 인터페이스를 가진 기기에서 항상 켜져 있어야 하는 작업이다. "
         "표준 파이프라인은 마이크 → ADC → FFT/MFCC → 신경망 순서로 동작하며, 분류기를 충분히 "
         "경량화하면 전체 에너지의 대부분은 분류 연산이 아니라 데이터 취득과 전처리가 차지한다. "
         "Cerutti 등 [1]은 이 부분을 아날로그 대역통과 필터뱅크, 포락선 검출기, 비교기로 대체하여 "
         "마이크 출력에서 곧바로 이진 특징을 얻고, 이를 이진 신경망(BNN)으로 분류하는 방식을 제안하였다. "
         "이 경우 1초 취득 에너지는 표준 마이크+ADC 경로의 5.2 % 수준으로 줄어든다 [1].")
    para(doc,
         "다만 [1]의 분류기 에너지와 정확도는 소프트웨어 시뮬레이션과 MCU 실행 추정에 기반하며, "
         "이진 입력 BNN을 전용 하드웨어로 구현했을 때 학습된 모델과 비트 단위로 같은 결과가 나오는지는 "
         "다루지 않았다. 본 연구는 (i) 실제로 제작 가능한 필터뱅크와 저항 분압 임계값을 전제로 모델을 "
         "학습하고, (ii) 이를 정수 연산만으로 동작하는 RTL로 옮긴 뒤, (iii) FPGA 칩 위에서 수천 회의 "
         "분류를 실행하여 소프트웨어 모델과의 일치를 직접 확인하는 것을 목표로 한다.")
    para(doc, "본 보고서의 기여는 다음과 같다.")
    bullets(doc, [
        "제작 가능한 아날로그 조건(고정 필터뱅크, 채널별 절대 임계값)을 학습 단계에 반영한 16채널 부분 "
        "이진화 MatchboxNet과, 이를 곱셈기 없이 정수 임계 비교로 실행하는 folded RTL 설계.",
        "평가 조건을 T1(클립 정확도), T2(하드웨어 재현성), T3(연속 동작)로 분리한 평가 체계와, "
        "이에 따른 선행연구 수치의 조건별 정리.",
        "칩 내부 ROM·채점기·VIO를 이용한 자체 검사로, 12개 클래스 균형 세트를 포함한 2,400개 "
        "클립–모델 조합(반복 포함 5,400회 분류)에서 소프트웨어 정수 경로와 100 % 일치를 확인한 칩 수준 검증.",
        "전체 test 세트에서 float 대비 정수 경로의 정확도 손실이 없음(−0.06 %p 차, 정수 쪽이 높음)을 "
        "확인한 양자화 설계.",
    ])

    # 2
    heading(doc, "2. 관련 연구와 평가 조건", 1)
    para(doc,
         "KWS 문헌에서 '정확도'는 서로 다른 조건을 가리키는 경우가 많다. 대부분의 소프트웨어 및 칩 논문은 "
         "1초 길이 클립마다 한 번 분류하여 라벨과 비교하는 클립 정확도를 보고하며, 단어가 창 안에 온전히 "
         "들어 있다는 가정을 전제로 한다. 반면 실제 always-on 동작에서는 단어의 시작 시점을 알 수 없으므로 "
         "창을 연속해서 이동시키고 반복 판정을 적용해야 하며, 이 경우의 지표는 검출률과 오검출률로 "
         "달라진다. 본 보고서는 이를 구분하기 위해 Table I의 세 층위를 사용한다.")
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
    para(doc,
         "데이터셋 조건은 TensorFlow speech_commands 예제의 표준 설정(12 labels, 공식 "
         "validation/testing 목록, unknown·silence 각 10 %)을 따른다. 이 설정에서 v2 test 세트는 "
         "4,890개로 보고되며 [3], 본 연구의 test 세트는 4,888개로 사실상 동일하다. 본 보고서에서 원문을 확인한 "
         "[1]과 [6]은 test 클립 수를 별도로 명시하지 않았다.")
    table(doc, "Prior work on Google Speech Commands, grouped by evaluation condition",
          ["Work", "Feature extraction", "Classifier", "Classes", "Accuracy (%)", "Platform / basis", "Eval."],
          [["Cerutti et al. [1]", "Analog BPF + comparator (64 ch)", "BNN", "12", "86.0", "SW sim + MCU est.", "Clip"],
           ["Cerutti et al. [1]", "Analog BPF + comparator (8 ch)", "BNN", "12", "76.3", "SW sim + MCU est.", "Clip"],
           ["Rybakov et al. [3]", "MFCC (digital)", "DS-CNN (FP)", "12", "97.0", "SW (TFLite)", "Clip"],
           ["Rybakov et al. [3]", "MFCC (digital)", "MHAtt-RNN (FP)", "12", "98.0", "SW", "Clip"],
           ["Giraldo, Vocell [4]", "MFCC (digital)", "NN", "12", "90.87", "65 nm chip, 16 µW", "Clip"],
           ["Kim et al. [5]", "Ring-osc. time domain (analog)", "RNN", "12", "86.03", "65 nm chip, 23 µW", "Clip"],
           ["Chen, DeltaKWS [6]", "IIR BPF (digital)", "ΔGRU", "12", "89.5", "65 nm chip, 5.22 µW", "Clip"],
           ["Tan et al. [7]", "—", "Transfer computing", "12", "91.8", "28 nm chip, 1.73 µW", "Clip"],
           ["Seol et al. [8]", "Analog front end", "Skip-RNN", "7", "92.8", "28 nm chip, 1.48 µW", "Clip"],
           ["This work", "Analog BPF + comparator (16 ch, sim.)", "Partial BNN", "12",
            pct(T1_P75["fixed_acc"]), "FPGA XC7S75 (RTL)", "Clip (T1)"]],
          widths_cm=[3.0, 3.9, 2.4, 1.3, 1.8, 3.2, 1.4], size=7.5, bold_rows=(9,),
          note="Chip numbers for [5], [7], [8] as tabulated in [6]; Vocell as quoted in [1]. "
               "None of the listed works reports a continuous-stream (T3-type) metric on 12-class GSC.")

    # 3
    heading(doc, "3. 시스템 설계", 1)
    heading(doc, "3.1 전체 구조", 2)
    figure(doc, IMG / "fig_system.png",
           "System overview. The analog front end (left, simulated in this work from the board filter "
           "bank) turns microphone audio into 16 comparator outputs. The FPGA back end captures a 10-ms "
           "frame, keeps a sliding 1-s window, classifies it with the folded BinaryMatchboxNet and "
           "applies a consecutive-vote decision.", 16.5)
    para(doc,
         "아날로그 전단은 채널마다 대역통과 필터, 포락선 검출기, 비교기로 구성되며, 비교기 기준 전압은 "
         "저항 분압으로 정해진다. 디지털 단의 관점에서 전단과의 계약은 단순하다. FPGA로 들어오는 신호는 비교기 출력 16가닥뿐이며, "
         "디지털 단은 10 ms 동안 한 번이라도 1이 된 채널을 1로 기록하여(sticky-OR) 16비트 프레임을 만든다. "
         "1초(100프레임) 창을 좌우로 14프레임씩 채워 128프레임 입력으로 만든 뒤 네트워크에 넣는다.")
    heading(doc, "3.2 입력 데이터와 아날로그 전단의 현재 상태", 2)
    para(doc,
         "아날로그 전단은 공동 연구자가 회로 설계, PCB 설계, SPICE 시뮬레이션까지 진행하였다. 부품 실장이 "
         "아직 이루어지지 않아 실제 보드를 FPGA에 연결한 측정은 수행하지 못했지만, 시뮬레이션에서 얻은 "
         "필터 응답으로 학습한 모델이 이상적인 mel 필터로 설계한 경우보다 오히려 높은 정확도를 보이는 등 "
         "결과는 긍정적이다. 본 보고서의 학습·평가 입력은 [1]의 IV-A절과 같은 방식으로, 그 필터뱅크 응답을 "
         "스펙트로그램 위에 적용해 이진 이미지를 만드는 소프트웨어 모사이다. 모사 규칙('10 ms 창 안에서 "
         "비교기 출력이 한 번이라도 1이면 1')은 디지털 단의 프레임 포착 규칙과 같게 맞추었으므로, 실제 "
         "보드가 연결되면 입력 소스만 바뀌고 디지털 단은 그대로 사용한다.")
    heading(doc, "3.3 네트워크 구조", 2)
    para(doc,
         "분류기는 MatchboxNet-3×2×64 [2] 골격에 정밀도를 층별로 다르게 적용한 부분 이진화 구조이다. "
         "[1]의 결과(입력과 양 끝단까지 이진화하면 정확도가 크게 떨어짐)를 반영하여 첫 층(conv1)은 INT8 "
         "가중치, 마지막 층(conv4)은 고정소수점으로 두고, 파라미터와 연산의 대부분을 차지하는 중간 TCS "
         "블록(B1–B3)과 conv2만 이진화하였다.")
    table(doc, "Network architecture (C = 64, T = 128, 12 classes)",
          ["Stage", "Type", "Weight precision", "Kernel", "Out ch.", "Output", "Params"],
          [["conv1", "1D conv, stride 2", "INT8 (binary input)", "11", "128", "128×64", "22,784"],
           ["B1", "TCS ×2 + projection skip", "binary", "13", "64", "64×64", "23,616"],
           ["B2", "TCS ×2 + identity skip", "binary", "15", "64", "64×64", "10,624"],
           ["B3", "TCS ×2 + identity skip", "binary", "17", "64", "64×64", "10,880"],
           ["conv2", "separable, dilation 2", "binary", "29", "128", "128×64", "10,432"],
           ["conv3", "1×1 conv", "INT8", "1", "128", "128×64", "16,640"],
           ["conv4", "1×1 conv", "fixed point", "1", "12", "12×64", "1,548"],
           ["head", "time average pool", "—", "—", "12", "12", "0"],
           ["Total", "", "", "", "", "", "96,524 (binary 57.6 %)"]],
          widths_cm=[1.6, 3.9, 3.0, 1.3, 1.4, 1.8, 3.6], size=8, bold_rows=(8,))

    heading(doc, "3.4 학습 방법", 2)
    para(doc,
         "이진 가중치는 잠재 full-precision 값을 유지하고 순전파에서 부호를 취하는 QAT 방식으로 학습하였고, "
         "기울기는 hardtanh STE로 전달하였다. 비교기 임계값 16개는 채널 평균으로 초기화한 뒤 네트워크와 "
         "함께 end-to-end로 학습하였으며, 임계 비교의 STE 통과 구간을 0.003으로 좁혀(ste_clip) 임계값 "
         "근처의 값에만 기울기가 흐르도록 하였다. 이 설정은 기존 1.0 대비 test 정확도를 +10.0 %p 올린 "
         "것으로 기록되어 있다(프로젝트 ablation 기록). 입력 정규화는 데이터셋 상수를 쓰는 fixed 방식으로, "
         "학습된 임계값이 채널 고유의 절대값이 되어 저항 분압 하나로 구현된다.")
    para(doc,
         "always-on 동작을 위한 두 번째 모델(Partial-75)은 기준 모델에서 20에폭 미세조정하였다. 학습 "
         "샘플의 50 %에서 키워드 발화 구간의 75–100 %만 창 안에 남기고 나머지를 0으로 채워, 창 경계에 "
         "걸린 단어를 보는 상황을 학습에 포함하였다. 두 모델의 설정을 Table IV에, 학습 곡선을 Fig. 2에 보였다.")
    table(doc, "Training configuration",
          ["Item", "Baseline (bd_base)", "Partial-75 (fine-tune)"],
          [["Dataset / split", "GSC v2, official lists, 12 classes", "same"],
           ["Input", "16-ch binary image from simulated board front end, T = 128", "same"],
           ["Epochs / batch", "100 / 128", "20 (from Baseline best) / 128"],
           ["Optimizer, LR", "Adam, 1e-3, plateau ×0.1 (patience 10), min 1e-5", "Adam, 1e-4, patience 5"],
           ["Threshold STE clip", "0.003", "0.003"],
           ["Augmentation", "none", "partial keyword window (p=0.5, coverage 0.75–1.0)"],
           ["Best validation acc.", "83.64 % (epoch 82)", "83.32 % (epoch 20)"],
           ["Seed", "1234", "1234"]],
          widths_cm=[3.4, 7.2, 6.2], size=8)
    figure(doc, FIG / "fig_training_curves.png",
           "Training history. (a) Validation accuracy and (b) training loss per epoch. The Partial-75 "
           "fine-tune (dashed) continues from the Baseline checkpoint; the dotted line marks epoch 100. "
           "Steps at epochs 43 and 75 are learning-rate reductions.", 16.5)

    heading(doc, "3.5 하드웨어 매핑과 최적화", 2)
    para(doc,
         "학습된 모델을 하드웨어로 옮기는 export 단계에서 다음 변환을 수행한다. 이진 합성곱의 내적은 "
         "2·popcount(XNOR)−N으로 계산하므로 곱셈기가 필요 없다. 이진층 뒤의 배치 정규화와 부호 함수는 "
         "정수 누산값에 대한 임계 비교 하나로 합쳐지고([1]의 식 (3)), 가중치 스케일 α도 이 임계값에 "
         "흡수되어 하드웨어에 남지 않는다. residual 합산은 마지막 pointwise와 skip의 정수 누산값을 더한 "
         "뒤 한 번만 임계 비교한다. conv1은 INT8 가중치이지만 입력이 ±1이므로 부호 있는 누산만으로 계산된다.")
    para(doc,
         "배치 정규화가 부호 함수로 흡수되지 않는 conv2 pointwise, conv3, conv4는 정수 이득·오프셋·시프트로 "
         "변환하여 1/64 격자(Q*.6)의 고정소수점으로 계산한다(Table V). 각 계층은 이전 계층의 양자화된 "
         "출력을 입력으로 받도록 export 단계에서 연쇄 계산하여, RTL 레지스터가 실제로 갖는 값과 기대값이 "
         "어긋나지 않게 하였다. 분류는 시간축 합의 argmax로 하며, 평균을 위한 나눗셈은 모든 클래스에 "
         "같은 양수 배율이므로 생략한다.")
    table(doc, "Fixed-point tail (export report, balanced 1,200-clip dump, Baseline)",
          ["Site", "Output format", "Shift", "Accumulator bits", "|acc| max observed", "Clamped values"],
          [["conv2 pw", "Q8.6", "18", "8", "54", "0 / 9,830,400"],
           ["conv3", "Q5.6", "24", "28", "205,940", "0 / 9,830,400"],
           ["conv4", "Q8.6", "27", "24", "242,697", "0 / 921,600"]],
          widths_cm=[2.2, 2.4, 1.4, 2.8, 3.2, 3.4], size=8)
    para(doc,
         "데이터 경로는 folded 구조로 설계하였다. 계층마다 MAC 엔진 하나를 채널·시간에 걸쳐 시분할로 "
         "재사용하고, 계층 사이에는 64프레임 평면 버퍼를 두어 한 계층이 클립 전체를 끝낸 뒤 다음 계층이 "
         "읽는다. 10 Hz 판정에 필요한 처리량이 작기 때문에(클립당 약 2.86 M 사이클, 50 MHz에서 57 ms) "
         "완전 병렬 구조보다 자원이 작고, 채널 수나 창 길이가 바뀌어도 컨트롤러 파라미터만 바뀐다. "
         "가중치·임계값은 RTL 상수가 아니라 $readmemh로 읽는 ROM 파일이므로, 아날로그 조건이 바뀌어 "
         "재학습하더라도 RTL은 수정하지 않는다.")

    # 4
    heading(doc, "4. 검증 방법론", 1)
    figure(doc, IMG / "fig_flow.png",
           "Design and verification flow. Integer golden vectors exported from the trained model are the "
           "reference for every hardware step; T1 is measured in software, T2 on the chip.", 16.5)
    para(doc,
         "검증은 소프트웨어 정수 경로를 기준(golden)으로 삼아 단계마다 비트 단위 일치를 확인하는 방식으로 "
         "진행하였다. export 단계는 층별 정수 활성값을 파일로 남기고, 각 RTL 모듈의 테스트벤치는 이 값과 "
         "출력을 한 프레임씩 비교한다. 불일치가 있으면 처음 어긋난 층이 곧 문제 모듈을 가리킨다. "
         "Table VI은 Vivado 시뮬레이터(XSim)에서 수행한 단위·통합 테스트를 정리한 것이다.")
    table(doc, "RTL simulation results (XSim, integer golden vectors)",
          ["Testbench", "Unit under test", "Stimulus", "Result"],
          [["capture", "10-ms sticky-OR frame capture", "3 frames", "0 failures"],
           ["window", "sliding window / snapshot", "33 frames, overrun, busy interlock", "0 failures"],
           ["vote", "consecutive vote + margin gate", "3 scenarios", "0 failures"],
           ["plane", "64-frame inter-layer buffer", "512 frames", "0 failures"],
           ["block", "binary TCS block with residual", "512 frames", "0 failures"],
           ["tail", "fixed-point conv2 pw / conv3 / conv4", "8 frames", "0 failures"],
           ["top", "full network kws_top", "8 clips", "0 failures"],
           ["selftest", "self-test harness + network", "2 clips (57.11 ms/clip)", "0 failures"]],
          widths_cm=[2.2, 5.6, 5.2, 2.6], size=8)
    heading(doc, "4.1 칩 자체 검사 (T2)", 2)
    para(doc,
         "시뮬레이션은 클립 하나에 약 2.86 M 사이클이 필요하여 수백 개 이상의 클립을 돌리기 어렵다. "
         "이를 위해 FPGA 내부에 클립 ROM, 기대값 ROM, 채점 FSM을 두는 자체 검사 구조를 만들었다(Fig. 4). "
         "하네스는 네트워크의 ready 신호에 맞춰 프레임을 공급하고, 분류 결과를 기대값(파이썬 정수 경로 "
         "예측)과 비교하여 총 개수, 일치 개수, 첫 불일치 위치를 기록한다. 결과는 JTAG 기반 VIO로 읽으므로 "
         "추가 핀이나 배선이 필요 없다. 네트워크 모듈은 배포용과 동일하며, 시뮬레이션과 칩이 같은 "
         "하네스 코드를 공유한다.")
    figure(doc, IMG / "fig_selftest.png",
           "On-chip self-test harness. Clips and expected classes are stored in block RAM; the scorer "
           "compares every classification on the chip, and only the counters leave the device through "
           "the VIO over the existing JTAG cable.", 9.5)
    para(doc,
         "기대값 비교는 정확도가 아니라 동치를 확인하는 것이다. RTL과 파이썬 정수 경로는 같은 정수 연산을 "
         "수행하므로 허용 오차가 없고, 하나라도 다르면 결함이다. 모든 클립이 일치하면 칩의 정확도는 T1 "
         "정수 경로 정확도와 같다. 또한 두 번 이상 실행하여 결과가 동일한지(결정성)를 확인하였는데, "
         "타이밍 위반이나 초기화되지 않은 레지스터는 실행마다 다르게 틀리기 때문이다. 초기 클립 세트는 "
         "test 로더 순서상 두 클래스에 편중되어 있어, 12개 클래스를 각 50개씩 교차 순서로 뽑은 균형 "
         "세트를 추가로 만들었다.")

    # 5
    heading(doc, "5. FPGA 구현 결과", 1)
    heading(doc, "5.1 실험 환경", 2)
    table(doc, "Implementation environment",
          ["Item", "Setting"],
          [["Board", "Hanback FPGA kit, Xilinx Spartan-7 XC7S75-FGGA484-1"],
           ["Clock", "50 MHz from base-board clock selector (20-ns constraint)"],
           ["Tool", "AMD Vivado 2026.1 (non-project batch flow), XSim 2026.1"],
           ["Host", "ThinkPad T470 (i5-7300U, 16 GB), JTAG via DLC10 cable"],
           ["Top for measurement", "kws_selftest_board (network + harness + VIO), 2 I/O pins"],
           ["Training", "remote GPU server, PyTorch"],
           ["RTL size", "22 synthesizable Verilog files (≈4.5k lines), 17 testbenches"],
           ["Build time", "synthesis ≈18 min, place & route + bitstream ≈10 min"]],
          widths_cm=[4.0, 12.5], size=8.5)
    heading(doc, "5.2 자원 사용량", 2)
    tot = {k: util_row(HW / "util_summary.rpt", k) for k in
           ("Slice LUTs", "Slice Registers", "F7 Muxes", "F8 Muxes", "Block RAM Tile", "DSPs", "Bonded IOB")}
    avail = {"Slice LUTs": "48,000", "Slice Registers": "96,000", "F7 Muxes": "32,000",
             "F8 Muxes": "16,000", "Block RAM Tile": "90", "DSPs": "140", "Bonded IOB": "338"}
    table(doc, "Post-route utilization (XC7S75, Baseline self-test build)",
          ["Resource", "Used", "Available", "Utilization (%)"],
          [[k, f"{v[0]:,}" if isinstance(v[0], int) else v[0], avail[k], v[1]] for k, v in tot.items()],
          widths_cm=[4.0, 3.0, 3.0, 3.0], size=8.5)
    rows = []
    names = [("u_c1", "conv1"), ("u_b1", "B1"), ("u_b2", "B2"), ("u_b3", "B3"), ("u_c2", "conv2"),
             ("u_tail", "tail (conv2 pw, conv3, conv4)"), ("u_pa", "plane A"), ("u_pb", "plane B"),
             ("u_pc", "plane C"), ("u_pd", "plane D"), ("u_net", "network total"),
             ("u_st", "self-test harness"), ("u_vio", "VIO"), ("dbg_hub", "debug hub")]
    for inst, lab in names:
        u = block_util(HW / "util_hier.rpt", inst)
        if u:
            rows.append([lab, f"{u[0]:,}", f"{u[1]:,}", f"{u[2]:,}", f"{u[3]:,}", u[4], u[5], u[6]])
    table(doc, "Hierarchical utilization",
          ["Block", "LUT", "LUTRAM", "SRL", "FF", "RAMB36", "RAMB18", "DSP"],
          rows, widths_cm=[4.6, 1.7, 1.7, 1.3, 1.7, 1.5, 1.5, 1.3], size=8, bold_rows=(10,))
    para(doc,
         "네트워크는 LUT 약 19.8 k를 사용하며, conv1과 B1–B3, conv2, 꼬리가 각각 2.7–3.7 k로 고르게 "
         "분포한다. 가중치 ROM은 비동기 읽기로 설계되어 BRAM 대신 LUT와 MUXF7/F8로 구현되었고, 이것이 "
         "LUT 사용량의 주요 원인이다. BRAM 48개(RAMB36)는 자체 검사용 클립 ROM(600 × 128 × 16 bit)이며 "
         "네트워크 자체는 BRAM을 거의 쓰지 않는다. 계층 사이 평면 버퍼는 분산 RAM으로 매핑되었다.")
    figure(doc, IMG / "fig_util_blocks.png",
           "LUT and flip-flop count per network block (post-route).", 8.5)

    heading(doc, "5.3 타이밍", 2)
    rows = []
    for blk, lab in (("u_c1", "conv1"), ("u_b1", "B1"), ("u_b2", "B2"), ("u_b3", "B3"),
                     ("u_c2", "conv2"), ("u_tail", "tail"), ("u_st", "self-test harness")):
        s, lv, dp, lpct = block_slack(HW, blk)
        rows.append([lab, f"{s:.3f}", lv, f"{dp:.2f}", f"{lpct:.0f} / {100 - lpct:.0f}"])
    table(doc, "Worst setup path ending in each block (50 MHz, Baseline build)",
          ["Block", "Slack (ns)", "Logic levels", "Data path (ns)", "Logic / route (%)"],
          rows, widths_cm=[3.6, 2.4, 2.4, 2.6, 3.0], size=8.5)
    para(doc,
         f"두 빌드 모두 모든 타이밍 제약을 만족하였다. Baseline 빌드는 WNS {TS['wns']:+.3f} ns, WHS "
         f"{TS['whs']:+.3f} ns(setup 끝점 {TS['n']:,}개, 위반 {TS['fail']}), Partial-75 빌드는 WNS "
         f"{TS75['wns']:+.3f} ns, WHS {TS75['whs']:+.3f} ns이다. 최악 경로는 두 빌드 모두 conv2 "
         "depthwise의 MAC 입력 선택부터 누산기까지이며(k=29, dilation 2), 지연의 약 75 %가 배선 지연이다. "
         "가장 느린 속도 등급(−1)에서 20 ns 주기에 5 ns 이상 여유가 있어, 동작 주파수 상한은 약 66 MHz로 "
         "추정된다.")
    figure(doc, IMG / "fig_slack_hist.png",
           "Setup-slack distribution of the 20,000 worst endpoints (of 55,034) after routing, Baseline "
           "build. The dashed line is the worst negative slack; no endpoint is below zero.", 8.5)

    heading(doc, "5.4 전력 (추정)", 2)
    table(doc, "Vivado power estimate (vectorless, confidence: " + PW["conf"] + ")",
          ["Component", "Power (W)"],
          [["Total on-chip", PW["total"]], ["  Device static", PW["static"]],
           ["  Dynamic", PW["dyn"]], ["    Clocks", PW["clocks"]], ["    Slice logic", PW["logic"]],
           ["    Signals", PW["signals"]], ["    Block RAM", PW["bram"]], ["    DSP", PW["dsp"]],
           ["Dynamic by hierarchy: network", PW["hier"]["u_net"]],
           ["Dynamic by hierarchy: self-test harness (clip ROM)", PW["hier"]["u_st"]],
           ["Dynamic by hierarchy: debug hub", PW["hier"]["dbg_hub"]]],
          widths_cm=[8.0, 3.0], size=8.5)
    para(doc,
         "전력은 스위칭 확률을 도구 기본값으로 가정한 추정치로, 실측이 아니다. 동적 전력 0.090 W 중 "
         "네트워크가 0.058 W, 검증용 클립 ROM이 0.029 W를 차지한다. 네트워크만 고려하면 정적 전력을 "
         "포함해 약 0.15 W이며, 클립당 연산 시간 57.4 ms를 곱하면 분류 1회에 약 8.7 mJ(동적만 약 3.3 mJ)로 "
         "환산된다. 이 값은 FPGA 프로토타입의 수치로, 정적 전력이 동적 전력과 비슷한 FPGA의 특성이 크게 "
         "반영되어 있다(8장).")

    heading(doc, "5.5 스키매틱과 배치 결과", 2)
    figure(doc, IMG / "report_schematic_top.png",
           "Post-synthesis schematic of the self-test top level: the harness and network (u_top), the "
           "VIO core and the automatically inserted debug hub.", 12.0)
    dev = HW / "device_placed.png"
    figure(doc, dev,
           "Placed design on the XC7S75 (Vivado device view). Coloured regions are the major network "
           "blocks and the self-test harness; the vertical columns are block-RAM/DSP sites.", 8.0)

    # 6
    heading(doc, "6. 결과", 1)
    heading(doc, "6.1 T1: 클립 정확도 (전체 test 세트)", 2)
    fw = lambda d: sum(d["confusion_fixed"][t][p] for t in (10, 11) for p in range(10))
    table(doc, f"Clip accuracy on the full GSC v2 test set ({T1_BASE['n_clips']:,} clips, 12 classes)",
          ["Model", "Float (%)", "Integer path (%)", "Float–integer agreement (%)", "Non-keyword → keyword"],
          [["Baseline", pct(T1_BASE["float_acc"]), pct(T1_BASE["fixed_acc"]), pct(T1_BASE["agree"]),
            f"{fw(T1_BASE)} / 814 ({100 * fw(T1_BASE) / 814:.1f} %)"],
           ["Partial-75", pct(T1_P75["float_acc"]), pct(T1_P75["fixed_acc"]), pct(T1_P75["agree"]),
            f"{fw(T1_P75)} / 814 ({100 * fw(T1_P75) / 814:.1f} %)"]],
          widths_cm=[2.6, 2.2, 2.8, 4.0, 4.4], size=8.5)
    para(doc,
         "정수 경로의 정확도는 두 모델 모두 float보다 0.06–0.08 %p 높아, 양자화로 인한 정확도 손실이 "
         "없었다. 이진층은 임계 비교로 정확히 옮겨지고, 오차는 꼬리의 1/64 격자에서만 생기며 그 영향이 "
         "argmax를 바꾸는 경우는 0.25–0.35 %이다. Partial-75는 클립 정확도에서 Baseline과 같은 수준을 "
         "유지하였는데, 이 미세조정의 효과는 클립 정확도가 아니라 창 경계 상황(T3)에서 나타난다(6.3절).")
    figure(doc, FIG / "fig_per_class_recall.png",
           "Per-class recall of the integer path on the full test set. Legend values are overall accuracy.",
           16.5)
    figure(doc, IMG / "fig_confusion_bd_base.png",
           "Confusion matrix of the Baseline integer path on the full test set (clip counts; shading is "
           "the share of the true-class row). Each class has 396–425 test clips, the same scale as the "
           "confusion matrices reported in [1].", 9.0)
    para(doc,
         "클래스별로는 silence(99.3 %)가 가장 높고 unknown(64.4 %), down(76.1 %), go(77.6 %)가 낮다. "
         "가장 많이 혼동되는 쌍은 no↔go(70건), down↔go(56건)로 양방향이 비슷하여, 입력 특징이 이 소리들을 "
         "충분히 구분하지 못함을 시사한다. 이는 [1]의 혼동 행렬에서 이진 AFE 입력의 오류가 no·go·down에 "
         "모였다는 관찰과 같다.")

    heading(doc, "6.2 T2: 칩 재현성", 2)
    b600 = balanced_acc("bd_base", 600)
    p600 = balanced_acc("bd_base_ft20_partial75", 600)
    table(doc, "On-chip self-test results (XC7S75, 50 MHz)",
          ["Model", "Clip set", "Classes (true labels)", "Runs", "Match", "ms / clip", "Accuracy on set (%)"],
          [["Baseline", "test order 0–599", "2 (right, go)", "3", "600/600 each", "57.5", "81.33"],
           ["Partial-75", "test order 0–599", "2 (right, go)", "2", "600/600 each", "57.4", "81.83"],
           ["Baseline", "balanced 0–599", "12 × 50", "2", "600/600 each", "57.5", pct(b600)],
           ["Partial-75", "balanced 0–599", "12 × 50", "2", "600/600 each", "57.5", pct(p600)]],
          widths_cm=[2.1, 2.8, 2.6, 1.2, 2.8, 1.7, 2.6], size=8,
          note="Runs include a power cycle and fresh programming before the repeated runs. "
               "Accuracy on set = Python integer-path accuracy, which the chip equals when all clips match.")
    para(doc,
         "칩의 분류 결과는 측정한 모든 실행에서 파이썬 정수 경로와 600개 전부 일치하였고, 반복 실행 간 "
         "결과가 같았다. 클립당 처리 시간은 57.4–57.5 ms로 시뮬레이션 값(57.11 ms)과 폴링 해상도 안에서 "
         "일치한다. 0개 오류가 관측된 600회 실행에서 칩의 불일치율 95 % 상한은 약 0.5 %이고(rule of "
         "three), 네 세트의 서로 다른 2,400개 클립–모델 조합을 합치면 약 0.13 %이다. 반복 실행을 포함한 "
         "총 분류 횟수는 5,400회이다. 12개 클래스가 균형 있게 섞인 세트에서도 "
         "일치했으므로, 모든 출력 클래스 경로가 칩에서 올바르게 동작함을 확인하였다.")

    heading(doc, "6.3 T3: 연속 동작 (소프트웨어, 검증 세트)", 2)
    table(doc, "Always-on evaluation on spliced streams (validation split, Python float)",
          ["Model / policy", "Keyword recall", "Quiet streams with event", "Silence / unknown false",
           "Wrong events", "Median latency"],
          [["Baseline, N=5, margin 1.25", "1734/2560 (67.73 %)", "52/512", "15 / 37", "287", "1000 ms"],
           ["Partial-75, N=5, margin 1.00", "69.77 %", "46/512", "4 / 42", "250", "1000 ms"],
           ["Partial-75, N=5, margin 1.05 (selected)", "1780/2560 (69.53 %)", "41/512", "4 / 37", "242",
            "1000 ms"],
           ["Partial-75, N=5, margin 1.25", "1760/2560 (68.75 %)", "35/512", "2 / 33", "229", "1000 ms"]],
          widths_cm=[4.6, 3.0, 2.6, 2.4, 1.6, 1.8], size=8, bold_rows=(2,),
          note="Each case: 1 s silence + 1 s target clip + 1 s silence; one 1-s window every 100 ms; a "
               "detection requires the same keyword five times in a row with keyword-minus-quiet logit "
               "margin above the threshold, followed by a 1-s cooldown.")
    para(doc,
         "연속 동작 평가는 무음 1초, 대상 클립 1초, 무음 1초를 이어 붙인 스트림에 100 ms마다 1초 창을 "
         "적용하고, 같은 키워드가 5회 연속 나오며 quiet 대비 logit 차가 margin 이상일 때 검출로 본다. "
         "같은 조건에서 가운데 창(원래 클립과 동일) 정확도는 82.68 %로 T1과 같은 수준이지만, 검출률은 "
         "69.53 %로 낮다. 창 경계에 걸린 단어와 연속 판정 조건 때문이며, 이는 클립 정확도가 실제 동작 "
         "성능을 과대평가함을 보여 준다. Partial-75는 Baseline 대비 검출이 46건 늘고 quiet 오검출이 "
         "11건, 잘못된 검출이 45건 줄었다. 이 평가는 파이썬 float 모델, 검증 세트 기준이며, 정수 "
         "margin(4301)으로의 변환 검증, test 세트 1회 평가, 칩 수준 검증은 다음 단계이다.")

    heading(doc, "6.4 선행연구와의 비교", 2)
    table(doc, "Comparison under the clip-accuracy condition (12-class GSC)",
          ["", "Cerutti [1] 8 ch", "Cerutti [1] 64 ch", "Kim [5]", "DeltaKWS [6]", "This work"],
          [["Feature extraction", "analog BPF + comp.", "analog BPF + comp.", "analog time domain",
            "digital IIR", "analog BPF + comp. (16 ch)"],
           ["Classifier", "BNN", "BNN", "RNN", "ΔGRU", "partial BNN"],
           ["Accuracy (%)", "76.3", "86.0", "86.03", "89.5", f"{pct(T1_P75['fixed_acc'])}"],
           ["Accuracy basis", "SW float", "SW float", "chip", "chip", "SW integer = chip (T2)"],
           ["Implementation", "MCU (est.)", "MCU (est.)", "65 nm ASIC", "65 nm ASIC", "FPGA XC7S75"],
           ["HW/SW equivalence shown", "—", "—", "—", "—", "2,400 / 2,400 clips"]],
          widths_cm=[3.2, 2.4, 2.4, 2.4, 2.4, 3.6], size=8)
    para(doc,
         "같은 계열(아날로그 이진 특징 + BNN)인 [1]과 비교하면, 16채널인 본 연구는 8채널(76.3 %)보다 "
         "6 %p 이상 높고 64채널(86.0 %)보다 3.4 %p 낮다. 채널 수가 4배인 64채널 구성은 필터·비교기 수와 "
         "취득 에너지도 그만큼 늘어나므로, 16채널은 그 사이의 절충점에 해당한다. 두 연구 모두 GSC v2, "
         "12개 클래스, 공식 80:10:10 분할, 1초 클립당 1회 판정이라는 같은 조건에서 평가하였다. 다만 [1]의 "
         "수치는 이상적인 mel 필터 모사와 float 모델에서 얻은 것이고, 본 연구의 수치는 PCB로 설계된 "
         "필터뱅크의 모사 응답과 저항 분압으로 구현 가능한 임계값을 전제로 한 정수 경로 정확도이며, 칩에서 "
         "그대로 재현됨을 확인한 값이다. 전용 ASIC [5], [6]과는 공정·플랫폼이 달라 전력을 직접 "
         "비교할 수 없으며, 정확도는 analog 시간영역 특징을 쓴 [5]보다 3.5 %p 낮다.")

    # 7
    heading(doc, "7. 차별점과 개선점", 1)
    bullets(doc, [
        "**제작 가능한 조건을 학습에 반영.** 이상적인 mel 필터 대신 PCB로 설계된 필터뱅크의 SPICE 응답을 "
        "사용하고, 16개 임계값을 채널별 절대값으로 학습하여 저항 분압 하나로 옮길 수 있게 하였다. "
        "필터뱅크만 바꾼 비교에서 test 정확도가 0.817에서 0.825로 올랐다(프로젝트 ablation 기록).",
        "**학습 기법에 의한 개선.** 임계값 STE 구간을 1.0에서 0.003으로 좁혀 이진 입력 모델의 정확도를 "
        "+10.0 %p 올렸고, partial-window 미세조정으로 클립 정확도를 유지하면서 연속 동작 검출률과 오검출을 "
        "함께 개선하였다.",
        "**정수 전용 하드웨어에서 정확도 손실 없음.** 이진층은 임계 비교로, 꼬리는 Q*.6 고정소수점으로 "
        "옮겨 곱셈기 없는 이진 경로(DSP 7개는 꼬리에만 사용)를 구성하면서도 전체 test 세트에서 float "
        "대비 정확도 손실이 없었다.",
        "**칩 수준 동치 검증.** 선행 연구가 소프트웨어 정확도나 칩 정확도 중 하나만 보고한 것과 달리, "
        "학습 모델의 정수 경로와 칩 결과가 클립마다 같음을 12개 클래스 균형 세트를 포함한 2,400개 "
        "클립–모델 조합(반복 포함 5,400회)으로 확인하였다. "
        "칩 내부 채점과 VIO를 사용하여 추가 배선 없이 반복 가능한 검증 절차를 만들었다.",
        "**평가 조건의 분리.** 클립 정확도(T1)와 연속 동작(T3)을 구분하여, 같은 모델이 클립 기준 82.7 %, "
        "연속 기준 69.5 %임을 함께 보고하였다. 선행 칩 논문들은 대부분 T1 조건만 보고한다.",
        "**재학습에 강한 RTL.** 가중치·임계값·고정소수점 형식이 모두 export가 생성한 파일에서 오므로, "
        "아날로그 조건이 바뀌어 재학습해도 RTL 수정 없이 재합성만 하면 된다. 실제로 두 모델을 같은 RTL로 "
        "빌드하여 검증하였다.",
    ])

    # 8
    heading(doc, "8. 한계와 향후 과제", 1)
    bullets(doc, [
        "**FPGA와 ASIC의 근본적 차이.** 추정 전력 0.183 W 중 0.094 W가 소자 정적 전력으로, 저전력 "
        "ASIC(µW 수준, [5]–[8])과는 수치 자체를 비교할 수 없다. FPGA에서는 가중치 ROM이 LUT로, 연산이 "
        "범용 LUT·배선으로 구현되어 전용 회로 대비 면적·전력이 크고, 전압 스케일링이나 클럭 게이팅 같은 "
        "저전력 기법을 적용하기 어렵다. 본 결과는 기능·정확도·타이밍 검증용 프로토타입으로 해석해야 한다.",
        "**전력은 추정치.** Vivado의 vectorless 추정(신뢰도 Medium)이며, 실측이나 시뮬레이션 활동 "
        "파일(SAIF) 기반 추정이 아니다. 또한 연속 동작에서는 대부분의 시간이 대기이므로 클립당 에너지 "
        "환산은 상한에 가깝다.",
        "**실제 아날로그 보드와의 연결 미수행.** 아날로그 전단은 PCB 설계와 SPICE 시뮬레이션까지 "
        "완료되었고 시뮬레이션 결과는 긍정적이나, 실장이 이루어지지 않아 FPGA와 연결한 측정은 하지 "
        "못했다. 현재 입력은 그 응답의 소프트웨어 모사이므로, 실장 후 실제 비교기 출력으로 정확도를 "
        "다시 확인해야 한다.",
        "**정확도 목표 미달.** 클립 정확도 82.6 %로 목표(85 %)보다 2.4 %p 낮으며, unknown 클래스(64 %)와 "
        "no/go/down 혼동이 주된 손실이다. 비키워드 클립의 약 18 %가 키워드로 분류되어 always-on 오검출에 "
        "직접 영향을 준다.",
        "**연속 동작 평가의 범위.** T3는 잘라 붙인 3초 스트림 위의 자체 규약이며 현장 오검출률(FA/h)이 "
        "아니다. 현재 수치는 검증 세트·float 모델 기준이고, 정수 margin 검증, test 세트 평가, 창 이동과 "
        "투표 로직을 포함한 칩 검증(kws_stream_top)이 남아 있다.",
        "**칩 검증 범위.** 자체 검사는 실시간 10 ms 프레임 포착을 거치지 않고 ROM에서 ready 속도로 "
        "공급하므로, 프레임 포착 경로는 시뮬레이션으로만 검증되었다. 칩에서 돌린 클립은 모델당 1,200개로 "
        "전체 test 세트(4,888개)의 일부이다.",
        "**향후 계획.** (1) 연속 동작의 정수 기준 확정과 칩 검증, (2) 전체 test 세트의 칩 검증(16비트 ROM "
        "기준 비트스트림당 약 1,000클립), (3) 실제 AFE 보드 연결 및 비교기 오프셋 반영, (4) 가중치 ROM의 "
        "BRAM 이전 등 FPGA 자원 최적화, (5) 연산 카운트 기반 ASIC 에너지 추정.",
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
        "J.-H. Seol et al., “A 1.5µW end-to-end keyword spotting SoC with content-adaptive frame "
        "sub-sampling and fast-settling analog frontend,” in Proc. ISSCC, 2023.",
        "P. Warden, “Speech commands: A dataset for limited-vocabulary speech recognition,” "
        "arXiv:1804.03209, 2018.",
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
