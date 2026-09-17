"""2026 한국 대학생 반도체 설계 경진대회 -- 참가신청서 + 설계보고서 (.docx).

Layout follows 2026-반도체설계경진대회_참가신청서_0426.pdf:
  p1  참가신청서 (작품명, 연구 분야, 참가자 표, 서명)
  붙임 1 설계보고서 (≤ 12 pages, 맑은고딕 10 pt, guidance text removed)
    1. 설계 요약 (≤ 2 pages): summary box ≤ 10 lines, 1) 창의성 2) 난이도 3) 완성도
    2. 구성 및 동작
    3. 설계 과정 및 실험 결과 (+ comparison with prior work, references)

Formatting helpers and every number come from build_report.py / the run files.

    out/.venv_report/Scripts/python docs/report/build_contest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R  # noqa: E402

from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt  # noqa: E402

# the contest asks for 맑은고딕 throughout
R.LATIN = "Malgun Gothic"
R.HANGUL = "Malgun Gothic"


def _full_grid(table):
    """Every cell boxed (instead of the three-rule paper style) -- easier to read here."""
    tblpr = table._element.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), "000000")
        b.append(e)
    tblpr.append(b)
    for cell in table.rows[0].cells:     # light header shading
        R.shade(cell, "E7E6E6")


R.borders = _full_grid
OUT = R.ROOT / "out/report_build/2026_반도체설계경진대회_참가신청서_설계보고서_v5.docx"

TITLE = ("An Always-On Keyword Spotting System Combining an Analog Binary Feature-Extraction "
         "Front End and a Partially Binarized MatchboxNet Accelerator on FPGA")
FIELD = "Analog + Digital / AI 반도체 (아날로그 특징 추출 전단 + 이진 신경망 가속기)"


def run(p, text, size=10, bold=False, underline=False, italic=False):
    r = p.add_run(text)
    R.set_fonts(r, size, bold, italic)
    r.font.underline = underline
    return r


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def cell_borders(cell, sz="8"):
    tcpr = cell._element.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), sz)
        e.set(qn("w:color"), "000000")
        b.append(e)
    tcpr.append(b)


def grid_table(doc, header, rows, widths_cm, size=9):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, r_ in enumerate([header] + rows):
        for j, v in enumerate(r_):
            c = t.rows[i].cells[j]
            c.text = ""
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            run(p, v, size, bold=(i == 0))
            c.width = Cm(widths_cm[j])
            cell_borders(c, "6")
    return t


# ---- page 1: application form ------------------------------------------------ #
def application(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run(p, "<2026 한국 대학생 반도체 설계 경진대회 참가신청서>", 15, True, underline=True)
    p.paragraph_format.space_after = Pt(24)

    p = doc.add_paragraph()
    run(p, "1. 연구 작품명: ", 11, True)
    p = doc.add_paragraph()
    run(p, TITLE, 11, underline=True)
    p.paragraph_format.space_after = Pt(18)

    p = doc.add_paragraph()
    run(p, "연구 분야: ", 10, True)
    run(p, FIELD, 10, underline=True)
    p.paragraph_format.space_after = Pt(24)

    p = doc.add_paragraph()
    run(p, "2. 참가자 정보", 11, True)
    grid_table(doc, ["소속", "이름", "이메일", "전화번호", "역할"],
               [["", "", "", "", "제1저자"], ["", "", "", "", "제2저자"], ["", "", "", "", "제3저자"],
                ["", "", "", "", "제4저자"], ["", "", "", "", "지도교수"]],
               [4.2, 2.6, 3.8, 3.2, 2.4], 9)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run(p, "*지도교수 포함 최대 5인 이하", 9, True)
    p.paragraph_format.space_after = Pt(30)

    p = doc.add_paragraph()
    run(p, "상기와 같이 2026 한국 대학생 반도체 설계 경진대회 참가 신청서와 설계보고서를 제출합니다.", 11, True)
    p.paragraph_format.space_after = Pt(14)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run(p, "붙임 1. 설계보고서 1부.", 10, True)
    p.paragraph_format.space_after = Pt(18)
    for text in ("2026년      월      일", "신청인(제1저자)                    (인)",
                 "지도교수                    (인)"):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run(p, text, 11, True)
        p.paragraph_format.space_after = Pt(16)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(60)
    run(p, "반 도 체 공 학 회   회장 ", 20, True)
    run(p, "귀하", 14, True)


# ---- design report ------------------------------------------------------------ #
def summary_box(doc, lines):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.rows[0].cells[0]
    c.width = Cm(16.6)
    cell_borders(c, "8")
    c.text = ""
    for i, ln in enumerate(lines):
        p = c.paragraphs[0] if i == 0 else c.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.1
        run(p, "• " + ln, 9.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def side_by_side(doc, images, caption):
    """Two pictures in one borderless row, one shared caption -- (a) left, (b) right."""
    t = doc.add_table(rows=2, cols=len(images))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, (path, width) in enumerate(images):
        c = t.rows[0].cells[j]
        c.width = Cm(width + 0.3)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Cm(width))
        lab = t.rows[1].cells[j].paragraphs[0]
        lab.alignment = WD_ALIGN_PARAGRAPH.CENTER
        lab.paragraph_format.space_after = Pt(0)
        run(lab, f"({'ab'[j]})", 9)
    R.FIG_N[0] += 1
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run(c, f"Fig. {R.FIG_N[0]}. ", 9, True)
    run(c, caption, 9)
    c.paragraph_format.space_after = Pt(8)


def h(doc, text, size=11):
    p = doc.add_paragraph()
    run(p, text, size, True)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    return p


def report(doc):
    T1B, T1P = R.T1_BASE, R.T1_P75
    TS = R.timing_summary(R.HW / "timing_summary.rpt")
    TS75 = R.timing_summary(R.HW75 / "timing_summary.rpt")
    PW = R.power(R.HW)
    PARAMS = dict(conv1=22784, b1=23616, b2=10624, b3=10880, conv2=10432, conv3=16640, conv4=1548)
    BITS = dict(conv1=8, b1=1, b2=1, b3=1, conv2=1, conv3=8, conv4=7)
    n_par = sum(PARAMS.values())
    mem_bits = sum(PARAMS[k] * BITS[k] for k in PARAMS)
    fp_bits = n_par * 32
    bin_par = PARAMS["b1"] + PARAMS["b2"] + PARAMS["b3"] + PARAMS["conv2"]
    fw = lambda d: sum(d["confusion_fixed"][t][p] for t in (10, 11) for p in range(10))

    p = doc.add_paragraph()
    run(p, "(붙임 1) 설계보고서", 10)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run(p, "2026 한국 대학생 반도체 설계 경진대회 설계보고서", 15, True, underline=True)
    p.paragraph_format.space_after = Pt(10)
    p = doc.add_paragraph()
    run(p, "설계 작품명: ", 10.5, True, underline=True)
    run(p, TITLE, 10.5, underline=True)
    p.paragraph_format.space_after = Pt(8)

    # ---- 1. 설계 요약 ----
    h(doc, "1. 설계 요약", 12)
    summary_box(doc, [
        "기능: 아날로그 필터뱅크·포락선 검출기·비교기로 ADC·FFT 없이 16채널 이진 특징을 추출하고, 이를 12개 클래스"
        "(키워드 10개 + silence + unknown)로 분류하는 always-on 키워드 스포팅(KWS) 시스템",
        "아날로그 전단: 16채널 특징 추출 회로 설계, PCB 설계 및 SPICE 시뮬레이션 완료",
        "구조: 부분 이진화 MatchboxNet(1D TCS, 96.5 k 파라미터) + 곱셈기 없는 정수 전용 folded RTL + 100 ms "
        "sliding window·5연속 투표 판정",
        f"정확도: Google Speech Commands v2 12-class test {T1B['n_clips']:,}개, 정수 경로 {R.pct(T1P['fixed_acc'])} % "
        "(float 대비 손실 없음)",
        f"구현: Xilinx Spartan-7 XC7S75, 50 MHz, WNS {TS['wns']:+.3f} ns, LUT 43 %, 추론 57.5 ms/클립(판정 주기 100 ms)",
        f"하드웨어 효율: 이진층 곱셈기 0개(DSP 7개는 후단만), 가중치 메모리 float32 대비 약 {fp_bits / mem_bits:.1f}배 축소",
        "검증: 칩 내부 자체 검사로 2,400개 클립–모델 조합(반복 포함 5,400회)에서 소프트웨어와 100 % 일치",
        "응용: 배터리 기반 IoT·웨어러블·스마트홈 기기의 음성 웨이크업, 초저전력 음성 인터페이스(ADC·FFT 제거)",
    ])

    h(doc, "1) 창의성", 11)
    R.bullets(doc, [
        "**아날로그 이진 특징 + 부분 이진화 MatchboxNet.** 기존 연구 [1]의 2D CNN BNN 대신 1D time-channel "
        "separable 구조를 채택하고, 첫 층(INT8)·마지막 층(고정소수점)을 제외한 중간 블록만 이진화하여 곱셈을 "
        "XNOR-popcount로, BN을 정수 비교 하나로 대체하였다.",
        "**제작 가능한 조건을 학습에 반영.** PCB로 설계된 필터뱅크의 SPICE 응답과, 저항 분압 하나로 구현되는 "
        "채널별 절대 임계값을 네트워크와 함께 학습하였다. 좁은 STE 구간으로 이진 입력 모델 정확도를 +10.0 %p 개선하였다.",
        "**트리거 방식에서 연속 판정으로.** [1]의 '첫 이벤트 후 1초 1회 분류'를 100 ms 간격 sliding window와 "
        "연속 투표로 바꾸고, 창 경계에 걸린 단어를 학습하는 partial-window 미세조정을 도입하였다.",
        "**칩 내부 자체 검사.** 클립 ROM·기대값 ROM·채점기를 칩에 두고 JTAG VIO로 결과만 읽는 구조로, 핀 2개만으로 "
        "수백 클립을 분 단위로 검증한다.",
    ], 10)
    h(doc, "2) 난이도", 11)
    R.bullets(doc, [
        "학습(PyTorch) → 정수 변환(export) → RTL → FPGA 구현 → 칩 검증까지 전 과정을 직접 설계하였으며, 소프트웨어 "
        "정수 경로와 **비트 단위로 일치**해야 하는 허용 오차 0의 목표를 설정하였다.",
        "BN 융합, 잔차 정수 합산, Q*.6 고정소수점 꼬리를 연쇄 양자화하여 **정확도 손실 없는** 정수 전용 경로를 만들었다"
        f"(정수 {R.pct(T1B['fixed_acc'])} % vs float {R.pct(T1B['float_acc'])} %).",
        "이진 입력·16채널이라는 정보 제약 아래에서 표준 12-class 전체 test 세트로 평가하였고, 창 경계·연속 판정 조건의 "
        "always-on 성능까지 별도로 측정하였다.",
    ], 10)
    h(doc, "3) 완성도", 11)
    R.bullets(doc, [
        "모듈 단위(프레임 포착, 창, 투표, 평면 버퍼, TCS 블록, 꼬리)부터 전체 네트워크까지 골든 벡터 시뮬레이션에서 "
        "0 failures, 두 모델 빌드 모두 타이밍 위반 0으로 구현을 완료하였다.",
        "칩에서 2개 클래스 편중 세트와 12개 클래스 균형 세트를 각각 두 모델로 실행하여 모든 실행에서 600/600 일치, "
        "전원 재투입 후 반복 실행도 동일함을 확인하였다.",
        "가중치를 ROM 파일로 분리하여 모델을 바꿔도 RTL 수정이 없으며, 학습·export·빌드·검증·보고서 생성 전 과정이 "
        "스크립트로 재현 가능하다.",
    ], 10)

    # ---- 2. 구성 및 동작 ----
    page_break(doc)
    h(doc, "2. 구성 및 동작", 12)
    R.figure(doc, R.IMG / "fig_system.png",
             "전체 구성. 아날로그 전단이 비교기 출력 16가닥을 내고, FPGA가 10 ms 프레임 포착 → 1초 sliding window → "
             "folded BinaryMatchboxNet → 연속 투표 판정을 수행한다.", 16.5)
    h(doc, "2.1 아날로그 특징 추출 전단", 10.5)
    R.para(doc, "[아날로그 전단 구성 및 동작 — 공동 연구자 작성 예정]", italic=True)
    h(doc, "2.2 디지털 분류 가속기", 10.5)
    R.para(doc,
           "아날로그 전단과의 인터페이스는 비교기 출력 16가닥뿐이다. "
           "디지털 단의 동작은 다음과 같다. (1) 프레임 포착: 10 ms 동안 한 번이라도 1이 된 채널을 1로 기록해 16비트 "
           "프레임을 만든다. (2) sliding window: 최근 100프레임을 원형 버퍼에 유지하고 100 ms마다 스냅숏을 떠서 좌우 "
           "14프레임을 채운 128프레임 입력을 만든다. (3) 네트워크: 계층마다 MAC 엔진 하나를 시분할로 재사용하는 folded "
           "구조로 57 ms 안에 12개 클래스 점수를 계산한다. (4) 판정: 같은 키워드가 quiet 대비 margin 이상으로 5회 연속 "
           "나오면 검출하고 1초간 재검출을 막는다. (1), (2), (4)의 연속 판정 블록은 RTL로 설계하여 XSim 모듈 시뮬레이션"
           "(프레임 포착 3프레임, sliding window 33프레임·overrun, 투표 3개 시나리오, 모두 0 failures)으로 검증하였고, "
           "칩 수준 검증은 (3) 네트워크를 대상으로 수행하였다.")
    R.table(doc, "Network architecture (C = 64, T = 128, 12 classes)",
            ["Stage", "Type", "Precision", "Kernel", "Out ch.", "Params", "Weight bits"],
            [["conv1", "1D conv, stride 2", "INT8 (binary input)", "11", "128", "22,784", "8"],
             ["B1", "TCS ×2 + projection skip", "binary", "13", "64", "23,616", "1"],
             ["B2", "TCS ×2 + identity skip", "binary", "15", "64", "10,624", "1"],
             ["B3", "TCS ×2 + identity skip", "binary", "17", "64", "10,880", "1"],
             ["conv2", "separable, dilation 2", "binary", "29", "128", "10,432", "1"],
             ["conv3", "1×1 conv", "INT8", "1", "128", "16,640", "8"],
             ["conv4", "1×1 conv", "fixed point", "1", "12", "1,548", "7"],
             ["Total", "", "", "", "", f"{n_par:,}", f"{mem_bits / 8 / 1000:.1f} kB"]],
            widths_cm=[1.6, 4.0, 3.2, 1.3, 1.4, 2.0, 2.2], size=9, bold_rows=(7,))
    R.para(doc,
           "이진층의 곱셈은 XNOR-popcount로, BN과 부호 함수는 정수 비교 하나로 바뀌며, 입력이 ±1이므로 conv1도 부호 있는 "
           "누산만으로 계산된다. 곱셈은 BN이 흡수되지 않는 후단에만 남는다(Table II).")
    R.table(doc, "Hardware consequences of the partial binarization",
            ["Property", "Floating-point MatchboxNet", "This design", "Effect"],
            [["Multiply in B1–B3, conv2", "real multiply", "XNOR + popcount", "no multiplier; DSP only in tail (7/140)"],
             ["Multiply in conv1", "real multiply", "±1 input → signed add", "no multiplier"],
             ["BatchNorm + activation", "scale, shift, nonlinearity", "one integer compare", "no BN arithmetic"],
             ["Weight memory", f"{fp_bits / 8 / 1000:.0f} kB", f"{mem_bits / 8 / 1000:.1f} kB",
              f"≈{fp_bits / mem_bits:.1f}× smaller (binary {100 * bin_par / n_par:.0f} % of params)"],
             ["Inter-layer activations", "32 bit / value", "1 bit / value", "4 planes = 669 LUT"],
             ["Quantization accuracy", "—", "integer path",
              f"{R.pct(T1B['fixed_acc'])} % vs float {R.pct(T1B['float_acc'])} %"]],
            widths_cm=[3.4, 3.4, 3.8, 5.9], size=8.5,
            note="Weight memory counts weights only; integer thresholds and tail gain/offset ROMs excluded.")
    R.figure(doc, R.IMG / "fig_selftest.png",
             "칩 자체 검사 구조. 클립과 기대값을 BRAM에 저장하고 칩 안에서 모든 분류를 채점하며, 카운터만 JTAG VIO로 읽는다.",
             8.5)

    # ---- 3. 설계 과정 및 실험 결과 ----
    page_break(doc)
    h(doc, "3. 설계 과정 및 실험 결과", 12)
    h(doc, "3.1 아날로그 전단 설계 및 시뮬레이션", 11)
    R.para(doc, "[회로·PCB 설계 및 SPICE 시뮬레이션 결과 — 공동 연구자 작성 예정]", italic=True)
    h(doc, "3.2 디지털 단 설계 과정", 11)
    R.para(doc,
           "(1) 학습: 아날로그 전단의 SPICE 필터 응답을 적용한 16채널 이진 입력으로 QAT 학습하고, 비교기 임계값 16개를 "
           "STE로 함께 학습하였다(Table III). (2) 정수 변환: BN을 정수 임계값으로 접고 후단을 Q*.6 고정소수점으로 바꿔 "
           "ROM 파일과 층별 골든 벡터를 생성하였다. (3) RTL 검증: 모듈별로 골든 벡터와 비트 단위 비교 후 합성·배치배선하였다. "
           "(4) 칩 검증: 자체 검사 비트스트림으로 클립 세트를 칩에서 실행하였다.")
    R.table(doc, "Training configuration and result",
            ["Item", "Baseline", "Partial-75 (fine-tune)"],
            [["Data", "GSC v2, official split, 12 classes, 16-ch binary input", "same"],
             ["Schedule", "Adam 1e-3, plateau ×0.1, 100 epochs", "Adam 1e-4, 20 epochs from Baseline"],
             ["Binarization", "QAT, hardtanh STE; threshold STE clip 0.003", "same"],
             ["Augmentation", "none", "partial keyword window (p = 0.5, 75–100 %)"],
             ["Best val. accuracy", "83.64 %", "83.32 %"]],
            widths_cm=[3.2, 7.4, 6.0], size=9)

    h(doc, "3.3 FPGA 구현 결과", 11)
    tot = {k: R.util_row(R.HW / "util_summary.rpt", k) for k in
           ("Slice LUTs", "Slice Registers", "Block RAM Tile", "DSPs", "Bonded IOB")}
    avail = {"Slice LUTs": "48,000", "Slice Registers": "96,000", "Block RAM Tile": "90", "DSPs": "140",
             "Bonded IOB": "338"}
    R.table(doc, "Post-route summary (XC7S75-1, Vivado 2026.1, 50 MHz)",
            ["Item", "Baseline build", "Partial-75 build", "Available"],
            [[k, f"{v[0]:,} ({v[1]} %)", f"{R.util_row(R.HW75 / 'util_summary.rpt', k)[0]:,}", avail[k]]
             for k, v in tot.items()]
            + [["WNS / WHS (ns)", f"{TS['wns']:+.3f} / {TS['whs']:+.3f}",
                f"{TS75['wns']:+.3f} / {TS75['whs']:+.3f}", "—"],
               ["Failing endpoints", f"{TS['fail']} of {TS['n']:,}", f"{TS75['fail']} of {TS75['n']:,}", "—"],
               ["Inference time", "57.5 ms / clip", "57.5 ms / clip", "hop 100 ms"]],
            widths_cm=[3.6, 4.2, 3.6, 2.6], size=9,
            note="BRAM is the self-test clip ROM (48 RAMB36); the network uses 6 RAMB18. Identical RTL, different ROMs.")
    rows = []
    for inst, lab in [("u_c1", "conv1"), ("u_b1", "B1"), ("u_b2", "B2"), ("u_b3", "B3"), ("u_c2", "conv2"),
                      ("u_tail", "tail (conv2 pw, conv3, conv4)")]:
        u = R.block_util(R.HW / "util_hier.rpt", inst)
        s, lv, dp, lpct = R.block_slack(R.HW, inst)
        rows.append([lab, f"{u[0]:,}", f"{u[3]:,}", u[6], f"{s:.3f}", lv])
    R.table(doc, "Per-block resources and worst setup path (Baseline build)",
            ["Block", "LUT", "FF", "DSP", "Worst slack (ns)", "Logic levels"],
            rows, widths_cm=[4.6, 1.8, 1.8, 1.2, 2.8, 2.2], size=9)
    R.para(doc,
           "최악 경로는 conv2 depthwise(k=29)의 MAC 입력 선택부터 누산기까지이며, 가장 느린 속도 등급에서 5 ns 이상 "
           "여유가 있다. 전력은 Vivado 추정으로 총 0.183 W(정적 0.094 W, 네트워크 동적 0.058 W)이다.")
    side_by_side(doc,
                 [(R.HW / "device_place.png", 4.6), (R.IMG / "schematic_net_wrapper.png", 11.4)],
                 "(a) XC7S75 배치 결과(Vivado device view). 색칠된 영역은 네트워크 블록, 자체 검사 하네스, 디버그 로직이다. "
                 "(b) 합성 후 스키매틱: 자체 검사 최상위(kws_selftest_top) 안의 네트워크(kws_top)와 입출력 인터페이스.")

    h(doc, "3.4 칩 수준 검증 결과", 11)
    b600 = R.balanced_acc("bd_base", 600)
    p600 = R.balanced_acc("bd_base_ft20_partial75", 600)
    R.table(doc, "On-chip self-test results (XC7S75, 50 MHz)",
            ["Model", "Clip set", "True classes", "Runs", "Chip = Python", "Accuracy on set (%)"],
            [["Baseline", "test order 0–599", "2", "3", "600/600 each run", "81.33"],
             ["Partial-75", "test order 0–599", "2", "2", "600/600 each run", "81.83"],
             ["Baseline", "balanced 0–599", "12 × 50", "2", "600/600 each run", R.pct(b600)],
             ["Partial-75", "balanced 0–599", "12 × 50", "2", "600/600 each run", R.pct(p600)]],
            widths_cm=[2.2, 3.0, 2.2, 1.3, 3.4, 3.0], size=9,
            note="Repeated runs include a power cycle and fresh programming.")
    R.para(doc,
           "모든 실행에서 칩과 소프트웨어 정수 경로가 600개 전부 일치하였다(불일치율 95 % 상한 약 0.13 %).")

    h(doc, "3.5 인식 성능", 11)
    R.table(doc, f"Clip accuracy on the full GSC v2 test set ({T1B['n_clips']:,} clips, 12 classes)",
            ["Model", "Float (%)", "Integer path (%)", "Agreement (%)", "Non-keyword → keyword"],
            [["Baseline", R.pct(T1B["float_acc"]), R.pct(T1B["fixed_acc"]), R.pct(T1B["agree"]), f"{fw(T1B)} / 814"],
             ["Partial-75", R.pct(T1P["float_acc"]), R.pct(T1P["fixed_acc"]), R.pct(T1P["agree"]), f"{fw(T1P)} / 814"]],
            widths_cm=[2.6, 2.2, 2.8, 2.6, 3.6], size=9)
    R.figure(doc, R.IMG / "fig_confusion_bd_base.png",
             "Baseline 정수 경로 혼동 행렬(클립 수, n = 4,888). 클래스당 396–425개로 [1]의 혼동 행렬과 같은 규모.", 8.5)
    R.para(doc,
           "주요 혼동은 no↔go, down↔go로 [1]과 같은 경향이다. 연속 동작 인식 성능은 소프트웨어 시뮬레이션으로 평가하였다(검증 세트, 무음–단어–무음 "
           "3초 스트림, 100 ms hop, 5연속 판정). 그 결과 Partial-75는 키워드 검출률 69.53 %(Baseline 67.73 %), quiet 오검출 "
           "41/512(Baseline 52/512)를 보였다.")

    h(doc, "3.6 기존 기술과의 비교", 11)
    R.table(doc, "Comparison with prior work on 12-class Google Speech Commands (clip accuracy)",
            ["Work", "Feature extraction", "Classifier", "Acc. (%)", "Platform / basis", "HW = SW shown"],
            [["Cerutti [1], 64 ch", "analog BPF + comparator", "BNN (2D CNN)", "86.0", "SW float + MCU est.", "—"],
             ["Cerutti [1], 8 ch", "analog BPF + comparator", "BNN (2D CNN)", "76.3", "SW float + MCU est.", "—"],
             ["Rybakov [3]", "MFCC (digital)", "DS-CNN (FP)", "97.0", "SW (TFLite)", "—"],
             ["Vocell [4]", "MFCC (digital)", "NN", "90.87", "65 nm chip, 16 µW", "—"],
             ["Kim [5]", "analog time domain", "RNN", "86.03", "65 nm chip, 23 µW", "—"],
             ["DeltaKWS [6]", "digital IIR", "ΔGRU", "89.5", "65 nm chip, 5.22 µW", "—"],
             ["This work", "analog BPF + comparator (16 ch)", "partial BNN (1D TCS)", R.pct(T1P["fixed_acc"]),
              "FPGA RTL (integer)", "2,400 / 2,400"]],
            widths_cm=[2.7, 3.9, 3.0, 1.5, 3.4, 2.2], size=8.5, bold_rows=(6,),
            note="[5] as tabulated in [6]; [4] as quoted in [1]. ASIC power is not comparable with an FPGA prototype.")
    R.table(doc, "Advances over the analog-binary-feature + BNN approach of [1]",
            ["Aspect", "Cerutti et al. [1]", "This work"],
            [["Classifier", "2D CNN (3×3), BNN", "1D TCS MatchboxNet, partial binarization"],
             ["Filter bank in training", "ideal mel filters", "PCB-designed filter bank (SPICE response)"],
             ["Thresholds", "learned on normalized envelopes", "absolute per channel → one resistor divider"],
             ["Classifier execution", "GAP8 MCU, energy estimated", "dedicated integer-only RTL on FPGA"],
             ["HW/SW agreement", "not reported", "2,400 / 2,400 on chip"],
             ["Acquisition", "trigger, one 1-s window", "sliding window (100-ms hop) + 5-in-a-row vote"],
             ["Window-boundary robustness", "—", "partial-window fine-tuning (+46 detections, simulation)"]],
            widths_cm=[3.6, 5.4, 7.6], size=9)
    R.para(doc,
           "16채널 정확도(82.6 %)는 [1]의 8채널(76.3 %)과 64채널(86.0 %) 사이이며, 제작 가능한 필터뱅크·임계값을 전제로 한 "
           "정수 경로 결과가 칩에서 그대로 재현된다. 다만 FPGA는 정적 전력이 커서 µW급 ASIC과 전력을 직접 비교할 수는 없다.")

    h(doc, "참고문헌", 11)
    refs = [
        "G. Cerutti et al., “Sub-mW keyword spotting on an MCU: Analog binary feature extraction and binary "
        "neural networks,” arXiv:2201.03386, 2022.",
        "S. Majumdar and B. Ginsburg, “MatchboxNet: 1D time-channel separable convolutional neural network "
        "architecture for speech commands recognition,” in Proc. Interspeech, 2020.",
        "O. Rybakov et al., “Streaming keyword spotting on mobile devices,” in Proc. Interspeech, 2020.",
        "J. S. P. Giraldo et al., “Vocell: A 65-nm speech-triggered wake-up SoC for 10-µW keyword spotting and "
        "speaker verification,” IEEE JSSC, vol. 55, no. 4, pp. 868–878, 2020.",
        "K. Kim et al., “A 23-µW keyword spotting IC with ring-oscillator-based time-domain feature extraction,” "
        "IEEE JSSC, vol. 57, no. 11, pp. 3298–3311, 2022.",
        "Q. Chen et al., “DeltaKWS: A 65nm 36nJ/decision bio-inspired temporal-sparsity-aware digital keyword "
        "spotting IC with 0.6V near-threshold SRAM,” IEEE TCASAI, 2025.",
        "P. Warden, “Speech commands: A dataset for limited-vocabulary speech recognition,” arXiv:1804.03209, 2018.",
    ]
    for i, r_ in enumerate(refs, 1):
        p = doc.add_paragraph()
        run(p, f"[{i}] {r_}", 8.5)
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.first_line_indent = Cm(-0.6)
        p.paragraph_format.space_after = Pt(1)


def main():
    doc = Document()
    R.style_doc(doc)
    doc.styles["Normal"].font.size = Pt(10)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.15
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(1.8)
    # page number footer "- n -"
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run(fp, "- ", 9)
    r = fp.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), kind)
        else:
            e = OxmlElement("w:instrText")
            e.set(qn("xml:space"), "preserve")
            e.text = text
        r._element.append(e)
    run(fp, " -", 9)

    application(doc)
    page_break(doc)
    report(doc)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT, "figures", R.FIG_N[0], "tables", R.TAB_N[0])


if __name__ == "__main__":
    main()
