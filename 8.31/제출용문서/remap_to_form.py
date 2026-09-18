"""동료 docx 내용을 그대로 두고, 서식_1(과제계획서)의 구분 체계로만 재배치한다.

  서식_1 구분:  □ 신청자 개요 (1 과제 대표학생 / 2 참여학생 / 3 지도교수)
               □ 과제계획 및 작품 개요
                 1 과제 개요 · 2 개발 내용 · 3 개발 방법 · 4 작품의 재료 · 5 과제추진계획 및 일정

본문·표·그림은 원본 XML을 그대로 옮기고, 제목의 번호와 수준만 바꾼다.
"""
import re, shutil, zipfile
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "IT창의챌린지_과제계획서_디지털수정본 (1).docx"
DST = HERE / "IT창의챌린지_과제계획서_서식1.docx"

H1, H2, H3 = "1", "21", "31"          # heading 1 / 2 / 3 style id
FONT = '<w:rFonts w:ascii="맑은 고딕" w:eastAsia="맑은 고딕" w:hAnsi="맑은 고딕"/>'


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── 본문 최상위 요소 분리 (모든 태그를 세어야 중첩 표가 안 깨진다) ──────
def split_body(body):
    els, depth, start = [], 0, None
    for m in re.finditer(r"<(/?)[A-Za-z0-9:._-]+(?:\s[^<>]*?)?(/?)>", body):
        close, self_ = m.group(1) == "/", m.group(2) == "/"
        if close:
            depth -= 1
            if depth == 0:
                els.append(body[start:m.end()])
        elif self_:
            if depth == 0:
                els.append(m.group(0))
        else:
            if depth == 0:
                start = m.start()
            depth += 1
    return els


# ── 제목 글자와 수준만 갈아끼우기 (나머지 XML은 보존) ──────────────────
def retitle(el, text, style):
    if "<w:pStyle" in el:
        el = re.sub(r'<w:pStyle w:val="[^"]*"/>',
                    f'<w:pStyle w:val="{style}"/>', el, count=1)
    n = [0]

    def rep(m):
        n[0] += 1
        return ('<w:t xml:space="preserve">' + esc(text) + "</w:t>") if n[0] == 1 \
            else (m.group(1) + "</w:t>")

    return re.sub(r"(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)", rep, el)


# ── 새 문단 / 표 ────────────────────────────────────────────────────
def P(text, style=None, bold=False, sz=20, align=None):
    ppr = "<w:pPr>"
    if style:
        ppr += f'<w:pStyle w:val="{style}"/>'
    if align:
        ppr += f'<w:jc w:val="{align}"/>'
    ppr += "</w:pPr>"
    rpr = f"<w:rPr>{FONT}{'<w:b/>' if bold else ''}<w:sz w:val='{sz}'/>" \
          f"<w:szCs w:val='{sz}'/></w:rPr>".replace("'", '"')
    return (f"<w:p>{ppr}<w:r>{rpr}"
            f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


def PAGEBREAK():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def cell(text, w, head=False):
    shd = '<w:shd w:val="clear" w:color="auto" w:fill="EDEDED"/>' if head else ""
    rpr = f"<w:rPr>{FONT}{'<w:b/>' if head else ''}<w:sz w:val=\"18\"/></w:rPr>"
    jc = '<w:jc w:val="center"/>' if head else ""
    return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shd}'
            f'<w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p><w:pPr>{jc}<w:spacing w:after="0" w:line="288" '
            f'w:lineRule="auto"/></w:pPr><w:r>{rpr}'
            f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p></w:tc>')


def table(rows, widths, head_row=False, min_h=380):
    b = ('<w:tblBorders>' + "".join(
        f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        for s in ("top", "left", "bottom", "right", "insideH", "insideV")) +
        "</w:tblBorders>")
    out = (f'<w:tbl><w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/>{b}'
           '<w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid>' +
           "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>")
    for i, r in enumerate(rows):
        head = head_row and i == 0
        out += (f'<w:tr><w:trPr><w:trHeight w:val="{min_h}"/>'
                + ('<w:tblHeader/>' if head else "") + "</w:trPr>"
                + "".join(cell(c, widths[j], head) for j, c in enumerate(r))
                + "</w:tr>")
    return out + "</w:tbl>"


# ═══════════════════════════════════════════════════════════════════
z = zipfile.ZipFile(SRC)
doc = z.read("word/document.xml").decode()
head, body, tail = (doc[:doc.index("<w:body>") + 8],
                    doc[doc.index("<w:body>") + 8: doc.rindex("</w:body>")],
                    doc[doc.rindex("</w:body>"):])
E = split_body(body)

# ET 로 얻은 진짜 자식 목록과 대조 — 인덱스가 하나라도 어긋나면 중단한다
import xml.etree.ElementTree as ET
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ref = list(ET.fromstring(doc).find(W + "body"))
assert len(E) == len(ref) == 136, f"요소 수 불일치: {len(E)} vs {len(ref)}"
for i, (raw, node) in enumerate(zip(E, ref)):
    assert raw.lstrip().startswith("<" + node.tag.replace(W, "w:")), f"{i}번 태그 불일치"
    a = "".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", raw))
    b = "".join(n.text or "" for n in node.iter(W + "t"))
    assert a == b, f"{i}번 본문 불일치"
sect = E[135]

# 원 제목 → 새 번호 (제목 글자는 그대로, 번호와 수준만 변경)
RE = {
    23: ("1.1 대회 제한사항 대응 및 시스템 목표", H2),
    121: ("1.2 기대 성과 및 차별성", H2),
    26: ("2.1 전체 회로 구성", H2),
    31: ("2.2 아날로그 전단 설계", H2),
    32: ("2.2.1 MM20 마이크, PreAMP 및 전원 구조", H3),
    35: ("2.2.2 GIC 대역통과 필터뱅크", H3),
    46: ("2.2.3 능동 Envelope Detector", H3),
    52: ("2.2.4 LPV7215 Comparator와 채널별 임계값", H3),
    56: ("2.2.5 아날로그 특징의 이진 시공간 패턴 변환", H3),
    60: ("2.3 MCU·FPGA 없는 디지털 판별부 설계", H2),
    61: ("2.3.1 기본 구조", H3),
    71: ("2.3.2 START 조건: 발화의 시간 원점", H3),
    74: ("2.3.3 시각 생성", H3),
    76: ("2.3.4 Template Match", H3),
    80: ("2.3.5 PASS 저장, 최종 WAKE 및 timeout", H3),
    85: ("2.3.6 NN기반 오프라인 파라미터 학습", H3),
    95: ("3.1 아날로그 회로 시뮬레이션", H2),
    97: ("3.2 음성 데이터와 8채널 선정", H2),
    99: ("3.3 디지털 판별부 학습", H2),
    101: ("3.4 연속 오디오 기반 오검출 평가", H2),
    103: ("3.5 성능 지표", H2),
    106: ("3.6 제작 및 실물 검증 순서", H2),
    127: ("3.7 현재 확정 사항과 추후 결정 항목", H2),
    131: ("참고문헌", H1),
}


def take(a, b):
    return [retitle(E[i], *RE[i]) if i in RE else E[i] for i in range(a, b)]


W4 = [1700, 3456, 1700, 3456]                       # 10312 twip = 본문 폭
W6 = [900, 1600, 2400, 2000, 1000, 2412]

new = []
new += E[0:19]                                       # 표지 (원본 그대로)

new += [P("□ 신청자 개요", H1)]
new += [P("1. 과제 대표학생", H2),
        table([["대학명", "", "성  명", ""], ["휴대폰", "", "E-Mail", ""]], W4)]
new += [P("2. 참여학생", H2),
        table([["연번", "성명", "대학명", "학과", "학년", "연락처"],
               ["1", "", "", "", "", ""], ["2", "", "", "", "", ""],
               ["3", "", "", "", "", ""], ["4", "", "", "", "", ""]],
              W6, head_row=True),
        P("※ 1번은 과제 대표학생을 포함하여 기재한다.", sz=16)]
new += [P("3. 지도교수", H2),
        table([["대학명", "", "성  명", ""], ["학과명", "", "E-Mail", ""]], W4)]

new += [PAGEBREAK(),
        P("□ 과제계획 및 작품 개요 (분량 자유, 사진 및 그림 등으로 표현 권장)", H1)]

new += [P("1. 과제 개요", H1)]
new += take(20, 23)          # 원 1절 본문
new += take(23, 26)          # 원 2절  → 1.1
new += take(121, 127)        # 원 10절 → 1.2

new += [P("2. 개발 내용", H1)]
new += take(26, 31)          # 원 3절 → 2.1
new += take(31, 60)          # 원 4절 → 2.2
new += take(60, 94)          # 원 5절 → 2.3

new += [P("3. 개발 방법", H1)]
new += take(95, 106)         # 원 6절 → 3.1~3.5 (6절 제목은 구분이 대신함)
new += take(106, 115)        # 원 7절 → 3.6
new += take(127, 131)        # 원 11절 → 3.7

new += [P("4. 작품의 재료", H1)]
new += take(116, 118)        # 원 8절 표

new += [P("5. 과제추진계획 및 일정", H1)]
new += take(119, 121)        # 원 9절 표

new += take(131, 135)        # 참고문헌
new += [sect]

out_doc = head + "".join(new) + tail

shutil.copy(SRC, DST)
zin = zipfile.ZipFile(SRC)
with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zo:
    for it in zin.infolist():
        zo.writestr(it, out_doc.encode("utf-8") if it.filename == "word/document.xml"
                    else zin.read(it.filename))

print(f"{DST.name}  {DST.stat().st_size // 1024} KB")
print(f"요소 {len(E)} → {len(new)},  제목 재번호 {len(RE)}개")
