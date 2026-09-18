"""Two edits on the merged (analog + digital) contest report.

1. The sliding window and the consecutive-vote decision get their own
   subsection. In the merged draft they were two sentences inside "2.2 디지털
   분류 가속기", between the network tables -- the part that makes this an
   always-on design rather than a clip classifier deserves its own heading,
   including its geometry, its cost and how far it is verified.
2. Section 3.5 states the accuracy metric (T1: one decision per 1-s clip)
   before the table, so the number cannot be read as a streaming figure.

Formatting is copied from the paragraph each insertion lands next to, so the
result keeps the document's own fonts. The input file is not modified: the
result is written next to it with a _v2 suffix.

    out/.venv_report/Scripts/python docs/report/patch_merged_report.py
"""
from __future__ import annotations

import copy
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

SRC = Path("2026_반도체설계경진대회_통합_설계보고서_양식정리.docx")
DST = SRC.with_name(SRC.stem + "_v2.docx")

# 2.2 keeps frame capture, the network and the tables; the window and the vote move out
BODY_22 = (
    "아날로그 front end와의 interface는 comparator 출력 16핀뿐이다. Frame capture는 10 ms 동안 한 번이라도 "
    "1이 된 채널을 1로 기록해 16비트 프레임을 만들고, network는 layer마다 MAC engine 하나를 "
    "time-multiplexing으로 재사용하는 folded 구조로 1초 창 하나를 57 ms 안에 12개 class score로 바꾼다. "
    "Sliding window와 연속 판정은 2.3절에서 따로 설명한다."
)

HEAD_23 = "2.3 Sliding window와 연속 판정"
BODY_23 = [
    "단어가 언제 시작되는지 알 수 없는 always-on 동작에서는 1초 창을 한 번 분류하는 것으로 부족하다. "
    "본 설계는 comparator 출력을 끊김 없이 포착하면서 창을 100 ms마다 옮기고, 연속 판정으로 검출을 "
    "선언한다. 창 이동과 판정은 모두 hardware에 있으며 host나 software의 개입이 없다.",

    "Sliding window는 최근 100프레임(1초)을 circular buffer에 유지하고, 10프레임(100 ms)마다 그 내용을 "
    "별도 snapshot buffer로 복사한 뒤 좌우에 14프레임의 padding을 붙여 128프레임 입력을 만든다. 복사를 "
    "분리한 이유는 replay 중에도 capture가 멈추지 않아야 하기 때문이다. 저장량은 history와 snapshot을 "
    "합쳐 3,200비트이며 distributed RAM으로 구현되어 BRAM을 쓰지 않는다. Snapshot 복사는 100 clock으로 "
    "끝나므로 다음 프레임 포착 전에 완료된다. 창 주기(100 ms)가 network 1회 추론(57 ms)보다 길어 요청이 "
    "밀리지 않으며, 만약 밀리는 경우에는 해당 요청을 건너뛰고 overrun counter를 올려 조용히 누락되지 "
    "않게 한다.",

    "판정은 창마다 얻은 최상위 keyword score와 최상위 quiet(silence/unknown) score의 차이를 문턱과 "
    "비교한다. 문턱은 validation에서 고른 float margin 1.05를 pooled 정수 단위로 옮긴 4,301이며, RTL에는 "
    "export가 생성한 상수로 들어간다. 같은 keyword가 문턱을 넘겨 5회 연속 나오면 검출을 선언하고, 이후 "
    "1초(창 10개)의 cooldown과 quiet 결과 한 번을 모두 본 뒤에 다시 무장한다. 한 발화가 여러 번 세어지는 "
    "것을 막기 위한 조건이다.",

    "검증 범위는 다음과 같다. Frame capture, sliding window, 투표 module은 각각 XSim module-level "
    "simulation에서 0 failure로 통과하였다(각 3프레임, 33프레임과 overrun·busy interlock, 3개 시나리오). "
    "세 module과 network를 합친 always-on 경로는 정수 reference와 창 단위로 비교하는 self-test를 "
    "simulation에서 통과하였다. 연속 동작의 인식 성능 수치는 3.5절과 같이 software simulation 결과이다.",
]

METRIC = (
    "인식 성능은 평가 조건을 두 가지로 구분하여 보고한다. 첫째는 clip 단위 정확도로, 단어가 창 안에 "
    "온전히 들어 있는 1초 clip을 한 번 분류하여 argmax를 정답 label과 비교한다. 대부분의 KWS 연구가 "
    "보고하는 조건이며, 본 설계는 Google Speech Commands v2의 공식 test 분할 전체 4,888개 clip(12 class, "
    "class별 396-425개)을 사용하였다. 둘째는 연속 동작 조건으로, 단어의 시작 시점을 모르는 stream에서 "
    "창을 100 ms마다 옮기며 5회 연속 판정으로 검출하는 비율이다. 같은 model이라도 창 경계에 걸린 단어와 "
    "연속 조건 때문에 두 수치가 다르므로 섞어 비교하지 않는다. Table VII은 clip 단위 정확도이다."
)


def find(doc, needle: str) -> int:
    for i, p in enumerate(doc.paragraphs):
        if needle in p.text:
            return i
    raise SystemExit(f"paragraph not found: {needle!r}")


def clone_after(model, text: str, *, bold=None, size=None, align=None,
                space_after=None):
    """Insert a copy of `model`'s paragraph after it, carrying its formatting."""
    new = copy.deepcopy(model._element)
    model._element.addnext(new)
    para = docx.text.paragraph.Paragraph(new, model._parent)
    for r in para.runs[1:]:
        r._element.getparent().remove(r._element)
    if not para.runs:
        para.add_run("")
    run = para.runs[0]
    run.text = text
    if bold is not None:
        run.bold = bold
    if size is not None:
        run.font.size = Pt(size)
    if align is not None:
        para.alignment = align
    if space_after is not None:
        para.paragraph_format.space_after = Pt(space_after)
    return para


def main() -> None:
    doc = docx.Document(str(SRC))

    # ---- 1. sliding window as its own subsection ----
    i22 = find(doc, "아날로그 front end와의 interface는")
    body = doc.paragraphs[i22]
    for r in body.runs[1:]:
        r._element.getparent().remove(r._element)
    body.runs[0].text = BODY_22

    # the new section goes after the last paragraph of section 2 (the Fig. 9 caption)
    anchor = doc.paragraphs[find(doc, "On-chip self-test 구조")]
    head_model = doc.paragraphs[find(doc, "2.2 디지털 분류 가속기")]
    head_run = head_model.runs[0]
    head = clone_after(anchor, HEAD_23, bold=True,
                       size=head_run.font.size.pt if head_run.font.size else None,
                       align=WD_ALIGN_PARAGRAPH.LEFT, space_after=3)
    prev = head
    for text in BODY_23:
        prev = clone_after(prev, text, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
        prev.paragraph_format.space_after = Pt(4)

    # ---- 2. the accuracy metric, before Table VII ----
    tbl = doc.paragraphs[find(doc, "TABLE VII")]
    sec35 = doc.paragraphs[find(doc, "3.5 인식 성능")]
    metric = clone_after(sec35, METRIC, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    metric.paragraph_format.space_after = Pt(4)
    assert tbl.text.strip().startswith("TABLE VII")

    doc.save(str(DST))
    print(f"wrote {DST}  (paragraphs {len(doc.paragraphs)}, tables {len(doc.tables)}, "
          f"images {len(doc.inline_shapes)})")


if __name__ == "__main__":
    main()
