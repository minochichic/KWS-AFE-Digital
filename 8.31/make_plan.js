// IT창의챌린지 과제계획서 (.docx) — 예제 PDF 양식에 맞춤
const fs = require("fs");
const d = require("docx");
const { TITLE, SUBTITLE, BODY, SCHEDULE, WEEKS } = require("./plan_content");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, ShadingType, VerticalAlign, PageBreak,
  ImageRun,
} = d;

const FONT = "바탕";              // 예제와 같은 명조 계열
const FONT_S = "맑은 고딕";
const CONTENT_W = 12240 - 1700 * 2;

// IHDR 청크에서 픽셀 크기를 읽는다. 비율을 지켜야 도판이 찌그러지지 않는다.
function pngSize(buf) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

const t = (text, o = {}) => new TextRun({
  text, font: o.font || FONT, size: o.size || 21, bold: o.bold, color: o.color,
});

const p = (text, o = {}) => new Paragraph({
  alignment: o.align,
  spacing: { after: o.after === undefined ? 130 : o.after, before: o.before || 0, line: 340 },
  indent: o.indent === undefined ? { firstLine: 200 } : o.indent,
  children: Array.isArray(text) ? text : [t(text, o)],
});

const doc_children = [
  // ── 표지 ──
  p("", { after: 200, indent: {} }),
  p([t("참여학생대표 :                              ")],
    { align: AlignmentType.RIGHT, after: 140, indent: {} }),
  p([t("지도교수 :                              ")],
    { align: AlignmentType.RIGHT, after: 1800, indent: {} }),
  p([t("IT창의챌린지 과제계획서", { bold: true, size: 40 })],
    { align: AlignmentType.CENTER, after: 1600, indent: {} }),
  p([t("과 제 명 : ", { size: 24 }), t(TITLE, { bold: true, size: 24 })],
    { align: AlignmentType.CENTER, after: 200, indent: {} }),
  p([t(SUBTITLE, { size: 20 })],
    { align: AlignmentType.CENTER, after: 1000, indent: {} }),
  p([t("(  과제팀명 :                                )", { size: 21 })],
    { align: AlignmentType.CENTER, after: 2600, indent: {} }),
  p([t("대한전자공학회 귀하", { bold: true, size: 26 })],
    { align: AlignmentType.CENTER, indent: {} }),
  new Paragraph({ children: [new PageBreak()] }),

  // ── 본문 머리 ──
  p([t("□ 과제계획 및 작품 개요 (분량 자유, 사진 및 그림 등으로 표현 권장)",
       { bold: true, size: 22 })], { after: 260, indent: {} }),
];

for (const item of BODY) {
  const [kind, text] = item;
  if (kind === "h") {
    doc_children.push(new Paragraph({
      spacing: { before: 300, after: 150 }, indent: {},
      children: [t(" " + text, { bold: true, size: 23 })],
    }));
  } else if (kind === "s") {
    doc_children.push(new Paragraph({
      spacing: { before: 190, after: 90 }, indent: { left: 100 },
      children: [t(text, { bold: true, size: 21 })],
    }));
  } else if (kind === "p") {
    doc_children.push(p(text));
  } else if (kind === "b") {
    doc_children.push(new Paragraph({
      spacing: { after: 70, line: 330 }, indent: { left: 440, hanging: 200 },
      children: [t("· " + text)],
    }));
  } else if (kind === "fig") {
    const png = fs.readFileSync(`figures/${text}.png`);
    const dim = pngSize(png);
    const wPt = 405;                                  // 본문 폭에 맞춘 표시 폭
    doc_children.push(new Paragraph({
      spacing: { before: 200, after: 50 }, alignment: AlignmentType.CENTER,
      indent: {},
      children: [new ImageRun({
        data: png, type: "png",
        transformation: { width: wPt, height: Math.round(wPt * dim.h / dim.w) },
      })],
    }));
    doc_children.push(new Paragraph({
      spacing: { after: 210 }, alignment: AlignmentType.CENTER, indent: {},
      children: [t(item[2], { font: FONT_S, size: 18, color: "444444" })],
    }));
  }
}

// ── 일정표 ──
const W_TASK = CONTENT_W - 560 * WEEKS;
const hcell = (txt, w) => new TableCell({
  width: { size: w, type: WidthType.DXA },
  shading: { type: ShadingType.CLEAR, fill: "EDEDED" },
  verticalAlign: VerticalAlign.CENTER,
  margins: { top: 70, bottom: 70, left: 80, right: 80 },
  children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
    children: [t(txt, { bold: true, size: 19 })] })],
});

doc_children.push(new Table({
  columnWidths: [W_TASK, ...Array(WEEKS).fill(560)],
  width: { size: CONTENT_W, type: WidthType.DXA },
  rows: [
    new TableRow({
      tableHeader: true,
      children: [hcell("수   행   내   용", W_TASK),
                 ...Array.from({ length: WEEKS }, (_, i) => hcell(String(i + 1), 560))],
    }),
    ...SCHEDULE.map(([name, from, to]) => new TableRow({
      children: [
        new TableCell({
          width: { size: W_TASK, type: WidthType.DXA },
          verticalAlign: VerticalAlign.CENTER,
          margins: { top: 62, bottom: 62, left: 110, right: 80 },
          children: [new Paragraph({ spacing: { after: 0 },
            children: [t(name, { size: 19 })] })],
        }),
        ...Array.from({ length: WEEKS }, (_, i) => {
          const on = i + 1 >= from && i + 1 <= to;
          return new TableCell({
            width: { size: 560, type: WidthType.DXA },
            verticalAlign: VerticalAlign.CENTER,
            margins: { top: 62, bottom: 62, left: 30, right: 30 },
            children: [new Paragraph({ alignment: AlignmentType.CENTER,
              spacing: { after: 0 },
              children: [t(on ? "■" : "", { size: 17, color: "444444" })] })],
          });
        }),
      ],
    })),
  ],
}));

doc_children.push(new Paragraph({
  spacing: { before: 150 }, indent: {},
  children: [t("※ 단위: 주. 실제 일정은 대회 일정에 맞추어 조정한다.",
               { font: FONT_S, size: 18, color: "666666" })],
}));

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 21 } } } },
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 },
              margin: { top: 1500, bottom: 1500, left: 1700, right: 1700 } },
    },
    children: doc_children,
  }],
});

Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync("IT창의챌린지_과제계획서.docx", b);
  const chars = BODY.filter(([k]) => k === "p" || k === "b")
                    .reduce((n, [, x]) => n + x.length, 0);
  console.log(`생성: IT창의챌린지_과제계획서.docx  (${b.length} bytes, 본문 ${chars}자)`);
});
