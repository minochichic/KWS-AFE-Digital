// IT창의챌린지 과제계획서 (.html) — docx 와 같은 본문(plan_content.js)을 쓴다.
// 화면으로 읽어도, 그대로 인쇄해도 문서로 보이게 한다.
const fs = require("fs");
const { TITLE, SUBTITLE, BODY, SCHEDULE, WEEKS } = require("./plan_content");

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

let body = "";
for (const item of BODY) {
  const [kind, text] = item;
  if (kind === "h") body += `  <h2>${esc(text)}</h2>\n`;
  else if (kind === "s") body += `  <h3>${esc(text)}</h3>\n`;
  else if (kind === "p") body += `  <p>${esc(text)}</p>\n`;
  else if (kind === "b") body += `  <p class="b">${esc(text)}</p>\n`;
  else if (kind === "fig") {
    const [file, cap] = [text, item[2]];
    // SVG 를 파일 안에 박는다. 상대경로로 걸면 HTML 만 따로 보냈을 때 그림이
    // 통째로 사라진다 -- 제출물은 한 파일로 완결되어야 한다.
    const svg = fs.readFileSync(`figures/${file}.svg`, "utf8")
      .replace(/<\?xml[^>]*\?>/, "")
      .replace(/<!DOCTYPE[^>]*>/, "")
      .replace(/<svg /, '<svg role="img" aria-label="' + esc(cap) + '" ');
    body += `  <figure>${svg}<figcaption>${esc(cap)}</figcaption></figure>\n`;
  }
}

const head = `<tr><th class="task">수&nbsp;&nbsp;&nbsp;행&nbsp;&nbsp;&nbsp;내&nbsp;&nbsp;&nbsp;용</th>` +
  Array.from({ length: WEEKS }, (_, i) => `<th>${i + 1}</th>`).join("") + `</tr>`;
const rows = SCHEDULE.map(([name, from, to]) =>
  `<tr><td class="task">${esc(name)}</td>` +
  Array.from({ length: WEEKS }, (_, i) =>
    `<td>${i + 1 >= from && i + 1 <= to ? "■" : ""}</td>`).join("") + `</tr>`).join("\n      ");

const html = `<!doctype html>
<html lang="ko">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IT창의챌린지 과제계획서</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&family=IBM+Plex+Sans+KR:wght@300;400;500&display=swap">
<style>
  :root{
    --paper:#FFFFFF; --ink:#111111; --muted:#555555; --faint:#8A8A8A;
    --rule:#BBBBBB; --rule-soft:#DDDDDD; --head:#EDEDED; --back:#8E9298;
  }
  @media (prefers-color-scheme: dark){
    :root:not([data-theme="light"]){
      --paper:#12161B; --ink:#E9EDF2; --muted:#A6B0BB; --faint:#7B8590;
      --rule:#39434E; --rule-soft:#28313A; --head:#1D242C; --back:#080A0D;
    }
  }
  :root[data-theme="dark"]{
    --paper:#12161B; --ink:#E9EDF2; --muted:#A6B0BB; --faint:#7B8590;
    --rule:#39434E; --rule-soft:#28313A; --head:#1D242C; --back:#080A0D;
  }
  *{box-sizing:border-box}
  body{
    margin:0; background:var(--back); color:var(--ink);
    font-family:"Gowun Batang","바탕",Batang,serif;
    font-size:16px; line-height:1.95; -webkit-font-smoothing:antialiased;
  }
  .page{
    background:var(--paper); max-width:820px; margin:26px auto;
    padding:74px 78px 86px; box-shadow:0 2px 30px rgba(0,0,0,.28);
  }

  /* ── 표지 ── */
  .cover{min-height:78vh; display:flex; flex-direction:column}
  .who{text-align:right; margin-top:22px; letter-spacing:.02em}
  .who div{margin-bottom:9px}
  .cover .mid{flex:1; display:flex; flex-direction:column; justify-content:center;
              text-align:center; padding-bottom:8vh}
  .cover h1{font-size:33px; font-weight:700; letter-spacing:.04em; margin:0 0 62px}
  .subj{font-size:19px; margin:0 0 8px}
  .subj b{font-weight:700}
  .sub2{font-size:15px; color:var(--muted); margin:0 0 46px}
  .team{font-size:16px; margin:0}
  .to{text-align:center; font-size:20px; font-weight:700; letter-spacing:.06em}
  /* 본문 p 규칙(들여쓰기·양쪽정렬)이 상속된 가운데 정렬을 덮어쓰므로 되돌린다 */
  .cover p{text-indent:0; text-align:center}

  /* ── 본문 ── */
  .banner{font-size:16.5px; font-weight:700; margin:0 0 26px}
  h2{font-size:18.5px; font-weight:700; margin:30px 0 12px; letter-spacing:.01em}
  h2:first-of-type{margin-top:6px}
  h3{font-size:16px; font-weight:700; margin:20px 0 8px}
  p{margin:0 0 11px; text-indent:1em; text-align:justify; word-break:keep-all}
  p.b{text-indent:0; padding-left:1.2em; position:relative; margin-bottom:6px}
  p.b::before{content:"·"; position:absolute; left:.35em}

  figure{margin:20px 0 22px; text-align:center}
  figure svg{max-width:100%; height:auto; display:block; margin:0 auto}
  /* 도판은 흑백 선화라 어두운 배경에서 사라진다 -- 반전해 되살린다 */
  @media (prefers-color-scheme: dark){
    :root:not([data-theme="light"]) figure svg{filter:invert(1) hue-rotate(180deg)}
  }
  :root[data-theme="dark"] figure svg{filter:invert(1) hue-rotate(180deg)}
  figcaption{
    margin-top:7px; font-family:"IBM Plex Sans KR",sans-serif;
    font-size:13px; font-weight:300; color:var(--muted);
  }

  table{width:100%; border-collapse:collapse; margin-top:6px;
        font-family:"IBM Plex Sans KR",sans-serif; font-size:13.5px; font-weight:300}
  th,td{border:1px solid var(--rule); padding:7px 5px; text-align:center; line-height:1.5}
  th{background:var(--head); font-weight:500}
  .task{text-align:left; padding-left:12px; width:46%}
  .foot{margin-top:11px; font-family:"IBM Plex Sans KR",sans-serif;
        font-size:12.5px; font-weight:300; color:var(--muted); text-indent:0}

  @media print{
    body{background:#fff}
    .page{box-shadow:none; margin:0; max-width:none; padding:0}
    .cover{min-height:auto; height:88vh; page-break-after:always}
    h2,h3{page-break-after:avoid}
    figure,tr{page-break-inside:avoid}
    @page{size:letter; margin:22mm 24mm}
  }
  @media (max-width:700px){ .page{padding:40px 26px 56px; margin:0} }
</style>

<div class="page cover">
  <div class="who"><div>참여학생대표 :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</div>
    <div>지도교수 :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</div></div>
  <div class="mid">
    <h1>IT창의챌린지 과제계획서</h1>
    <p class="subj">과 제 명 : <b>${esc(TITLE)}</b></p>
    <p class="sub2">${esc(SUBTITLE)}</p>
    <p class="team">(&nbsp; 과제팀명 :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; )</p>
  </div>
  <div class="to">대한전자공학회 귀하</div>
</div>

<div class="page">
  <p class="banner">□ 과제계획 및 작품 개요 (분량 자유, 사진 및 그림 등으로 표현 권장)</p>
${body}
  <table>
    <thead>${head}</thead>
    <tbody>
      ${rows}
    </tbody>
  </table>
  <p class="foot">※ 단위: 주. 실제 일정은 대회 일정에 맞추어 조정한다.</p>
</div>
`;

fs.writeFileSync("IT창의챌린지_과제계획서.html", html);
console.log(`생성: IT창의챌린지_과제계획서.html  (${html.length} bytes)`);
