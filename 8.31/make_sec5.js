// 통합 계획서 5절 — 우리 설계 명세로 새로 작성한 판별부(digital stage) 절.
// 용어·블록명·신호명은 전부 우리 설계 문서(설계명세.html, 개념설명.html) 기준.
const fs = require("fs");
const d = require("docx");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, ShadingType, VerticalAlign,
} = d;

const FONT = "맑은 고딕";
const W = 12240 - 1700 * 2;          // 본문 폭 8840 twip = 442 pt

const t = (x, o = {}) => new TextRun({ text: x, font: FONT, size: o.size || 20,
  bold: o.bold, italics: o.italics, color: o.color });

const p = (x, o = {}) => new Paragraph({
  spacing: { after: o.after === undefined ? 120 : o.after, before: o.before || 0, line: 320 },
  alignment: o.align,
  children: Array.isArray(x) ? x : [t(x, o)],
});

const h2 = (x) => new Paragraph({ spacing: { before: 340, after: 150 },
  children: [t(x, { bold: true, size: 26 })] });
const h3 = (x) => new Paragraph({ spacing: { before: 260, after: 110 },
  children: [t(x, { bold: true, size: 22 })] });
const bullet = (x) => new Paragraph({ spacing: { after: 60, line: 310 },
  indent: { left: 400, hanging: 190 }, children: [t("· " + x)] });

// ── PNG 삽입 (가로폭 고정, 세로는 원본 비율)
function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }

function figure(name, cap, wPt = 430) {
  const png = fs.readFileSync(`${__dirname}/figures/${name}.png`);
  const dim = pngSize(png);
  return [
    new Paragraph({
      spacing: { before: 200, after: 60 }, alignment: AlignmentType.CENTER,
      children: [new ImageRun({
        data: png, type: "png",
        transformation: { width: wPt, height: Math.round(wPt * dim.h / dim.w) },
      })],
    }),
    new Paragraph({
      spacing: { after: 220 }, alignment: AlignmentType.CENTER,
      children: [t(cap, { size: 18, color: "555555" })],
    }),
  ];
}

function table(headers, rows, widths) {
  const cell = (txt, o = {}) => new TableCell({
    width: { size: o.w, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    shading: o.head ? { type: ShadingType.CLEAR, fill: "EDEDED" } : undefined,
    margins: { top: 70, bottom: 70, left: 110, right: 90 },
    children: [new Paragraph({
      alignment: o.head ? AlignmentType.CENTER : AlignmentType.LEFT,
      spacing: { after: 0, line: 290 },
      children: [t(txt, { bold: o.head, size: 18 })],
    })],
  });
  return new Table({
    columnWidths: widths,
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    rows: [
      new TableRow({ tableHeader: true,
        children: headers.map((x, i) => cell(x, { head: true, w: widths[i] })) }),
      ...rows.map((r) => new TableRow({
        children: r.map((c, i) => cell(c, { w: widths[i] })) })),
    ],
  });
}

const gap = (n = 100) => new Paragraph({ spacing: { after: n }, children: [t("")] });

// ═══════════════════════════════════════════════════════════ 본문
const K = [
h2("5. 판별부(Digital Stage) 설계"),

p("아날로그 전단은 마이크 신호를 N개의 대역으로 나누고, 각 대역의 포락선을 문턱과 " +
  "비교하여 채널마다 0/1 한 가닥을 내보낸다. 판별부는 그 N가닥을 받아 " +
  "\"목표 키워드가 방금 발성되었는가\"를 WAKE 한 가닥으로 답하는 부분이며, 본 절의 " +
  "설계 대상이다. 판별부에는 MCU도 FPGA도 두지 않는다. 로직 게이트·카운터·디코더·" +
  "플립플롭·저항·비교기만으로 구성하고, 학습으로 얻는 것은 소자값과 배선뿐이다."),

p("이하의 논리는 채널 수 N에 의존하지 않는다. N은 아날로그 전단이 확정될 때 " +
  "함께 정해지며, 판별부에서 바뀌는 것은 형판 저항망의 입력 가닥 수뿐이다."),

// ── 5.1
h3("5.1 설계 원칙"),
bullet("연산이 아니라 배선으로 푼다. 곱셈·누산을 수행하는 소자를 두지 않고, " +
  "\"어느 채널을 볼 것인가\"와 \"몇 개 맞으면 통과인가\"를 저항 연결과 기준전압으로 표현한다."),
bullet("판정을 시각별로 쪼갠다. 1초 구간 전체를 한 번에 비교하지 않고, 발성 시작 시점을 " +
  "원점으로 하는 소수의 시각(state) s에서만 스냅숏을 찍어 비교한다. 시각의 개수는 4를 " +
  "기준으로 하며, 이 값이 회로 규모를 결정한다."),
bullet("파라미터 수를 소자 수와 같게 둔다. 학습이 만들어내는 자유도는 " +
  "형판 M(s,c), 허용 오차 k(s), 시각 τ(s), timeout 다섯 종류뿐이고, 전부 저항·디코더 " +
  "결선·기준전압으로 1:1 대응된다."),
bullet("모든 상태를 비동기 CLR 한 가닥으로 되돌린다. 오판정 후 회복 경로가 하나뿐이어야 " +
  "디버깅과 검증이 가능하다."),

// ── 5.2
h3("5.2 전체 신호 배선"),
p("판별부의 전체 구성은 그림 9와 같다. 채널 버스는 두 갈래로만 갈라진다. " +
  "하나는 발성 시작을 잡는 START detect로, 다른 하나는 형판 비교를 수행하는 " +
  "Template match로 들어간다. 그 사이의 모든 블록은 \"언제 볼 것인가\"를 만드는 " +
  "타이밍 경로이며, 채널 값 자체를 건드리지 않는다."),
...figure("sec5_wiring", "그림 9. 판별부 전체 신호 배선", 434),

p("블록별 역할은 다음과 같다."),
table(
  ["블록", "역할", "구현 소자"],
  [
    ["START detect", "채널 버스가 무음에서 유성으로 바뀌는 순간을 한 번 잡는다. 이 시점이 시간 원점 t=0이다.", "OR/AND 게이트 조합 또는 저항 합산 + 비교기"],
    ["RUN latch", "START를 받아 계측 구간을 연다. RUN=1인 동안에만 카운터가 돈다.", "D 플립플롭 1개"],
    ["100 Hz osc", "10 ms 시간 격자를 만든다. 학습에서 쓰는 프레임 간격과 같은 값이다.", "슈미트 트리거 인버터 + R·C"],
    ["Timing counter", "RUN 구간에서 10 ms를 세어 경과 시각을 이진수로 유지한다.", "8 bit 이진 카운터"],
    ["Time decode", "카운터 값이 학습으로 정해진 τ(s) 및 timeout과 같아지는 순간 해당 선을 올린다.", "3-to-8 디코더 2개 + NOR"],
    ["SAMPLE gate", "디코드 출력을 클럭 반주기만큼 늦춰 카운터 천이 구간을 피한다.", "AND 게이트"],
    ["Template match", "현재 채널 값과 형판을 비교해 상태별 MATCH를 낸다. 레벨 신호이다.", "저항망 + 비교기 (상태 수만큼)"],
    ["PASS F/F", "SAMPLE 시점의 MATCH를 붙잡아 둔다. 이후 채널이 변해도 결과가 유지된다.", "D 플립플롭 (상태 수만큼)"],
    ["Final AND", "모든 상태가 통과했을 때만 WAKE를 올린다.", "2입력 AND 게이트 3개"],
  ],
  [1900, 4300, 2640],
),
gap(),

// ── 5.3
h3("5.3 레벨과 펄스"),
p("판별부에는 성격이 다른 두 종류의 신호가 있고, 이 구분이 회로 전체의 골격이다."),
bullet("레벨(level) — MATCH는 \"지금 이 순간 채널 패턴이 형판과 맞는가\"를 나타내며 " +
  "언제나 값을 가진다. 발성 중에는 수시로 0과 1을 오간다."),
bullet("펄스(pulse) — SAMPLE은 \"지금이 볼 시각인가\"를 나타내며 τ(s)에서 한 번만 뜬다."),
p("두 신호는 PASS 플립플롭에서 만난다. MATCH는 D 입력으로, SAMPLE은 CLK 입력으로 " +
  "들어간다. 즉 \"항상 답을 내고 있는 신호\"를 \"정해진 순간에만 채점\"하는 구조이며, " +
  "이 때문에 형판 비교 회로는 조합논리만으로 충분하고 별도의 메모리가 필요 없다."),

// ── 5.4
h3("5.4 시간 원점과 계측 구간"),
p("형판은 절대 시각이 아니라 발성 시작으로부터의 상대 시각에 정의된다. 같은 단어라도 " +
  "말하기 시작하는 시점은 매번 다르므로, 원점을 먼저 잡아야 형판이 의미를 갖는다."),
p("START detect가 원점을 잡고 RUN latch가 계측 구간을 연다. RUN=1이 되면 " +
  "Timing counter의 리셋이 풀리고 10 ms마다 1씩 증가한다. 계측 구간은 두 가지 방법으로만 " +
  "닫힌다. WAKE가 나오거나, timeout 시각에 도달하는 것이다."),
p("timeout은 학습이 정하는 값이며, 마지막 상태 τ(4)보다 크게 잡는다. 그 시각에 " +
  "Time decode가 DECODE_TO를 올리면 RUN latch와 PASS 플립플롭이 동시에 비동기 CLR되어 " +
  "회로가 IDLE로 돌아간다. timeout을 짧게 잡을수록 한 번의 오검출이 다음 발성을 " +
  "가리는 시간이 줄어들지만, τ(4) 직후로 지나치게 붙이면 WAKE 출력 펄스가 짧아진다. " +
  "WAKE의 폭은 (timeout − τ(4)) × 10 ms이다."),

// ── 5.5
h3("5.5 시각 생성과 헛펄스 제거"),
p("Timing counter는 경과 시간을 세기만 하고, 어느 값이 의미 있는지는 모른다. " +
  "의미 부여는 Time decode가 한다. 학습이 τ(s)=17을 냈다면 카운터 출력 8비트 중 " +
  "하위 3비트와 중간 3비트를 각각 3-to-8 디코더에 넣고 두 출력을 NOR로 합쳐 " +
  "\"카운터 == 17\"인 순간에만 1이 되는 선을 만든다. 디코더 출력이 액티브 로우이므로 " +
  "AND가 아니라 NOR를 쓴다."),
p("카운터의 비트들은 완전히 동시에 바뀌지 않는다. 예컨대 15에서 16으로 넘어가는 " +
  "순간 비트 전파 지연 때문에 중간에 다른 값이 수 나노초 스쳐 지나가고, 디코더는 이를 " +
  "그대로 짧은 펄스로 내보낸다. 이 헛펄스가 PASS 플립플롭을 잘못 클럭하면 판정이 " +
  "무너진다."),
p("SAMPLE gate가 이를 막는다. 디코드 출력을 그대로 쓰지 않고 100 Hz 클럭의 반전과 " +
  "AND한다. 즉 SAMPLE = DECODE · /CLK이다. 카운터는 클럭 상승 에지에서 값을 바꾸므로 " +
  "천이는 클럭 전반부에 몰려 있고, /CLK가 1이 되는 후반부에는 카운터 출력이 이미 " +
  "안정되어 있다. 결과적으로 채점은 항상 10 ms 구간의 중앙 부근에서 일어난다."),

// ── 5.6
h3("5.6 형판 비교"),
p("각 상태 s는 채널마다 세 값 중 하나를 갖는 형판 M(s,c)를 가진다. " +
  "1은 \"이 채널이 켜져 있어야 한다\", 0은 \"꺼져 있어야 한다\", X는 " +
  "\"보지 않는다\"이다. X인 채널은 저항망에 아예 연결하지 않으므로, X가 많을수록 " +
  "부품 수가 줄고 화자 변동에 둔감해진다."),
p("비교는 다수결로 한다. 형판이 요구하는 값과 실제 채널 값이 일치하는 개수를 세고, " +
  "그 수가 허용 오차 k(s) 이상이면 MATCH(s)=1로 둔다. 회로에서는 일치 여부 신호들을 " +
  "같은 크기의 저항으로 묶어 평균 전압을 만들고, 그 전압을 비교기에서 기준전압과 " +
  "비교한다. 상태 s에서 실제로 보는 채널이 m(s)개일 때"),
p([t("        V_score = VDD × (일치 개수) / m(s),     "),
   t("V_TH(s) = VDD × (k(s) − 0.5) / m(s)", { bold: true })]),
p("기준전압은 상태마다 다르다. X 때문에 분모 m(s)가 상태마다 달라지기 때문이며, " +
  "하나의 전압을 공유할 수 없다. 각 상태의 비교기 기준전압은 자체 분압 저항 두 개로 만든다."),
p("채널별 가중치는 두지 않는다. 가중치를 넣으려면 채널마다 다른 저항값이 필요하고 " +
  "저항 오차가 곧 판정 오차가 된다. 저항을 전부 같은 값으로 두는 대신, 중요도의 표현은 " +
  "\"본다 / 보지 않는다\"의 이진 선택(X)으로만 허용한다."),

// ── 5.7
h3("5.7 판정 출력"),
p("SAMPLE(s)의 상승 에지에서 PASS 플립플롭이 그 순간의 MATCH(s)를 붙잡는다. " +
  "이후 채널 값이 어떻게 변해도 PASS(s)는 유지되므로, 서로 다른 시각에 일어난 네 번의 " +
  "판정을 같은 시점에 모아 볼 수 있다. 네 PASS를 Final AND로 묶은 것이 WAKE이다. " +
  "즉 WAKE는 \"네 시각 모두에서 형판이 맞았다\"는 뜻이며, 어느 하나라도 어긋나면 " +
  "출력되지 않는다."),
gap(60),

table(
  ["신호", "성격", "의미"],
  [
    ["CH0…CHn", "레벨", "아날로그 전단의 비교기 출력. 판별부의 유일한 입력."],
    ["START", "펄스", "발성 시작 검출. 시간 원점."],
    ["RUN", "레벨", "계측 구간 열림. 카운터 동작 허가."],
    ["CLK", "펄스", "100 Hz, 10 ms 격자."],
    ["DECODE S1…S4", "펄스", "카운터 값이 τ(s)와 일치하는 순간."],
    ["DECODE_TO", "펄스", "카운터 값이 timeout과 일치하는 순간."],
    ["SAMPLE", "펄스", "DECODE · /CLK. 실제 채점 시각."],
    ["MATCH(s)", "레벨", "상태 s의 형판 일치 여부. 상시 유효."],
    ["PASS(s)", "레벨", "SAMPLE 시점에 확정된 MATCH(s)."],
    ["CLR", "레벨", "비동기 클리어. RUN과 PASS를 동시에 IDLE로 되돌린다."],
    ["WAKE", "레벨", "최종 출력. PASS 전부의 AND."],
  ],
  [1750, 1000, 6090],
),
gap(),

// ── 5.8
h3("5.8 오프라인 학습 모델"),
p("위 회로의 소자값은 손으로 정하지 않는다. 그림 10의 모델을 Google Speech Commands " +
  "데이터셋으로 학습시켜 얻는다. 이 모델은 회로와 같은 순서로 계산하도록 만들었으므로, " +
  "학습이 끝나면 각 블록의 파라미터가 그대로 회로 상수가 된다."),
...figure("sec5_model", "그림 10. 오프라인 학습 모델", 386),

p("그림 왼쪽이 전체 흐름이고, 오른쪽 점선 안은 그중 Template Head 한 상태를 펼쳐 놓은 " +
  "것이다. 블록별로 하는 일은 다음과 같다."),
table(
  ["블록", "하는 일", "회로에서는"],
  [
    ["Binary feature",
     "아날로그 전단이 내보낼 0/1 이미지. 세로가 채널, 가로가 10 ms 단위의 시간이다. 여기까지 오는 경로는 아날로그 사양에 맞춰 고정해 두고 학습하지 않는다.",
     "비교기 출력 N가닥"],
    ["START detect + Align",
     "말이 시작된 지점을 찾아 시간의 원점을 맞춘다. 사람마다 말하기 시작하는 때가 다르므로, 이걸 맞춰야 뒤의 비교가 의미를 갖는다.",
     "START detect, RUN latch"],
    ["Template Head",
     "모델의 본체. 정해진 네 시각에서 이미지를 한 줄씩 떼어내, 미리 학습해 둔 무늬와 얼마나 닮았는지 본다.",
     "저항망 + 비교기"],
    ["AND over states",
     "네 시각이 모두 닮았을 때만 통과시킨다. 하나라도 어긋나면 탈락이다.",
     "Final AND"],
    ["Binary Cross Entropy",
     "목표 단어에는 통과, 다른 소리에는 탈락이 나오도록 점수를 매겨 학습을 이끈다.",
     "학습에만 쓰이고 회로에는 남지 않음"],
  ],
  [2050, 4400, 2390],
),
gap(80),
p("Template Head 안쪽은 다시 다섯 단계로 나뉜다."),
table(
  ["단계", "하는 일", "회로에서는"],
  [
    ["Gather at τ(s)",
     "정해진 그 시각의 세로줄 하나만 뽑는다. 나머지 시간은 보지 않는다.",
     "Time decode, SAMPLE gate"],
    ["Ternary template",
     "채널마다 '켜져 있어야 함 / 꺼져 있어야 함 / 상관없음' 셋 중 하나를 학습한다. 이것이 형판이다.",
     "저항의 연결 여부와 극성"],
    ["Match count",
     "형판이 요구한 대로 되어 있는 채널이 몇 개인지 센다.",
     "저항으로 만든 평균 전압"],
    ["Tolerance k(s)",
     "몇 개 이상 맞으면 통과로 볼지의 기준. 전부 맞을 필요는 없게 두어 화자 차이를 흡수한다.",
     "비교기의 기준전압"],
    ["PASS(s)",
     "그 시각의 판정 결과. 이후 채널이 변해도 유지된다.",
     "PASS 플립플롭"],
    ["L1 on g",
     "'상관없음'을 늘리도록 미는 힘. 정확도를 조금 내주는 대신 필요한 저항 수를 줄인다.",
     "학습에만 쓰이고 회로에는 남지 않음"],
  ],
  [2050, 4400, 2390],
),
gap(80),
p("형판의 세 값 중 '상관없음'이 많을수록 저항이 빠지고 화자 변동에 둔감해진다. " +
  "그래서 L1 페널티의 세기를 조절해 가며 정확도와 부품 수의 곡선을 그리고, 그 위에서 " +
  "실장 가능한 지점을 고른다. 무늬 비교와 개수 세기는 계단 함수라 그대로는 학습이 " +
  "되지 않으므로, 기존 이진 신경망 과제에서 채널 문턱을 학습할 때 쓰던 것과 같은 " +
  "STE(straight-through estimator) 기법을 적용한다."),
p("시각 τ(s)와 START 검출 조건은 학습으로 미분해 구할 수 있는 값이 아니므로 후보를 " +
  "전수 탐색하고, 후보마다 나머지를 학습시켜 비교한다. 학습 중에는 START 시점을 " +
  "±1 프레임 흔들어 넣어, 실제 회로에서 START가 100 Hz 클럭과 비동기이기 때문에 생기는 " +
  "정렬 오차에 미리 대비한다."),

// ── 5.9
h3("5.9 회로가 학습에 거는 제약"),
p("모델을 회로보다 자유롭게 두면 학습 결과를 실장할 수 없다. 다음 제약을 학습 단계에서 " +
  "미리 강제한다."),
table(
  ["제약", "이유"],
  [
    ["채널 가중치 없음 (형판은 1/0/X 세 값)", "저항을 전부 같은 값으로 두어야 오차가 판정에 실리지 않는다."],
    ["상태 수 4 이하", "상태 하나가 비교기 1개·플립플롭 1개·저항 한 벌을 추가한다."],
    ["τ(s)는 10 ms 정수배", "카운터 격자 위에서만 디코드할 수 있다."],
    ["τ(1) ≥ 1", "IDLE에서 카운터가 0으로 묶여 있으므로 0은 디코드해도 뜨지 않는다."],
    ["timeout > τ(4)", "마지막 채점 전에 회로가 리셋되면 WAKE가 나올 수 없다."],
    ["START 정렬 오차 ±1 프레임", "START는 100 Hz 클럭과 비동기이다."],
  ],
  [3600, 5240],
),
gap(),

// ── 5.10
h3("5.10 학습 결과에서 회로 상수로"),
table(
  ["학습 산출물", "회로에서의 형태"],
  [
    ["형판 M(s,c)", "상태 s의 저항망에 어느 채널을 어느 극성으로 연결할지. X는 미연결."],
    ["허용 오차 k(s)", "상태 s 비교기의 기준전압 분압비. V_TH = VDD × (k−0.5) / m(s)."],
    ["시각 τ(s)", "Time decode의 디코더 출력 결선."],
    ["timeout", "Time decode의 DECODE_TO 결선. WAKE 폭도 함께 정해진다."],
    ["START 검출 조건", "START detect의 게이트 조합 또는 합산 저항과 기준전압."],
  ],
  [2500, 6340],
),
gap(),

// ── 5.11
h3("5.11 검증 계획"),
p("판별부는 두 단계로 검증한다. 먼저 학습 모델과 동일한 순서로 동작하는 논리 시뮬레이션을 " +
  "작성해 두 결과가 비트 단위로 일치하는지 확인한다. 여기까지가 설계 오류를 잡는 단계이다. " +
  "다음으로 소자 오차·전파 지연·비교기 오프셋을 넣은 회로 시뮬레이션을 돌려, 저항 오차 " +
  "1%와 비교기 오프셋 범위 안에서 판정이 유지되는지 확인한다."),
p("평가 지표는 정확도가 아니라 검출 성능으로 보고한다. 목표 키워드를 정확히 잡아내는 " +
  "비율(TPR)과, 키워드가 없는 연속 음성·잡음 구간에서 WAKE가 잘못 뜨는 빈도(시간당 오검출 " +
  "횟수, FA/h) 두 가지를 함께 제시한다. 상시 대기 회로에서는 후자가 사용성을 좌우한다."),
];

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [{
    properties: { page: { margin: { top: 1420, right: 1700, bottom: 1420, left: 1700 } } },
    children: K,
  }],
});

Packer.toBuffer(doc).then((b) => {
  const out = `${__dirname}/5절_교체안.docx`;
  fs.writeFileSync(out, b);
  const chars = K.filter((x) => x.constructor === Paragraph).length;
  console.log(`${out}  ${(b.length / 1024).toFixed(0)} KB, 단락 ${chars}개`);
});
