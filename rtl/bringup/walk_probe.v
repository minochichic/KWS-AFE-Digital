// 브링업 2 — 한 번에 한 핀씩 High 로 흘려서 핀 배정과 클럭을 동시에 확인한다.
//
// vcco_probe 와 나눈 이유: 저쪽은 클럭을 안 쓰므로 실패하면 툴체인/JTAG 문제다.
// 이쪽이 추가로 답하는 것은 **클럭이 살아 있는가** 와 **핀 순서가 매뉴얼 표대로인가**
// 둘이고, 저쪽이 통과한 뒤에 돌려야 원인이 섞이지 않는다.
//
// DMM 한 대로 핀 배정을 확인할 수 있게 만든 것이 핵심이다. 정적 패턴으로는
// 핀당 1 비트뿐이라 16 개를 구별할 수 없지만, **시간축을 쓰면 구별된다** --
// 프로브를 한 핀에 대고 몇 번째 칸에서 올라오는지 세면 그게 채널 번호다.
//
// 클럭이 죽어 있으면: 카운터가 안 돌아 ch00 만 계속 High 로 남는다. 즉
//   전부 0 V        -> 비트스트림이 안 올라갔거나 핀 배정이 틀림
//   ch00 만 계속 High -> 비트스트림은 올라갔고 **클럭이 안 들어온다**
//   차례로 흐름       -> 전부 정상
// 세 경우가 눈으로 구분된다. 그래서 리셋 후 상태를 0 으로 두는 것이 의도다.
//
// ⚠️ AFE 를 물린 채로 올리지 말 것 -- vcco_probe.v 의 경고와 같다.

`timescale 1ns/1ps
`default_nettype none

module walk_probe #(
    parameter integer N_CH     = 16,
    // 킷의 Clock control block 은 16 단이다(docs/hanback_kit.md 3.2). 스위치를
    // 다른 자리에 두면 여기만 고쳐 다시 합성한다 -- F=50 MHz, E=25 MHz, D=5 MHz.
    parameter integer CLK_HZ   = 50_000_000,
    // 사람이 세기 좋은 속도. 너무 빠르면 DMM 이 못 따라가고(응답이 수백 ms),
    // 너무 느리면 16 칸 도는 데 오래 걸린다. 2 초면 한 바퀴가 32 초다.
    parameter integer DWELL_MS = 2000
) (
    input  wire            clk,
    output reg  [N_CH-1:0] ext
);

    // 나눗셈은 상수 폴딩으로 합성 시점에 끝난다. 런타임 나눗셈이 아니다.
    localparam integer DWELL_CYCLES = (CLK_HZ / 1000) * DWELL_MS;
    localparam integer CW = (DWELL_CYCLES <= 2) ? 1 : $clog2(DWELL_CYCLES);
    localparam integer SW = (N_CH <= 2) ? 1 : $clog2(N_CH);

    reg [CW-1:0] tick  = {CW{1'b0}};
    reg [SW-1:0] state = {SW{1'b0}};

    // 리셋 포트가 없다. 킷의 리셋 버튼 핀을 아직 모르고(매뉴얼의 그 페이지가 없다),
    // FPGA 는 컨피그 직후 레지스터 초기값으로 시작하므로 초기화 구문이면 충분하다.
    // 이 설계에는 되돌릴 상태도 없다.
    always @(posedge clk) begin
        if (tick == DWELL_CYCLES - 1) begin
            tick  <= {CW{1'b0}};
            state <= (state == N_CH - 1) ? {SW{1'b0}} : state + 1'b1;
        end else begin
            tick <= tick + 1'b1;
        end
    end

    // 디코더 하나. state 는 SW 비트이고 N_CH 에서 되돌므로 색인이 범위를 벗어날 수
    // 없다. for 루프로 비교식을 16 개 펼치는 것보다 의도가 분명하다.
    always @(*) begin
        ext        = {N_CH{1'b0}};
        ext[state] = 1'b1;
    end

endmodule

`default_nettype wire
