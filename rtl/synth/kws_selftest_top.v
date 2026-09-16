// 자체 검사 최상위 -- 하네스 + 네트워크. 아날로그 경계가 없다.
//
//   kws_selftest_top
//     +- kws_selftest      클립 ROM + 정답 ROM + FSM + 카운터
//     +- kws_top_synth     매크로 바인딩 (기계 생성, 그대로 둔다)
//          +- kws_top
//
// kws_board_top 과 **형제**이고 대체물이 아니다. 둘이 묻는 질문이 다르다:
//
//   kws_board_top      실시간. cmp 16 가닥 -> 10 ms 프레임 -> 추론 -> LED
//   kws_selftest_top   검증.   ROM 클립 -> ready 속도로 -> 추론 -> 채점
//
// 그래서 kws_frame_ctrl 이 여기 없다. 10 ms 타이머는 실시간 요구이고, 저장된
// 벡터에는 그 요구가 없다 -- 클립당 57.4 ms 로 끝나는 이유다.
//
// ---- 결과를 어떻게 읽어낼 것인가는 아직 안 정했다 -------------------------- //
//
// 지금은 포트로 낸다. 시뮬에서는 테스트벤치가 그대로 읽으므로 이것으로 충분하고,
// **시뮬에서 하네스 자체를 먼저 검증하는 것이 순서**다 -- 하네스가 프레임을
// 잘못 먹이면 "칩이 틀렸다" 로 나타나고, 그게 가장 시간을 많이 먹는 오진이다.
//
// 보드로 갈 때 이 포트들(약 60 비트)을 핀에 다 낼 수는 없다. 선택지:
//
//   VIO        JTAG 으로 레지스터를 읽는다. 핀 0 개. IP 하나 + program.tcl 몇 줄
//   직렬화     한두 핀으로 밀어내고 스코프/LA 로 받는다
//   일부만     done / any_fail / first_fail_idx 만 EXT 여유 핀에 (EXT0~29 가 빈다)
//
// VIO 가 맞는 답으로 보이지만 하네스가 시뮬에서 깨끗해진 뒤에 붙인다.

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module kws_selftest_top #(
    // 몇 개를 돌릴 것인가. ROM 이 이 수만큼 선언되므로 BRAM 예산이 여기서
    // 정해진다 -- 32 비트 폭에서 약 680 개가 상한(docs/fpga_primitives.md 4).
    parameter integer CLIPS     = 600,
    // rtl/build_selftest.tcl 이 1000 클립 벡터에서 잘라 만든 파일들.
    parameter         CLIP_FILE = "selftest_clips.hex",
    parameter         EXP_FILE  = "selftest_expected.hex"
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        go,

    output wire        done,
    output wire [15:0] total,
    output wire [15:0] match,
    output wire        any_fail,
    output wire [15:0] first_fail_idx,
    output wire [3:0]  first_fail_got,
    output wire [3:0]  first_fail_exp
);

    wire                    net_start;
    wire                    net_valid;
    wire [`KWS_N_CH-1:0]    net_frame;
    wire                    net_ready;
    wire                    net_busy;
    wire                    net_cls_valid;
    wire [3:0]              net_cls;

    kws_selftest #(
        .N_CH      (`KWS_N_CH),
        .T_IN      (`KWS_T),
        .WORD_BITS (`KWS_WORD_BITS),
        .CLS_BITS  (4),
        .CLIPS     (CLIPS),
        .CLIP_FILE (CLIP_FILE),
        .EXP_FILE  (EXP_FILE)
    ) u_st (
        .clk           (clk),
        .rst_n         (rst_n),
        .go            (go),
        .net_start     (net_start),
        .net_valid     (net_valid),
        .net_frame     (net_frame),
        .net_ready     (net_ready),
        .net_busy      (net_busy),
        .net_cls_valid (net_cls_valid),
        .net_cls       (net_cls),
        .done          (done),
        .total         (total),
        .match         (match),
        .any_fail      (any_fail),
        .first_fail_idx(first_fail_idx),
        .first_fail_got(first_fail_got),
        .first_fail_exp(first_fail_exp)
    );

    // 스트리밍 게이트의 출력 셋은 여기서 쓰지 않는다. 자체 검사가 묻는 것은
    // **창 하나가 맞게 분류되는가**이고, 투표/margin 은 그 위에 얹힌 훨씬 작은
    // 로직이라 짧은 벡터로 따로 검증한다(docs/windowing_plan.md 단계 4).
    //
    // 이름을 붙여 두는 이유: 빈 포트로 두면 `score_valid remains unconnected`
    // 경고가 나고, 그 경고는 "연결을 잊었다" 와 "일부러 안 썼다" 를 구별하지
    // 못한다. kws_block.v 가 t_rom 에 쓰는 것과 같은 관용구다.
    /* verilator lint_off UNUSEDSIGNAL */
    wire                                     unused_score_valid;
    wire [3:0]                               unused_keyword_idx;
    wire signed [`KWS_CONV4_POOL_BITS:0]     unused_keyword_margin;
    /* verilator lint_on UNUSEDSIGNAL */

    kws_top_synth u_net (
        .clk           (clk),
        .rst_n         (rst_n),
        .start         (net_start),
        .in_valid      (net_valid),
        .in_frame      (net_frame),
        .in_ready      (net_ready),
        .busy          (net_busy),
        .class_valid   (net_cls_valid),
        .class_idx     (net_cls),
        .score_valid   (unused_score_valid),
        .keyword_idx   (unused_keyword_idx),
        .keyword_margin(unused_keyword_margin)
    );

endmodule

`default_nettype wire
