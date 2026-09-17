// 자체 검사를 보드에 올리는 최상위 -- kws_selftest_top + VIO.
//
//   kws_selftest_board          핀: clk, rst_n 두 개뿐
//     +- vio_st                 JTAG 으로 결과를 읽고 go / 소프트 리셋을 준다
//     +- kws_selftest_top       시뮬에서 검증한 그대로 (tb_selftest)
//
// kws_selftest_top 을 고치지 않고 **한 겹 감싼다.** 시뮬이 통과한 모듈과 칩에 올라가는
// 모듈이 같아야 칩 결과를 시뮬 결과와 비교할 수 있다. VIO 는 시뮬에 없는 IP 라
// 안쪽에 넣으면 tb_selftest 가 더는 돌지 않는다.
//
// ---- 왜 VIO 인가 ---------------------------------------------------------- //
//
// 결과가 약 58 비트다(done/total/match/any_fail/first_fail_*). 핀으로 내려면
// EXT 핀 58 개 + 그걸 읽을 장비가 필요하다. VIO 는 칩 안의 레지스터를 **이미 꽂혀
// 있는 JTAG 케이블**로 읽는다 -- 추가 배선 0, LED/FND 0. 사람이 읽는 창은
// Vivado Hardware Manager 이고, rtl/selftest/run_selftest.tcl 이 배치로도 읽는다.
//
// 부수 효과로 **진행이 보인다**: total 이 34 초 동안 0 -> 600 으로 올라간다.
//
// ---- 조작 ----------------------------------------------------------------- //
//
//   go      (probe_out0)  0 -> 1 로 바꾸는 순간 한 번 시작. 1 로 둬도 다시 안 돈다.
//   soft_rst(probe_out1)  1 인 동안 리셋. 두 번째 실행 전에 1 -> 0 을 한 번 준다.
//
// 두 번째 실행에 리셋을 요구하는 이유: kws_selftest 는 go 에서 total/match 만
// 지우고 any_fail / first_fail_* 는 **첫 실패를 보존**한다(그게 그 레지스터의
// 목적이다). 결정성을 보려고 두 번 돌릴 때 첫 실행의 실패가 남아 있으면 둘째
// 실행이 깨끗한지 알 수 없다. 리셋이 모두 지운다.

`timescale 1ns/1ps
`default_nettype none

module kws_selftest_board #(
    parameter integer CLIPS     = 600,
    parameter         CLIP_FILE = "selftest_clips.hex",
    parameter         EXP_FILE  = "selftest_expected.hex"
) (
    input  wire clk,
    input  wire rst_n          // 핀(EXT16). 비동기. 내부 풀업 -- 안 꽂으면 해제
);

    // ---- VIO ------------------------------------------------------------- //
    // 포트 폭은 rtl/build.tcl 의 create_ip 설정과 **짝**이다. 한쪽만 바꾸면
    // 합성이 포트 폭 불일치로 멈춘다(조용히 잘리지 않는다 -- 블랙박스 스텁이
    // 폭을 못박으므로).
    wire        vio_go;
    wire        vio_rst;

    wire        done;
    wire [15:0] total;
    wire [15:0] match;
    wire        any_fail;
    wire [15:0] first_fail_idx;
    wire [3:0]  first_fail_got;
    wire [3:0]  first_fail_exp;

    vio_st u_vio (
        .clk       (clk),
        .probe_in0 (done),
        .probe_in1 (total),
        .probe_in2 (match),
        .probe_in3 (any_fail),
        .probe_in4 (first_fail_idx),
        .probe_in5 (first_fail_got),
        .probe_in6 (first_fail_exp),
        .probe_out0(vio_go),
        .probe_out1(vio_rst)
    );

    // ---- 리셋 ------------------------------------------------------------ //
    // kws_board_top 과 같은 관용구: 걸 때는 비동기, 놓을 때는 동기.
    // 핀 리셋과 VIO 리셋 중 하나라도 걸리면 리셋이다.
    wire rst_any_n = rst_n & ~vio_rst;

    (* ASYNC_REG = "TRUE" *) reg [1:0] rst_sync;
    always @(posedge clk or negedge rst_any_n) begin
        if (!rst_any_n) rst_sync <= 2'b00;
        else            rst_sync <= {rst_sync[0], 1'b1};
    end
    wire rst_n_s = rst_sync[1];

    // ---- go: 올라가는 에지 한 번 = 펄스 한 사이클 ----------------------------- //
    // VIO 출력은 clk 도메인 레지스터라 동기화기가 필요 없다. 에지로 자르는
    // 이유: kws_selftest 는 S_DONE 에서 go 를 보면 S_IDLE 로, S_IDLE 에서 보면
    // 시작으로 간다. 레벨을 그대로 넘기면 1 로 둔 동안 끝없이 다시 돈다.
    //
    // 리셋값이 1 인 이유: go 를 1 로 둔 채 soft_rst 를 놓으면, 0 이었다면 그 순간
    // "에지" 가 생겨 **저절로 시작**한다. 1 이면 사람이 go 를 0 -> 1 로 다시 줘야
    // 한다 -- 시작은 항상 명시적이다.
    reg go_q;
    always @(posedge clk or negedge rst_n_s) begin
        if (!rst_n_s) go_q <= 1'b1;
        else          go_q <= vio_go;
    end
    wire go_pulse = vio_go & ~go_q;

    kws_selftest_top #(
        .CLIPS     (CLIPS),
        .CLIP_FILE (CLIP_FILE),
        .EXP_FILE  (EXP_FILE)
    ) u_top (
        .clk            (clk),
        .rst_n          (rst_n_s),
        .go             (go_pulse),
        .done           (done),
        .total          (total),
        .match          (match),
        .any_fail       (any_fail),
        .first_fail_idx (first_fail_idx),
        .first_fail_got (first_fail_got),
        .first_fail_exp (first_fail_exp)
    );

endmodule

`default_nettype wire
