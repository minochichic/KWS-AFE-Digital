// Streaming self-test top: harness + the deployable always-on core.
//
//   kws_stream_selftest_top
//     +- kws_stream_selftest   stream ROM, frame driver aligned to capture, scorer
//     +- kws_stream_core       capture -> window -> network -> vote (as deployed)
//
// Policy constants come from parameters.vh exactly as in kws_stream_top, so the
// core under test is configured by the same macros as the deployable top.
// Shared by tb_stream_selftest (simulation) and kws_stream_selftest_board (chip).

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module kws_stream_selftest_top #(
    parameter integer CASES        = 240,
    parameter integer FRAMES       = 300,
    parameter integer WINDOWS      = 21,
    parameter integer FRAME_CYCLES = 500000,
    parameter         FRAME_FILE   = "stream_frames.hex",
    parameter         EXP_FILE     = "stream_expected.hex"
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        go,

    output wire        done,
    output wire [15:0] total,
    output wire [15:0] match,
    output wire [15:0] hit,
    output wire [15:0] wrong,
    output wire [15:0] outside,
    output wire [15:0] quiet_false,
    output wire        any_fail,
    output wire [15:0] first_fail_case,
    output wire [39:0] first_fail_got,
    output wire [39:0] first_fail_exp,
    output wire [4:0]  window_now,
    output wire [7:0]  overrun_count
);

    localparam integer MARGIN_BITS = `KWS_CONV4_POOL_BITS + 1;
    localparam integer PAD_RIGHT   = `KWS_T - `KWS_PAD_LEFT - `KWS_NATIVE_T;

    wire                          core_rst_n;
    wire [`KWS_N_CH-1:0]          cmp;
    wire                          score_valid;
    wire [3:0]                    keyword_idx;
    wire signed [MARGIN_BITS-1:0] keyword_margin;
    wire                          det_valid;
    wire [3:0]                    det_idx;

    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_busy;
    /* verilator lint_on UNUSEDSIGNAL */

    kws_stream_selftest #(
        .N_CH(`KWS_N_CH), .FRAME_CYCLES(FRAME_CYCLES), .CASES(CASES),
        .FRAMES(FRAMES), .WINDOWS(WINDOWS), .HOP(`KWS_STREAM_HOP_FRAMES),
        .NATIVE_T(`KWS_NATIVE_T), .MARGIN_BITS(MARGIN_BITS),
        .FRAME_FILE(FRAME_FILE), .EXP_FILE(EXP_FILE)
    ) u_st (
        .clk(clk), .rst_n(rst_n), .go(go),
        .core_rst_n(core_rst_n), .cmp(cmp),
        .obs_score_valid(score_valid), .obs_keyword_idx(keyword_idx),
        .obs_keyword_margin(keyword_margin),
        .det_valid(det_valid), .det_idx(det_idx), .overrun_count(overrun_count),
        .done(done), .total(total), .match(match), .hit(hit), .wrong(wrong),
        .outside(outside), .quiet_false(quiet_false), .any_fail(any_fail),
        .first_fail_case(first_fail_case), .first_fail_got(first_fail_got),
        .first_fail_exp(first_fail_exp), .window_now(window_now)
    );

    kws_stream_core #(
        .N_CH(`KWS_N_CH), .FRAME_CYCLES(FRAME_CYCLES),
        .NATIVE_T(`KWS_NATIVE_T), .PAD_LEFT(`KWS_PAD_LEFT),
        .PAD_RIGHT(PAD_RIGHT), .T(`KWS_T),
        .TRIGGER_FRAMES(`KWS_STREAM_HOP_FRAMES),
        .REQUIRED(`KWS_STREAM_REQUIRED),
        .COOLDOWN_WINDOWS(`KWS_STREAM_COOLDOWN_WINDOWS),
        .MARGIN_BITS(MARGIN_BITS),
        .MARGIN_INT(`KWS_STREAM_MARGIN_INT), .CMP_INVERT(0)
    ) u_core (
        .clk(clk), .rst_n(core_rst_n), .force_start(1'b0), .cmp(cmp),
        .busy(unused_busy), .detection_valid(det_valid), .detection_idx(det_idx),
        .overrun_count(overrun_count),
        .obs_score_valid(score_valid), .obs_keyword_idx(keyword_idx),
        .obs_keyword_margin(keyword_margin)
    );

endmodule

`default_nettype wire
