// Streaming self-test on the board: kws_stream_selftest_top + VIO.
//
// Same wrapper pattern as kws_selftest_board.v: pins are clk and rst_n only,
// results are read over JTAG by rtl/selftest/run_stream_selftest.tcl.
//   go       (probe_out0) 0 -> 1 edge starts one sweep over all cases
//   soft_rst (probe_out1) 1 = reset (clears counters and first-failure record)
// The VIO port widths are paired with the vio_sst settings in rtl/build.tcl.

`timescale 1ns/1ps
`default_nettype none

module kws_stream_selftest_board #(
    parameter integer CASES        = 240,
    parameter integer FRAME_CYCLES = 500000,
    parameter         FRAME_FILE   = "stream_frames.hex",
    parameter         EXP_FILE     = "stream_expected.hex"
) (
    input  wire clk,
    input  wire rst_n
);

    wire        vio_go, vio_rst;
    wire        done, any_fail;
    wire [15:0] total, match, hit, wrong, outside, quiet_false, first_fail_case;
    wire [39:0] first_fail_got, first_fail_exp;
    wire [4:0]  window_now;
    wire [7:0]  overrun_count;

    vio_sst u_vio (
        .clk        (clk),
        .probe_in0  (done),
        .probe_in1  (total),
        .probe_in2  (match),
        .probe_in3  (any_fail),
        .probe_in4  (first_fail_case),
        .probe_in5  (hit),
        .probe_in6  (wrong),
        .probe_in7  (quiet_false),
        .probe_in8  (outside),
        .probe_in9  (first_fail_got),
        .probe_in10 (first_fail_exp),
        .probe_in11 (window_now),
        .probe_in12 (overrun_count),
        .probe_out0 (vio_go),
        .probe_out1 (vio_rst)
    );

    wire rst_any_n = rst_n & ~vio_rst;
    (* ASYNC_REG = "TRUE" *) reg [1:0] rst_sync;
    always @(posedge clk or negedge rst_any_n) begin
        if (!rst_any_n) rst_sync <= 2'b00;
        else            rst_sync <= {rst_sync[0], 1'b1};
    end
    wire rst_n_s = rst_sync[1];

    // reset value 1: releasing soft_rst with go held high must not start a run
    reg go_q;
    always @(posedge clk or negedge rst_n_s) begin
        if (!rst_n_s) go_q <= 1'b1;
        else          go_q <= vio_go;
    end
    wire go_pulse = vio_go & ~go_q;

    kws_stream_selftest_top #(
        .CASES(CASES), .FRAME_CYCLES(FRAME_CYCLES),
        .FRAME_FILE(FRAME_FILE), .EXP_FILE(EXP_FILE)
    ) u_top (
        .clk(clk), .rst_n(rst_n_s), .go(go_pulse),
        .done(done), .total(total), .match(match), .hit(hit), .wrong(wrong),
        .outside(outside), .quiet_false(quiet_false), .any_fail(any_fail),
        .first_fail_case(first_fail_case), .first_fail_got(first_fail_got),
        .first_fail_exp(first_fail_exp), .window_now(window_now),
        .overrun_count(overrun_count)
    );

endmodule

`default_nettype wire
