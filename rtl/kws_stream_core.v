// Continuous KWS composition: capture/window -> folded network -> vote.
//
// Policy parameters are deliberately ports of this reusable composition. The
// deployable kws_stream_top passes values emitted into parameters.vh; this file
// does not contain the validation-selected margin as a literal.

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module kws_stream_core #(
    parameter integer N_CH             = 16,
    parameter integer FRAME_CYCLES     = 500000,
    parameter integer NATIVE_T         = 100,
    parameter integer PAD_LEFT         = 14,
    parameter integer PAD_RIGHT        = 14,
    parameter integer T                = 128,
    parameter integer TRIGGER_FRAMES   = 1,
    parameter integer REQUIRED         = 1,
    parameter integer COOLDOWN_WINDOWS = 0,
    parameter integer MARGIN_BITS      = 22,
    parameter integer MARGIN_INT       = 0,
    parameter integer CMP_INVERT       = 0
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire            force_start,
    input  wire [N_CH-1:0] cmp,

    output wire            busy,
    output wire            detection_valid,
    output wire [3:0]      detection_idx,
    output wire [7:0]      overrun_count,

    // Per-window network result, the input of the vote. Observation only: the
    // streaming self-test checks every window against the Python integer
    // reference, not just the final detections. The deployable top leaves
    // these unconnected.
    output wire                          obs_score_valid,
    output wire [3:0]                    obs_keyword_idx,
    output wire signed [MARGIN_BITS-1:0] obs_keyword_margin
);

    wire            clip_start;
    wire            frame_valid;
    wire [N_CH-1:0] frame;
    wire            window_busy;
    wire            window_overrun;
    wire            net_ready;
    wire            net_busy;

    kws_window #(
        .N_CH(N_CH), .FRAME_CYCLES(FRAME_CYCLES),
        .NATIVE_T(NATIVE_T), .PAD_LEFT(PAD_LEFT),
        .PAD_RIGHT(PAD_RIGHT), .T(T),
        .TRIGGER_FRAMES(TRIGGER_FRAMES), .CMP_INVERT(CMP_INVERT)
    ) u_window (
        .clk(clk), .rst_n(rst_n), .cmp(cmp),
        .force_start(force_start), .launch_ready(!net_busy),
        .clip_start(clip_start),
        .out_valid(frame_valid), .out_frame(frame),
        .out_ready(net_ready), .busy(window_busy),
        .overrun(window_overrun),
        .overrun_count(overrun_count)
    );

    wire       raw_class_valid;
    wire [3:0] raw_class_idx;
    wire       score_valid;
    wire [3:0] keyword_idx;
    wire signed [MARGIN_BITS-1:0] keyword_margin;

    kws_top_synth u_net (
        .clk(clk), .rst_n(rst_n), .start(clip_start),
        .in_valid(frame_valid), .in_frame(frame), .in_ready(net_ready),
        .busy(net_busy), .class_valid(raw_class_valid),
        .class_idx(raw_class_idx), .score_valid(score_valid),
        .keyword_idx(keyword_idx), .keyword_margin(keyword_margin)
    );

    // A skipped request occurs chronologically after the result currently in
    // flight. Remember it, let that result reach the voter, then break the
    // streak on the following cycle. At the selected 57.4 ms / 100 ms timing
    // this path should remain inactive; it makes a timing violation visible
    // without turning non-adjacent windows into an apparent streak.
    reg gap_pending, gap_after_result;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gap_pending     <= 1'b0;
            gap_after_result <= 1'b0;
        end else begin
            gap_after_result <= 1'b0;
            if (window_overrun)
                gap_pending <= 1'b1;
            if (score_valid && (gap_pending || window_overrun)) begin
                gap_pending      <= 1'b0;
                gap_after_result <= 1'b1;
            end
        end
    end

    wire margin_accept_nc, armed_nc;
    kws_vote #(
        .CLASS_BITS(4), .MARGIN_BITS(MARGIN_BITS), .MARGIN_INT(MARGIN_INT),
        .REQUIRED(REQUIRED), .COOLDOWN_WINDOWS(COOLDOWN_WINDOWS)
    ) u_vote (
        .clk(clk), .rst_n(rst_n),
        .score_valid(score_valid), .keyword_idx(keyword_idx),
        .keyword_margin(keyword_margin), .window_gap(gap_after_result),
        .detection_valid(detection_valid), .detection_idx(detection_idx),
        .margin_accept(margin_accept_nc), .armed(armed_nc)
    );

    assign busy = window_busy || net_busy;

    assign obs_score_valid    = score_valid;
    assign obs_keyword_idx    = keyword_idx;
    assign obs_keyword_margin = keyword_margin;

`ifdef KWS_ASSERT
    initial begin
        if (MARGIN_BITS != `KWS_CONV4_POOL_BITS + 1) begin
            $display("ASSERT %m: MARGIN_BITS=%0d, generated pool difference needs %0d",
                     MARGIN_BITS, `KWS_CONV4_POOL_BITS + 1);
            $finish;
        end
    end
    always @(posedge clk) if (clip_start && net_busy) begin
        $display("ASSERT %m: clip_start while folded network is busy");
        $finish;
    end
`endif

endmodule

`default_nettype wire
