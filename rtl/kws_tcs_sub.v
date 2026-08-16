// One TCS sub-block: depthwise -> threshold -> pointwise.
//
// models/binary_matchboxnet.py _TCSSub is literally
//     pw(sign(bn(dw(x))))
// and both halves are already built and verified, so this is wiring plus one
// handshake. Nothing here does arithmetic.
//
// THE re-binarisation BETWEEN THEM IS NOT OPTIONAL. The pointwise half is
// XNOR-popcount, which requires +-1 inputs; the depthwise accumulator is an
// integer. kws_dw_conv already ends in the fused threshold, so its output is
// the {-1,+1} the pointwise needs and the boundary is where CLAUDE.md 3.4 puts
// it -- not an extra stage bolted on here.
//
// SCOPE. This matches a NON-FINAL sub-block, whose pointwise ends in a
// threshold. The last sub-block of a residual block feeds its raw integer
// accumulator into the residual add instead (manifest: epilogue "none"), so
// kws_block drives that path itself rather than reusing this module. Wiring
// this one in there would silently insert a threshold that the trained network
// does not have.
//
// Sequential, not overlapped: dw runs, then pw runs, then the next frame. At
// 10 Hz inference the folded datapath has three orders of magnitude of slack
// (CLAUDE.md 0), and a pipelined handoff would need a frame buffer between the
// halves for no reachable benefit.

`timescale 1ns/1ps
`default_nettype none

module kws_tcs_sub #(
    parameter integer C_IN      = 128,   // block input channels
    parameter integer C_OUT     = 64,
    parameter integer K         = 13,
    parameter integer PAD       = 6,
    parameter integer DW_ACC    = 5,     // manifest acc_bits, depthwise
    parameter integer PW_ACC    = 9,     // manifest acc_bits, pointwise
    parameter integer WORD_BITS = 32,
    parameter DW_W_FILE = "", parameter DW_T_FILE = "",
    parameter PW_W_FILE = "", parameter PW_T_FILE = ""
) (
    input  wire              clk,
    input  wire              rst_n,

    input  wire              start,      // new clip
    input  wire              in_push,
    input  wire              in_real,
    input  wire [C_IN-1:0]   in_frame,

    output wire              busy,
    output wire              out_valid,
    output wire [C_OUT-1:0]  out_frame
);

    // depthwise keeps the channel count; only the pointwise mixes
    wire              dw_busy;
    wire              dw_ov;
    wire [C_IN-1:0]   dw_of;

    kws_dw_conv #(.C(C_IN), .K(K), .PAD(PAD), .ACC_BITS(DW_ACC),
                  .WORD_BITS(WORD_BITS),
                  .W_FILE(DW_W_FILE), .T_FILE(DW_T_FILE)) u_dw (
        .clk(clk), .rst_n(rst_n), .start(start),
        .in_push(in_push), .in_real(in_real), .in_frame(in_frame),
        .busy(dw_busy), .out_valid(dw_ov), .out_frame(dw_of));

    // one-cycle handoff. dw_ov is a pulse, and kws_pw_conv latches in_frame on
    // acceptance, so the frame does not need holding beyond it.
    reg              pw_iv;
    reg [C_IN-1:0]   pw_if;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pw_iv <= 1'b0;
            pw_if <= {C_IN{1'b0}};
        end else begin
            pw_iv <= dw_ov;
            if (dw_ov) pw_if <= dw_of;
        end
    end

    wire pw_busy;
    kws_pw_conv #(.C_IN(C_IN), .C_OUT(C_OUT), .ACC_BITS(PW_ACC),
                  .WORD_BITS(WORD_BITS),
                  .W_FILE(PW_W_FILE), .T_FILE(PW_T_FILE)) u_pw (
        .clk(clk), .rst_n(rst_n),
        .in_valid(pw_iv), .in_frame(pw_if),
        .busy(pw_busy), .out_valid(out_valid), .out_frame(out_frame));

    // BOTH handoff cycles count. dw_busy falls on the edge that raises dw_ov,
    // and pw_iv does not rise until the edge after that, so leaving dw_ov out
    // opens a one-cycle hole where busy reads low while a frame is in flight.
    // The caller then pushes into dw while pw is about to start, and the frame
    // that was in flight never produces an output.
    assign busy = dw_busy | dw_ov | pw_iv | pw_busy;

`ifdef KWS_ASSERT
    always @(posedge clk) if (in_push && busy) begin
        $display("ASSERT %m: pushed while busy");
        $finish;
    end
    // the halves must never be mid-frame at the same time; if they are, the
    // sequential assumption above has been broken by a caller pushing early
    always @(posedge clk) if (dw_busy && pw_busy) begin
        $display("ASSERT %m: dw and pw both busy -- handoff overlapped");
        $finish;
    end
`endif

endmodule

`default_nettype wire
