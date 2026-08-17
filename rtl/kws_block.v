// One residual TCS block: two sub-blocks, a skip path, and an integer add.
//
// models/binary_matchboxnet.py BinaryTCSBlock:
//     res = skip(x)                    # integer, unscaled
//     y   = sub0(x)   -> BN -> sign
//     acc = sub1(y)                    # integer, NOT thresholded
//     out = sign(BN(acc + res))        # ONE threshold, after the add
//
// TWO THINGS THIS CANNOT DO BY REUSING kws_tcs_sub TWICE.
//
// 1. The last sub-block's pointwise has no threshold (manifest epilogue
//    "none"). Its raw accumulator is what gets added. Wiring kws_tcs_sub there
//    would insert a threshold the trained network does not have -- a little
//    accuracy and no error message.
// 2. The residual is added in the INTEGER domain, before any threshold, so the
//    skip is also an unscaled pointwise accumulator. Adding +-1 outputs instead
//    would be a different network.
//
// ALIGNMENT IS THE HARD PART. Each depthwise delays its output by PAD frames
// (its line buffer runs PAD pushes ahead, docs/diagrams/24_pipeline_drain.svg),
// so the sub chain emits frame t only 2*PAD pushes after x[t] went in. The skip
// needs that same x[t]. Rather than hold 2*PAD frames of 64 accumulators, the
// block delays the INPUT -- 2*PAD x 128 bits of shift register instead of
// 2*PAD x 64 x 9 -- and runs the skip late, on the delayed frame.
//
// DRAIN PROPAGATES BY HAND. sub0 emits only real output frames, so its own
// drain produces real pushes into sub1's depthwise and never a flush. When the
// caller's flush pushes stop producing sub0 output, this block injects the
// flushes sub1 still needs. Caller pushes T + 2*PAD; sub1's depthwise ends up
// with T real + PAD flush, which is exactly what it needs for T outputs.

`timescale 1ns/1ps
`default_nettype none

module kws_block #(
    parameter integer C_IN      = 128,   // block input channels
    parameter integer C_MID     = 64,    // after sub0's pointwise
    parameter integer C_OUT     = 64,
    parameter integer K         = 13,
    parameter integer PAD       = 6,
    parameter integer S0_DW_ACC = 5,
    parameter integer S0_PW_ACC = 9,
    parameter integer S1_DW_ACC = 5,
    parameter integer S1_PW_ACC = 8,
    parameter integer SKIP_ACC  = 9,
    parameter integer ADD_ACC   = 9,     // manifest: b1_add acc_bits
    parameter integer WORD_BITS = 32,
    parameter S0_DW_W = "", parameter S0_DW_T = "",
    parameter S0_PW_W = "", parameter S0_PW_T = "",
    parameter S1_DW_W = "", parameter S1_DW_T = "",
    parameter S1_PW_W = "",
    parameter SKIP_W  = "",
    parameter ADD_T   = ""               // the block's single threshold ROM
) (
    input  wire              clk,
    input  wire              rst_n,

    input  wire              start,
    input  wire              in_push,
    input  wire              in_real,
    input  wire [C_IN-1:0]   in_frame,

    output wire              busy,
    output reg               out_valid,
    output reg  [C_OUT-1:0]  out_frame
);

    localparam integer CO_BITS = (C_OUT <= 2) ? 1 : $clog2(C_OUT);
    localparam integer DEPTH   = 2 * PAD;        // frames the sub chain lags by
    localparam integer TH_BITS = (ADD_ACC + 1 > 33) ? ADD_ACC + 1 : 33;
    // one wider than every addend, so no sign extension below is zero-width
    localparam integer SUM_BITS = ADD_ACC + 1;

    localparam [2:0] S_IDLE = 3'd0, S_SUB0 = 3'd1, S_SUB1 = 3'd2,
                     S_SKIP = 3'd3, S_PW   = 3'd4;
    reg [2:0] st;
    assign busy = (st != S_IDLE);

    // ---- input delay line: x[t] must survive until the chain reaches t ---- //
    reg [C_IN-1:0] xdly [0:DEPTH-1];
    integer d;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || start)
            for (d = 0; d < DEPTH; d = d + 1) xdly[d] <= {C_IN{1'b0}};
        else if (in_push && !busy) begin
            for (d = 0; d < DEPTH - 1; d = d + 1) xdly[d] <= xdly[d + 1];
            xdly[DEPTH-1] <= in_frame;
        end
    end

    // ---- sub-block 0: dw -> threshold -> pw -> threshold ------------------ //
    wire              s0_busy, s0_ov;
    wire [C_MID-1:0]  s0_of;
    reg               s0_push;

    kws_tcs_sub #(.C_IN(C_IN), .C_OUT(C_MID), .K(K), .PAD(PAD),
                  .DW_ACC(S0_DW_ACC), .PW_ACC(S0_PW_ACC),
                  .WORD_BITS(WORD_BITS),
                  .DW_W_FILE(S0_DW_W), .DW_T_FILE(S0_DW_T),
                  .PW_W_FILE(S0_PW_W), .PW_T_FILE(S0_PW_T)) u_sub0 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .in_push(s0_push), .in_real(in_real), .in_frame(in_frame),
        .busy(s0_busy), .out_valid(s0_ov), .out_frame(s0_of));

    // ---- sub-block 1: dw -> threshold, then a pointwise with NO threshold -- //
    reg               s1_push, s1_real;
    reg  [C_MID-1:0]  s1_frame;
    wire              s1dw_busy, s1dw_ov;
    wire [C_MID-1:0]  s1dw_of;

    kws_dw_conv #(.C(C_MID), .K(K), .PAD(PAD), .ACC_BITS(S1_DW_ACC),
                  .WORD_BITS(WORD_BITS),
                  .W_FILE(S1_DW_W), .T_FILE(S1_DW_T)) u_s1_dw (
        .clk(clk), .rst_n(rst_n), .start(start),
        .in_push(s1_push), .in_real(s1_real), .in_frame(s1_frame),
        .busy(s1dw_busy), .out_valid(s1dw_ov), .out_frame(s1dw_of));

    reg [C_MID-1:0] y_lat;                 // sub1's depthwise output, held
    reg [C_IN-1:0]  x_lat;                 // the aligned block input for skip

    // These two instances have no threshold ROM (T_FILE is empty) because the
    // layers they implement end in epilogue "none" -- their accumulators are
    // what the residual add consumes. The thresholded outputs therefore carry
    // nothing meaningful and are deliberately dropped. Named rather than left
    // as empty pin connections so the intent is visible, and the suppression is
    // scoped to these four nets only.
    /* verilator lint_off UNUSEDSIGNAL */
    wire              skip_ov_nc, s1pw_ov_nc;
    wire [C_OUT-1:0]  skip_of_nc, s1pw_of_nc;
    /* verilator lint_on UNUSEDSIGNAL */

    // skip: unscaled pointwise on the DELAYED input; only its accumulator is
    // used, so T_FILE is left empty
    reg                          skip_iv;
    wire                         skip_busy, skip_av;
    wire [CO_BITS-1:0]           skip_ach;
    wire signed [SKIP_ACC-1:0]   skip_aval;

    kws_pw_conv #(.C_IN(C_IN), .C_OUT(C_OUT), .ACC_BITS(SKIP_ACC),
                  .WORD_BITS(WORD_BITS), .W_FILE(SKIP_W), .T_FILE(""),
                  .CO_BITS_P(CO_BITS)) u_skip (
        .clk(clk), .rst_n(rst_n),
        .in_valid(skip_iv), .in_frame(x_lat),
        .busy(skip_busy), .out_valid(skip_ov_nc), .out_frame(skip_of_nc),
        .acc_valid(skip_av), .acc_ch(skip_ach), .acc_out(skip_aval));

    // one accumulator per output channel, held while the last pointwise runs
    reg signed [SKIP_ACC-1:0] skip_acc [0:C_OUT-1];
    always @(posedge clk) if (skip_av) skip_acc[skip_ach] <= skip_aval;

    reg                          s1pw_iv;
    wire                         s1pw_busy, s1pw_av;
    wire [CO_BITS-1:0]           s1pw_ach;
    wire signed [S1_PW_ACC-1:0]  s1pw_aval;

    kws_pw_conv #(.C_IN(C_MID), .C_OUT(C_OUT), .ACC_BITS(S1_PW_ACC),
                  .WORD_BITS(WORD_BITS), .W_FILE(S1_PW_W), .T_FILE(""),
                  .CO_BITS_P(CO_BITS)) u_s1_pw (
        .clk(clk), .rst_n(rst_n),
        .in_valid(s1pw_iv), .in_frame(y_lat),
        .busy(s1pw_busy), .out_valid(s1pw_ov_nc), .out_frame(s1pw_of_nc),
        .acc_valid(s1pw_av), .acc_ch(s1pw_ach), .acc_out(s1pw_aval));

    // ---- the residual add, then ONE threshold ----------------------------- //
    reg [31:0] add_t_rom [0:2*C_OUT-1];
    initial if (ADD_T != "") $readmemh(ADD_T, add_t_rom);

    wire signed [SUM_BITS-1:0] a_pw =
        {{(SUM_BITS-S1_PW_ACC){s1pw_aval[S1_PW_ACC-1]}}, s1pw_aval};
    wire signed [SUM_BITS-1:0] a_sk =
        {{(SUM_BITS-SKIP_ACC){skip_acc[s1pw_ach][SKIP_ACC-1]}},
         skip_acc[s1pw_ach]};
    wire signed [SUM_BITS-1:0] sum = a_pw + a_sk;

    wire signed [31:0]        add_thr = add_t_rom[{1'b0, s1pw_ach}];
    wire                      add_ge  = add_t_rom[{1'b1, s1pw_ach}][0];
    wire signed [TH_BITS-1:0] sum_x =
        {{(TH_BITS-SUM_BITS){sum[SUM_BITS-1]}}, sum};
    wire signed [TH_BITS-1:0] thr_x = {{(TH_BITS-32){add_thr[31]}}, add_thr};
    wire                      add_fired = ((sum_x >= thr_x) == add_ge);

    always @(posedge clk) if (s1pw_av) out_frame[s1pw_ach] <= add_fired;

    // ---- sequencing -------------------------------------------------------- //
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; out_valid <= 1'b0;
            s0_push <= 1'b0; s1_push <= 1'b0; s1_real <= 1'b0;
            skip_iv <= 1'b0; s1pw_iv <= 1'b0;
            y_lat <= {C_MID{1'b0}}; x_lat <= {C_IN{1'b0}};
        end else begin
            out_valid <= 1'b0;
            s0_push   <= 1'b0;
            s1_push   <= 1'b0;
            skip_iv   <= 1'b0;
            s1pw_iv   <= 1'b0;
            case (st)
            S_IDLE:
                if (in_push && !start) begin
                    s0_push <= 1'b1;
                    st      <= S_SUB0;
                end
            S_SUB0:
                // sub0 is sequential inside; wait for it to settle, then hand
                // its frame on -- or inject a flush if the drain has reached it
                if (!s0_busy && !s0_push) begin
                    if (s0_ov) begin
                        s1_frame <= s0_of;
                        s1_real  <= 1'b1;
                        s1_push  <= 1'b1;
                        st       <= S_SUB1;
                    end else if (!in_real) begin
                        s1_frame <= {C_MID{1'b0}};
                        s1_real  <= 1'b0;      // the flush sub1 still needs
                        s1_push  <= 1'b1;
                        st       <= S_SUB1;
                    end else begin
                        st <= S_IDLE;          // still filling; nothing yet
                    end
                end
            S_SUB1:
                if (!s1dw_busy && !s1_push) begin
                    if (s1dw_ov) begin
                        y_lat   <= s1dw_of;
                        x_lat   <= xdly[0];    // the frame that entered 2*PAD ago
                        skip_iv <= 1'b1;
                        st      <= S_SKIP;
                    end else begin
                        st <= S_IDLE;
                    end
                end
            S_SKIP:
                // skip_acc[] is filled by the time skip goes idle
                if (!skip_busy && !skip_iv) begin
                    s1pw_iv <= 1'b1;
                    st      <= S_PW;
                end
            S_PW:
                if (!s1pw_busy && !s1pw_iv) begin
                    out_valid <= 1'b1;
                    st        <= S_IDLE;
                end
            default: st <= S_IDLE;
            endcase
        end
    end

`ifdef KWS_ASSERT
    always @(posedge clk) if (in_push && busy) begin
        $display("ASSERT %m: pushed while busy");
        $finish;
    end
    // the skip's accumulator must be for the same channel the pointwise is on,
    // or the residual pairs the wrong channels -- silent and plausible
    always @(posedge clk) if (s1pw_av && skip_busy) begin
        $display("ASSERT %m: skip still running while the add consumes it");
        $finish;
    end
`endif

endmodule

`default_nettype wire
