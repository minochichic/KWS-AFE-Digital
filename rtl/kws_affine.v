// The tail's epilogue: an integer accumulator becomes a fixed-point activation.
//
//     y = (A[ch] * acc + B[ch] + half) >>> SHIFT      -> relu? -> saturate
//
// WHY THIS EXISTS AT ALL. The three binary blocks end in sign(), so their
// BatchNorm collapses into one integer compare and nothing survives to compute
// (docs/diagrams/30_bn_vanishes.svg). The tail ends in relu, which asks "how
// much" rather than "which side of zero", and a value cannot be pre-solved the
// way an inequality can. So the BN stays -- as arithmetic, in this module.
//
// ONE MODULE FOR ALL THREE SITES. conv2_pw, conv3 and conv4 differ only in
// their constants: what multiplies the accumulator is a BN gain times a binary
// alpha, or times an int8 scale, or a bare power of two, but by the time
// export/tailbuild.py is done all three are "one integer per output channel"
// (docs/diagrams/31_fold_arithmetic.svg). Three instances, three ROMs, one
// piece of RTL.
//
// THE `+ half` IS ROUNDING, NOT A FUDGE FACTOR. `>>>` truncates toward minus
// infinity, so it always rounds down and loses half an LSB on average. Adding
// half the divisor first turns that truncate into a round-to-nearest -- the
// same trick as adding 5 before dividing by 10. It costs one constant add,
// where real rounding logic would cost a bit test.
//
// EVERY WIDTH COMES FROM THE MANIFEST. GAIN_BITS/BIAS_BITS/SHIFT/OUT_BITS all
// move when the network is retrained, because the shift is chosen from the
// trained BN gains. None of them may be written into this file (CLAUDE.md 5).

`timescale 1ns/1ps
`default_nettype none

module kws_affine #(
    parameter integer C         = 128,  // output channels
    parameter integer ACC_BITS  = 8,    // incoming accumulator width, signed
    parameter integer GAIN_BITS = 22,   // manifest: tail.sites[].gain_bits
    parameter integer BIAS_BITS = 28,   // manifest: tail.sites[].bias_bits
    parameter integer SHIFT     = 18,
    parameter integer OUT_BITS  = 14,   // 8.6 -> 14
    parameter integer RELU      = 1,
    // $clog2(C). A localparam cannot be used in a port declaration in
    // Verilog-2005, and the ports need this width, so the caller passes it --
    // the same workaround kws_pw_conv uses for CO_BITS_P.
    parameter integer CH_BITS   = 7,
    parameter         ROM_FILE  = ""    // C gains, then C offsets, int32 each
) (
    input  wire                       clk,
    input  wire                       rst_n,

    input  wire                       in_valid,
    input  wire [CH_BITS-1:0]         in_ch,
    input  wire signed [ACC_BITS-1:0] in_acc,

    output reg                        out_valid,
    output reg  [CH_BITS-1:0]         out_ch,
    output reg  signed [OUT_BITS-1:0] out_val
);

    // A*acc, then room for the bias add and the half add on top. The two extra
    // bits are not decoration: without them a large bias on a channel already
    // near its product's extreme wraps, and a wrapped sign looks like a
    // legitimate large negative activation that relu then zeroes.
    // ROM address. NOT {sel, ch}: that packs the offsets at C rounded up to a
    // power of two, which is only the same place as C when C already is one.
    // conv2_pw and conv3 have 128 channels and would never have noticed;
    // conv4 has 12, and {1'b1, ch} would have read entries 16..27 of a 24-entry
    // array -- past the end for the last four classes.
    localparam integer AW = (2 * C <= 2) ? 1 : $clog2(2 * C);
    localparam [AW-1:0] C_A = C[AW-1:0];

    localparam integer PROD_BITS = GAIN_BITS + ACC_BITS;
    localparam integer SUM_BITS  = ((PROD_BITS > BIAS_BITS) ? PROD_BITS
                                                            : BIAS_BITS) + 2;

    // ---- constants ------------------------------------------------------ //
    reg [31:0] rom [0:2*C-1];
    integer i;
    initial begin
        // cleared first: $readmemh leaves untouched entries as X, and a stale
        // or short ROM would then read as a plausible-looking constant
        for (i = 0; i < 2 * C; i = i + 1) rom[i] = 32'h0;
        if (ROM_FILE != "") $readmemh(ROM_FILE, rom, 0, 2 * C - 1);
    end

    // ---- stage 1: fetch the channel's two constants --------------------- //
    reg                        v1;
    reg [CH_BITS-1:0]          ch1;
    reg signed [ACC_BITS-1:0]  acc1;
    reg signed [31:0]          a1, b1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0; ch1 <= {CH_BITS{1'b0}}; acc1 <= {ACC_BITS{1'b0}};
            a1 <= 32'sd0; b1 <= 32'sd0;
        end else begin
            v1   <= in_valid;
            ch1  <= in_ch;
            acc1 <= in_acc;
            a1   <= rom[{{(AW-CH_BITS){1'b0}}, in_ch}];
            b1   <= rom[{{(AW-CH_BITS){1'b0}}, in_ch} + C_A];
        end
    end

    // ---- stage 2: multiply, add the bias, add half ---------------------- //
    wire signed [SUM_BITS-1:0] prod =
        $signed({{(SUM_BITS-GAIN_BITS){a1[31]}}, a1[GAIN_BITS-1:0]}) *
        $signed({{(SUM_BITS-ACC_BITS){acc1[ACC_BITS-1]}}, acc1});
    wire signed [SUM_BITS-1:0] bias_x =
        {{(SUM_BITS-BIAS_BITS){b1[31]}}, b1[BIAS_BITS-1:0]};
    // SHIFT is never 0 in this design (the fold always needs headroom), but a
    // zero-width replication is a syntax error rather than a warning, so the
    // degenerate case is written out instead of assumed away.
    wire signed [SUM_BITS-1:0] half =
        (SHIFT == 0) ? {SUM_BITS{1'b0}}
                     : {{(SUM_BITS-1){1'b0}}, 1'b1} << (SHIFT - 1);

    reg                        v2;
    reg [CH_BITS-1:0]          ch2;
    reg signed [SUM_BITS-1:0]  sum2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v2 <= 1'b0; ch2 <= {CH_BITS{1'b0}}; sum2 <= {SUM_BITS{1'b0}};
        end else begin
            v2   <= v1;
            ch2  <= ch1;
            sum2 <= prod + bias_x + half;
        end
    end

    // ---- stage 3: shift, relu, saturate --------------------------------- //
    wire signed [SUM_BITS-1:0] shifted = sum2 >>> SHIFT;
    localparam signed [SUM_BITS-1:0] OUT_HI =
        {{(SUM_BITS-OUT_BITS){1'b0}}, 1'b0, {(OUT_BITS-1){1'b1}}};
    localparam signed [SUM_BITS-1:0] OUT_LO =
        {{(SUM_BITS-OUT_BITS){1'b1}}, 1'b1, {(OUT_BITS-1){1'b0}}};

    // relu before the clamp. Order matters at the top end only: a positive
    // overflow must still saturate after relu has passed it through. At the
    // bottom the two orders agree, since anything below OUT_LO is negative and
    // relu zeroes it either way.
    wire signed [SUM_BITS-1:0] relud =
        (RELU != 0 && shifted[SUM_BITS-1]) ? {SUM_BITS{1'b0}} : shifted;

    // The same two limits at the output's own width. Comparing needs them wide
    // (relud is wide); the RESULT is OUT_BITS by construction, and building it
    // wide and slicing later leaves SUM_BITS-OUT_BITS bits that are pure sign
    // extension -- which is exactly what lint reported as unused. Narrowing
    // here instead makes the truncation explicit and guarded: `relud` is only
    // sliced on the branch where the comparison has already proved it fits.
    localparam signed [OUT_BITS-1:0] SAT_HI = {1'b0, {(OUT_BITS-1){1'b1}}};
    localparam signed [OUT_BITS-1:0] SAT_LO = {1'b1, {(OUT_BITS-1){1'b0}}};
    wire signed [OUT_BITS-1:0] sat =
        (relud > OUT_HI) ? SAT_HI :
        ((relud < OUT_LO) ? SAT_LO : $signed(relud[OUT_BITS-1:0]));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0; out_ch <= {CH_BITS{1'b0}};
            out_val   <= {OUT_BITS{1'b0}};
        end else begin
            out_valid <= v2;
            out_ch    <= ch2;
            out_val   <= sat;
        end
    end

`ifdef KWS_ASSERT
    // CH_BITS is passed in, so it can disagree with C. Every address in this
    // module is built from it, and one bit short silently aliases the top half
    // of the channels onto the bottom half.
    initial if (CH_BITS != ((C <= 2) ? 1 : $clog2(C))) begin
        $display("ASSERT %m: CH_BITS=%0d does not match C=%0d", CH_BITS, C);
        $finish;
    end
    // AW is always CH_BITS+1 for any C >= 2, and the zero-extension above is
    // written assuming that. A zero-width replication is a syntax error rather
    // than a warning, so the assumption is checked instead of trusted.
    initial if (AW <= CH_BITS) begin
        $display("ASSERT %m: AW=%0d must exceed CH_BITS=%0d", AW, CH_BITS);
        $finish;
    end

    // The ROM words are 32 bits but the datapath is sized from GAIN_BITS and
    // BIAS_BITS. If a constant does not actually fit the width the manifest
    // claims, the narrowing above silently drops its top bits.
    //
    // Tested by shifting rather than by sign-extending. A 32-bit signed v fits
    // in W bits exactly when `v >>> (W-1)` is all zeros or all ones -- and that
    // form has no replication, so it survives W = 32, where writing
    // `{{(32-W){...}}, v[W-1:0]}` is a zero-width replication and a syntax
    // error rather than a warning. conv3's BIAS_BITS is exactly 32.
    always @(posedge clk) if (v1) begin
        if (($signed(a1) >>> (GAIN_BITS - 1)) != 0 &&
            ($signed(a1) >>> (GAIN_BITS - 1)) != -1) begin
            $display("ASSERT %m: gain %0d does not fit GAIN_BITS=%0d",
                     $signed(a1), GAIN_BITS);
            $finish;
        end
        if (($signed(b1) >>> (BIAS_BITS - 1)) != 0 &&
            ($signed(b1) >>> (BIAS_BITS - 1)) != -1) begin
            $display("ASSERT %m: offset %0d does not fit BIAS_BITS=%0d",
                     $signed(b1), BIAS_BITS);
            $finish;
        end
    end
`endif

endmodule

`default_nettype wire
