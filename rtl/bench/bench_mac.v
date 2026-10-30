// MAC microbenchmark: what one binary MAC costs against int8 MACs that do the
// same work, on the same chip (docs/conv2_int8_compare.md, option A).
//
//   KIND 0  bin32        kws_bin_mac, the design's own engine     32 terms/cycle
//   KIND 1  int8x32      32 lanes of int8 x int8 + accumulate     32 terms/cycle
//   KIND 2  int8x1       one int8 x int8 lane                      1 term/cycle
//   KIND 3  int8pm1x32   32 lanes of int8 x (+-1): add or subtract 32 terms/cycle
//                        (conv1's regime -- no multiplier)
//   KIND 4  int8x32_lut  KIND 1 with use_dsp = "no": multipliers in LUTs
//   KIND 5  stim         no engine at all -- the stimulus and the output XOR
//                        alone, to subtract from the others
//   KIND 6  int8x32_dsp  KIND 1 with use_dsp = "yes". Vivado maps an 8x8
//                        multiply to LUTs by default (first run, 2026-10-30:
//                        KIND 1 == KIND 4, 0 DSP); a real int8 design would
//                        use the 140 DSP48s, so this is the fair int8 point
//   KIND 7  bin32x8      8 x kws_bin_mac, 256 terms/cycle. One bin32 draws
//                        ~3 mW, and report_power prints 1 mW steps -- +-17%.
//                        Eight of them put the engine in the tens of mW
//
// The network's op counts are already known (conv2_int8_compare.md 4, 9.2);
// what is missing is what one term of each kind costs in LUT/FF/DSP and power.
// rtl/bench/run_mac_bench.tcl builds every KIND and tabulates that.
//
// STIMULUS IS GENERATED INSIDE: eight xorshift64 generators feed activations
// and weights, so no input pins and no I/O power. Random data is a high-
// activity input -- the same for every KIND, so the comparison holds even if
// the absolute power is pessimistic.
//
// EVERY ACCUMULATOR REACHES A PIN through an XOR reduction into one register:
// otherwise synthesis removes logic whose result nobody reads, and the
// cheapest design "wins" by vanishing. All 512 stimulus bits are XORed in as
// well, so every KIND carries the SAME stimulus logic (a KIND that read only
// 64 bits would otherwise have the other generators pruned) and KIND 5
// measures exactly that overhead.
//
// Vectors are 64 words long (2048 terms for 32-wide kinds), then every
// accumulator restarts -- the folded engine's rhythm, one output per vector.

`timescale 1ns/1ps
`default_nettype none

module bench_mac #(
    parameter integer KIND = 0,
    parameter integer ACC  = 24        // int8 lanes: 2048 terms x 127 x 127 < 2^25
) (
    input  wire clk,
    input  wire rst_n,
    output reg  out
);

    localparam integer LANES = (KIND == 2) ? 1 : (KIND == 5 || KIND == 7) ? 0 : 32;
    localparam integer VEC   = 64;      // words per vector

    // ---- stimulus: 8 x xorshift64 = 512 fresh bits a cycle ------------------ //
    reg [63:0] s [0:7];
    integer g;
    function [63:0] xs64(input [63:0] x);
        reg [63:0] y;
        begin
            y = x ^ (x << 13);
            y = y ^ (y >> 7);
            xs64 = y ^ (y << 17);
        end
    endfunction
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (g = 0; g < 8; g = g + 1)
                s[g] <= 64'h9E3779B97F4A7C15 ^ ({56'd0, g[7:0]} << (g * 7));
        end else begin
            for (g = 0; g < 8; g = g + 1)
                s[g] <= xs64(s[g]);
        end
    end
    wire [255:0] act_bits = {s[3], s[2], s[1], s[0]};
    wire [255:0] wgt_bits = {s[7], s[6], s[5], s[4]};

    // ---- vector rhythm: start, then VEC words --------------------------------- //
    reg [6:0] pos;                      // 0 = start, 1..VEC = data
    wire      start    = (pos == 7'd0);
    wire      in_valid = !start;
    always @(posedge clk or negedge rst_n)
        if (!rst_n)              pos <= 7'd0;
        else if (pos == VEC[6:0]) pos <= 7'd0;
        else                     pos <= pos + 7'd1;

    // ---- the engines ---------------------------------------------------------- //
    wire [ACC-1:0] acc_x;               // XOR of every accumulator, one ACC word

    generate
    if (KIND == 5) begin : g_stim
        assign acc_x = {ACC{1'b0}};
    end else if (KIND == 7) begin : g_bin8
        wire [15:0] accs [0:7];
        genvar m;
        for (m = 0; m < 8; m = m + 1) begin : g_m
            wire signed [15:0] acc;
            wire               ov;
            kws_bin_mac #(.WORD_BITS(32), .ACC_BITS(16), .CNT_BITS(16)) u_mac (
                .clk(clk), .rst_n(rst_n), .start(start),
                .n_valid(16'd2048), .in_valid(in_valid),
                .act(act_bits[32*m +: 32]), .wgt(wgt_bits[32*m +: 32]),
                .out_valid(ov), .acc(acc));
            assign accs[m] = acc;
        end
        wire [15:0] x8 = accs[0] ^ accs[1] ^ accs[2] ^ accs[3]
                       ^ accs[4] ^ accs[5] ^ accs[6] ^ accs[7];
        assign acc_x = {{(ACC-16){1'b0}}, x8};
    end else if (KIND == 0) begin : g_bin
        wire signed [15:0] acc;
        wire               ov;
        kws_bin_mac #(.WORD_BITS(32), .ACC_BITS(16), .CNT_BITS(16)) u_mac (
            .clk(clk), .rst_n(rst_n), .start(start),
            .n_valid(16'd2048), .in_valid(in_valid),
            .act(act_bits[31:0]), .wgt(wgt_bits[31:0]),
            .out_valid(ov), .acc(acc));
        assign acc_x = {{(ACC-16){acc[15]}}, acc};
    end else begin : g_int
        reg  signed [ACC-1:0] acc [0:LANES-1];
        reg         [ACC-1:0] x;
        integer l;
        always @* begin
            x = {ACC{1'b0}};
            for (l = 0; l < LANES; l = l + 1)
                x = x ^ acc[l];
        end
        assign acc_x = x;

        genvar k;
        for (k = 0; k < LANES; k = k + 1) begin : g_lane
            wire signed [7:0] a = act_bits[8*k +: 8];
            wire signed [7:0] w = wgt_bits[8*k +: 8];
            if (KIND == 3) begin : g_pm1
                // conv1: the activation is +-1 (one bit), so the weight is
                // added or subtracted -- no multiplier
                wire signed [ACC-1:0] t = a[0] ? {{(ACC-8){w[7]}}, w}
                                               : -{{(ACC-8){w[7]}}, w};
                always @(posedge clk or negedge rst_n)
                    if (!rst_n)        acc[k] <= {ACC{1'b0}};
                    else if (start)    acc[k] <= {ACC{1'b0}};
                    else if (in_valid) acc[k] <= acc[k] + t;
            end else if (KIND == 6) begin : g_dsp
                (* use_dsp = "yes" *) reg signed [ACC-1:0] acc_d;
                (* use_dsp = "yes" *) wire signed [15:0] p = a * w;
                always @(posedge clk or negedge rst_n)
                    if (!rst_n)        acc_d <= {ACC{1'b0}};
                    else if (start)    acc_d <= {ACC{1'b0}};
                    else if (in_valid) acc_d <= acc_d + {{(ACC-16){p[15]}}, p};
                always @* acc[k] = acc_d;
            end else if (KIND == 4) begin : g_lut
                (* use_dsp = "no" *) wire signed [15:0] p = a * w;
                always @(posedge clk or negedge rst_n)
                    if (!rst_n)        acc[k] <= {ACC{1'b0}};
                    else if (start)    acc[k] <= {ACC{1'b0}};
                    else if (in_valid) acc[k] <= acc[k] + {{(ACC-16){p[15]}}, p};
            end else begin : g_mul
                wire signed [15:0] p = a * w;
                always @(posedge clk or negedge rst_n)
                    if (!rst_n)        acc[k] <= {ACC{1'b0}};
                    else if (start)    acc[k] <= {ACC{1'b0}};
                    else if (in_valid) acc[k] <= acc[k] + {{(ACC-16){p[15]}}, p};
            end
        end
    end
    endgenerate

    always @(posedge clk or negedge rst_n)
        if (!rst_n) out <= 1'b0;
        else        out <= ^acc_x ^ ^act_bits ^ ^wgt_bits;

endmodule

`default_nettype wire
