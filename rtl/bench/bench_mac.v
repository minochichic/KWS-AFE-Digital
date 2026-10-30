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
// KINDs 0-7 feed the engine straight from the stimulus: they price the
// arithmetic alone. KINDs 8-12 add the OPERAND MEMORY the way kws_pw_conv
// feeds its MAC -- a weight ROM read one word a cycle, and a 128-channel
// activation frame latched once and read one word a cycle -- because a term's
// weight has to come from somewhere, and an int8 term reads 8x the bits:
//
//   KIND 8  bin32x8_mem      8 x (kws_bin_mac + 256x32 weight ROM in LUTs
//                            + 128-bit frame register)            256 terms/cycle
//   KIND 9  int8x32_dsp_mem  KIND 6 + 256x256 weight ROM in LUTs
//                            + 128x8-bit frame register            32 terms/cycle
//   KIND 10 int8x32_dsp_bram KIND 9 with the weight ROM in block RAM
//                            (synchronous read, one pipeline stage)
//   KIND 11 int8pm1x32_mem   KIND 3 + 256x256 weight ROM in LUTs
//                            + 128-bit (+-1) frame register        32 terms/cycle
//   KIND 12 bin32_mem        one copy of KIND 8, for its LUT/FF    32 terms/cycle
//
// Same ROM shape as the design: pw 128 -> 64 is 64 x 4 words = 256 words, read
// asynchronously from LUTs (kws_pw_conv w_rom). The frame register is reloaded
// every 4 vectors (260 cycles) -- pw 128 -> 64 latches a frame per 64 output
// channels x 4 words. Contents are fixed pseudo-random constants from mix().
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
//
// mon_* are taps for tb_bench_mac: copy 0 / lane 0 at the engine's input and
// its accumulator, whatever the KIND. Nothing reads them in hardware.

`timescale 1ns/1ps
`default_nettype none

module bench_mac #(
    parameter integer KIND = 0,
    parameter integer ACC  = 24        // int8 lanes: compared mod 2^24 in the tb
) (
    input  wire clk,
    input  wire rst_n,
    output reg  out
);

    localparam integer LANES = (KIND == 2) ? 1 : 32;   // g_int only
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
    wire [511:0] pool     = {wgt_bits, act_bits};

    // ROM contents: 256 pseudo-random bits per (word, salt), a constant
    function [255:0] mix(input integer idx, input integer salt);
        reg [63:0] x;
        integer c;
        begin
            for (c = 0; c < 4; c = c + 1) begin
                x = ({32'd0, idx} << 2) + c;
                x = x ^ ({32'd0, salt} << 40) ^ 64'h9E3779B97F4A7C15;
                mix[64*c +: 64] = xs64(xs64(xs64(x)));
            end
        end
    endfunction

    // ---- vector rhythm: start, then VEC words --------------------------------- //
    reg [6:0] pos;                      // 0 = start, 1..VEC = data
    wire      start    = (pos == 7'd0);
    wire      in_valid = !start;
    always @(posedge clk or negedge rst_n)
        if (!rst_n)              pos <= 7'd0;
        else if (pos == VEC[6:0]) pos <= 7'd0;
        else                     pos <= pos + 7'd1;

    reg [1:0] vcnt;                     // frame register reloads when 0
    always @(posedge clk or negedge rst_n)
        if (!rst_n)               vcnt <= 2'd0;
        else if (pos == VEC[6:0]) vcnt <= vcnt + 2'd1;

    // weight ROM address for the memory kinds: one word per term-cycle
    reg [7:0] addr;
    always @(posedge clk or negedge rst_n)
        if (!rst_n)        addr <= 8'd0;
        else if (in_valid) addr <= addr + 8'd1;

    // ---- the engines ---------------------------------------------------------- //
    wire [ACC-1:0] acc_x;               // XOR of every accumulator, one ACC word
    wire [255:0]   mon_act, mon_wgt;
    wire           mon_start, mon_valid;
    wire [ACC-1:0] mon_acc;

    generate
    if (KIND == 5) begin : g_stim
        assign acc_x     = {ACC{1'b0}};
        assign mon_act   = act_bits;
        assign mon_wgt   = wgt_bits;
        assign mon_start = start;
        assign mon_valid = in_valid;
        assign mon_acc   = {ACC{1'b0}};
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
        assign acc_x     = {{(ACC-16){1'b0}}, x8};
        assign mon_act   = act_bits;
        assign mon_wgt   = wgt_bits;
        assign mon_start = start;
        assign mon_valid = in_valid;
        assign mon_acc   = {{(ACC-16){accs[0][15]}}, accs[0]};
    end else if (KIND == 0) begin : g_bin
        wire signed [15:0] acc;
        wire               ov;
        kws_bin_mac #(.WORD_BITS(32), .ACC_BITS(16), .CNT_BITS(16)) u_mac (
            .clk(clk), .rst_n(rst_n), .start(start),
            .n_valid(16'd2048), .in_valid(in_valid),
            .act(act_bits[31:0]), .wgt(wgt_bits[31:0]),
            .out_valid(ov), .acc(acc));
        assign acc_x     = {{(ACC-16){acc[15]}}, acc};
        assign mon_act   = act_bits;
        assign mon_wgt   = wgt_bits;
        assign mon_start = start;
        assign mon_valid = in_valid;
        assign mon_acc   = {{(ACC-16){acc[15]}}, acc};
    end else if (KIND == 8 || KIND == 12) begin : g_bmem
        localparam integer NC = (KIND == 8) ? 8 : 1;
        wire [15:0] accs [0:NC-1];
        wire [31:0] aws  [0:NC-1];
        wire [31:0] wws  [0:NC-1];
        genvar m;
        for (m = 0; m < NC; m = m + 1) begin : g_m
            // the 128-channel frame, latched once, read a word a cycle -- as
            // kws_pw_conv's `act`. Each copy loads at its own pos so no two
            // copies' registers are equivalent (synthesis would merge them).
            reg [127:0] frame;
            always @(posedge clk or negedge rst_n)
                if (!rst_n)                          frame <= 128'd0;
                else if (vcnt == 2'd0 && pos == m)   frame <= pool[(128 * m) % 512 +: 128];
            wire [31:0] aw = frame[32 * pos[1:0] +: 32];

            // the weight ROM: 256 x 32, asynchronous, in LUTs -- kws_pw_conv w_rom
            (* rom_style = "distributed" *) reg [31:0] rom [0:255];
            integer i;
            initial for (i = 0; i < 256; i = i + 1) rom[i] = mix(i, m);
            wire [31:0] ww = rom[addr];

            wire signed [15:0] acc;
            wire               ov;
            kws_bin_mac #(.WORD_BITS(32), .ACC_BITS(16), .CNT_BITS(16)) u_mac (
                .clk(clk), .rst_n(rst_n), .start(start),
                .n_valid(16'd2048), .in_valid(in_valid),
                .act(aw), .wgt(ww), .out_valid(ov), .acc(acc));
            assign accs[m] = acc;
            assign aws[m]  = aw;
            assign wws[m]  = ww;
        end
        reg [15:0] x;
        integer l;
        always @* begin
            x = 16'd0;
            for (l = 0; l < NC; l = l + 1)
                x = x ^ accs[l];
        end
        assign acc_x     = {{(ACC-16){1'b0}}, x};
        assign mon_act   = {224'd0, aws[0]};
        assign mon_wgt   = {224'd0, wws[0]};
        assign mon_start = start;
        assign mon_valid = in_valid;
        assign mon_acc   = {{(ACC-16){accs[0][15]}}, accs[0]};
    end else if (KIND >= 9 && KIND <= 11) begin : g_imem
        localparam integer AB = (KIND == 11) ? 1 : 8;    // activation bits/lane
        localparam integer AW = 32 * AB;                  // activation word
        localparam integer BRAM = (KIND == 10);

        // the 128-channel frame, 4 words, word j loaded at pos j
        reg [4*AW-1:0] frame;
        always @(posedge clk or negedge rst_n)
            if (!rst_n)
                frame <= {(4*AW){1'b0}};
            else if (vcnt == 2'd0 && pos < 7'd4)
                frame[AW * pos[1:0] +: AW] <= pool[(AB == 8 ? 256 * pos[0] : 32 * pos[1:0]) +: AW];
        wire [AW-1:0] aw = frame[AW * pos[1:0] +: AW];

        // the weight ROM: 256 x 256 (32 int8 weights a word)
        wire [255:0] ww;
        if (BRAM) begin : g_bram
            (* rom_style = "block" *) reg [255:0] rom [0:255];
            reg [255:0] q;
            integer i;
            initial for (i = 0; i < 256; i = i + 1) rom[i] = mix(i, 99);
            always @(posedge clk) q <= rom[addr];
            assign ww = q;
        end else begin : g_lut
            (* rom_style = "distributed" *) reg [255:0] rom [0:255];
            integer i;
            initial for (i = 0; i < 256; i = i + 1) rom[i] = mix(i, 99);
            assign ww = rom[addr];
        end

        // block RAM reads a cycle late: delay the rest to meet it
        reg          d_start, d_valid;
        reg [AW-1:0] d_act;
        always @(posedge clk or negedge rst_n)
            if (!rst_n) begin
                d_start <= 1'b0; d_valid <= 1'b0; d_act <= {AW{1'b0}};
            end else begin
                d_start <= start; d_valid <= in_valid; d_act <= aw;
            end
        wire          e_start = BRAM ? d_start : start;
        wire          e_valid = BRAM ? d_valid : in_valid;
        wire [AW-1:0] e_act   = BRAM ? d_act   : aw;

        reg  signed [ACC-1:0] acc [0:31];
        reg         [ACC-1:0] x;
        integer l;
        always @* begin
            x = {ACC{1'b0}};
            for (l = 0; l < 32; l = l + 1)
                x = x ^ acc[l];
        end
        assign acc_x = x;

        genvar k;
        for (k = 0; k < 32; k = k + 1) begin : g_lane
            wire signed [7:0] w = ww[8*k +: 8];
            if (KIND == 11) begin : g_pm1
                wire signed [ACC-1:0] t = e_act[k] ? {{(ACC-8){w[7]}}, w}
                                                   : -{{(ACC-8){w[7]}}, w};
                always @(posedge clk or negedge rst_n)
                    if (!rst_n)       acc[k] <= {ACC{1'b0}};
                    else if (e_start) acc[k] <= {ACC{1'b0}};
                    else if (e_valid) acc[k] <= acc[k] + t;
            end else begin : g_dsp
                wire signed [7:0] a = e_act[8*k +: 8];
                (* use_dsp = "yes" *) reg signed [ACC-1:0] acc_d;
                (* use_dsp = "yes" *) wire signed [15:0] p = a * w;
                always @(posedge clk or negedge rst_n)
                    if (!rst_n)       acc_d <= {ACC{1'b0}};
                    else if (e_start) acc_d <= {ACC{1'b0}};
                    else if (e_valid) acc_d <= acc_d + {{(ACC-16){p[15]}}, p};
                always @* acc[k] = acc_d;
            end
        end
        assign mon_act   = {{(256-AW){1'b0}}, e_act};
        assign mon_wgt   = ww;
        assign mon_start = e_start;
        assign mon_valid = e_valid;
        assign mon_acc   = acc[0];
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
        assign mon_act   = act_bits;
        assign mon_wgt   = wgt_bits;
        assign mon_start = start;
        assign mon_valid = in_valid;
        assign mon_acc   = acc[0];
    end
    endgenerate

    always @(posedge clk or negedge rst_n)
        if (!rst_n) out <= 1'b0;
        else        out <= ^acc_x ^ ^act_bits ^ ^wgt_bits;

endmodule

`default_nettype wire
