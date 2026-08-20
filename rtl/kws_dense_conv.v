// conv3 and conv4: a 1x1 conv with MULTI-BIT weights and multi-bit activations.
//
//     acc[o] = sum over i of  w[o][i] * x[i]
//
// WHY THIS IS NOT kws_pw_conv. Everything upstream is binary, so a dot product
// there is `2*popcount(XNOR) - N` -- no multiplier at all, and 32 terms settled
// per cycle because they are one bit each. Here neither side is one bit: conv3
// multiplies int8 weights by a 14-bit activation, so there is a real multiplier
// and terms arrive one per cycle. The two cannot share a datapath.
//
// It is still cheap enough to leave folded. conv3 is 128x128 and conv4 128x12,
// so the tail is 17,920 MAC cycles per frame and 1.15M per inference; at 10 Hz
// that is 11.5 MHz with a single multiplier, against a KC705 that runs ten
// times faster. Parallelism here would buy nothing.
//
// THE PORTS ARE kws_affine's, MIRRORED. Its output side is (valid, ch, value)
// and its input side is (valid, ch, accumulator), so this module takes the
// first and produces the second. The tail is then a chain with no glue:
//
//     conv2_pw acc -> affine -> dense(conv3) -> affine -> dense(conv4) -> affine
//
// ONE FRAME AT A TIME. The whole input frame must be resident before the sweep
// starts, because output channel o touches every input channel. That is C_IN
// activations of IN_BITS -- 1792 bits for conv3 -- which is small enough to
// hold in registers and lets the inner loop read an activation and a weight in
// the same cycle.
//
// THE WEIGHT ROM MUST BE DECLARED EXACTLY W_BITS WIDE. The .hex holds two's
// complement at that width, so a negative weight in a wider register would
// zero-extend: -5 written as `fb` reads as 251. The manifest carries
// `weight_bits` for exactly this reason.

`timescale 1ns/1ps
`default_nettype none

module kws_dense_conv #(
    parameter integer C_IN     = 128,
    parameter integer C_OUT    = 128,
    parameter integer IN_BITS  = 14,   // activation width, signed
    parameter integer W_BITS   = 8,    // manifest: roms[<layer>_w].weight_bits
    parameter integer ACC_BITS = 28,   // manifest: tail.sites[].acc_bits
    parameter integer CI_BITS  = 7,    // $clog2(C_IN),  see kws_affine on why
    parameter integer CO_BITS  = 7,    // $clog2(C_OUT)
    parameter         W_FILE   = ""    // C_OUT*C_IN words, [out_ch][in_ch]
) (
    input  wire                       clk,
    input  wire                       rst_n,

    // ---- load one activation per cycle (kws_affine's output side) ------- //
    input  wire                       in_valid,
    input  wire [CI_BITS-1:0]         in_ch,
    input  wire signed [IN_BITS-1:0]  in_val,

    input  wire                       start,     // frame is loaded, sweep it
    output wire                       busy,

    // ---- one accumulator per cycle (kws_affine's input side) ------------ //
    output reg                        acc_valid,
    output reg  [CO_BITS-1:0]         acc_ch,
    output reg  signed [ACC_BITS-1:0] acc_out
);

    localparam integer N_W    = C_OUT * C_IN;
    localparam integer WA_BITS = (N_W <= 2) ? 1 : $clog2(N_W);
    localparam integer PROD_BITS = IN_BITS + W_BITS;

    // ---- the frame, and the weights ------------------------------------- //
    reg signed [IN_BITS-1:0] act [0:C_IN-1];
    always @(posedge clk) if (in_valid) act[in_ch] <= in_val;

    reg signed [W_BITS-1:0] w [0:N_W-1];
    integer k;
    initial begin
        for (k = 0; k < N_W; k = k + 1) w[k] = {W_BITS{1'b0}};
        if (W_FILE != "") $readmemh(W_FILE, w, 0, N_W - 1);
    end

    // ---- stage A: address ------------------------------------------------ //
    // The weight ROM is far too big for distributed RAM (conv3 is 128 Kbit), so
    // it reads synchronously and the sweep is a two-stage pipeline: address
    // this cycle, multiply the next. The counters therefore run one term ahead
    // of the accumulator.
    reg                  run;
    reg [CI_BITS-1:0]    ci;
    reg [CO_BITS-1:0]    co;
    reg [WA_BITS-1:0]    wa;

    localparam [CI_BITS-1:0] CI_LAST = C_IN - 1;
    localparam [CO_BITS-1:0] CO_LAST = C_OUT - 1;

    wire ci_last = (ci == CI_LAST);
    wire co_last = (co == CO_LAST);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run <= 1'b0; ci <= {CI_BITS{1'b0}}; co <= {CO_BITS{1'b0}};
            wa  <= {WA_BITS{1'b0}};
        end else if (start) begin
            run <= 1'b1; ci <= {CI_BITS{1'b0}}; co <= {CO_BITS{1'b0}};
            wa  <= {WA_BITS{1'b0}};
        end else if (run) begin
            wa <= wa + {{(WA_BITS-1){1'b0}}, 1'b1};
            if (ci_last) begin
                ci <= {CI_BITS{1'b0}};
                if (co_last) run <= 1'b0;
                else co <= co + {{(CO_BITS-1){1'b0}}, 1'b1};
            end else begin
                ci <= ci + {{(CI_BITS-1){1'b0}}, 1'b1};
            end
        end
    end

    // ---- stage B: multiply and accumulate -------------------------------- //
    reg                       vB, lastB;
    reg [CO_BITS-1:0]         coB;
    reg signed [W_BITS-1:0]   wq;
    reg signed [IN_BITS-1:0]  xq;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vB <= 1'b0; lastB <= 1'b0; coB <= {CO_BITS{1'b0}};
            wq <= {W_BITS{1'b0}}; xq <= {IN_BITS{1'b0}};
        end else begin
            vB <= run;
            if (run) begin
                wq    <= w[wa];
                xq    <= act[ci];
                lastB <= ci_last;
                coB   <= co;
            end
        end
    end

    wire signed [PROD_BITS-1:0] prod_n = wq * xq;
    wire signed [ACC_BITS-1:0]  prod =
        {{(ACC_BITS-PROD_BITS){prod_n[PROD_BITS-1]}}, prod_n};

    reg signed [ACC_BITS-1:0] acc;
    wire signed [ACC_BITS-1:0] sum = acc + prod;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= {ACC_BITS{1'b0}}; acc_valid <= 1'b0;
            acc_ch <= {CO_BITS{1'b0}}; acc_out <= {ACC_BITS{1'b0}};
        end else begin
            acc_valid <= 1'b0;
            if (start) acc <= {ACC_BITS{1'b0}};
            else if (vB) begin
                if (lastB) begin
                    // emit the completed channel and clear for the next one in
                    // the same cycle -- the next term must land on zero, not on
                    // the sum that was just published
                    acc_out   <= sum;
                    acc_ch    <= coB;
                    acc_valid <= 1'b1;
                    acc       <= {ACC_BITS{1'b0}};
                end else begin
                    acc <= sum;
                end
            end
        end
    end

    assign busy = run | vB;

`ifdef KWS_ASSERT
    initial if (CI_BITS != ((C_IN <= 2) ? 1 : $clog2(C_IN)) ||
                CO_BITS != ((C_OUT <= 2) ? 1 : $clog2(C_OUT))) begin
        $display("ASSERT %m: CI_BITS/CO_BITS do not match C_IN/C_OUT");
        $finish;
    end
    // Loading a frame while the sweep is running changes the operand under it,
    // and the result stays plausible.
    always @(posedge clk) if (in_valid && busy) begin
        $display("ASSERT %m: frame written while the sweep is running");
        $finish;
    end
    always @(posedge clk) if (start && busy) begin
        $display("ASSERT %m: restarted while still busy");
        $finish;
    end
`endif

endmodule

`default_nettype wire
