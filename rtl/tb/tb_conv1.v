// kws_conv1 against the golden vectors.
//
// Input  : <gen>/golden/input.hex     (the AFE output, 16 ch x 128)
// Expect : <gen>/golden/conv1_out.hex (128 ch x 64, packed +-1)
//
// Three things this layer does that no other one does, and each has its own way
// of failing quietly:
//
//   stride 2   -- output t is due after input frame 2t + PAD. One push either
//                 way shifts the whole feature map by a frame.
//   int8 x +-1 -- no multiplier; the weight is added or subtracted. Negating at
//                 eight bits would return -128 unchanged.
//   zero pad   -- a padded tap contributes 0, not -1, and only the six edge
//                 frames can tell the difference. A middle frame uses every
//                 tap and passes either way.
//
// The flush count is derived, not typed: the last output needs push
// STRIDE*(T_OUT-1) + PAD = 131, which is four past the last real frame.
//
// The slow one, alongside tb_tail: 22,528 cycles a frame.
//
// <gen> is whichever export run_tb.sh selected; the second
// argument picks it and defaults to xl_g12.
//
//   ./rtl/run_tb.sh conv1

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module tb_conv1;

    // The export decides how many clips the vectors hold (export/golden.py
    // --clips), so reading it from the generated header is the only way the
    // two cannot drift. Hardcoding 2 against a default of 8 checked a quarter
    // of the vectors and still printed ok.
    localparam integer CLIPS  = `KWS_GOLD_CLIPS;
    localparam integer C_IN   = `KWS_L0_CONV1_IN_CH;
    localparam integer C_OUT  = `KWS_L0_CONV1_OUT_CH;
    localparam integer K      = `KWS_L0_CONV1_KERNEL;
    localparam integer PAD    = `KWS_L0_CONV1_PADDING;
    localparam integer STRIDE = `KWS_L0_CONV1_STRIDE;
    localparam integer T_IN   = `KWS_T;
    localparam integer T_OUT  = (T_IN + 2*PAD - K) / STRIDE + 1;
    localparam integer P_LAST = STRIDE * (T_OUT - 1) + PAD;
    localparam integer FLUSH  = P_LAST - (T_IN - 1);
    localparam integer NWI    = (C_IN  + `KWS_WORD_BITS - 1) / `KWS_WORD_BITS;
    localparam integer NWO    = (C_OUT + `KWS_WORD_BITS - 1) / `KWS_WORD_BITS;

    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    reg              start = 1'b0, push = 1'b0, real_f = 1'b0;
    reg  [C_IN-1:0]  frame = {C_IN{1'b0}};
    wire             busy, ov;
    wire [C_OUT-1:0] of;

    kws_conv1 #(.C_IN(C_IN), .C_OUT(C_OUT), .K(K), .PAD(PAD), .STRIDE(STRIDE),
                .T_IN(T_IN), .W_BITS(8), .ACC_BITS(`KWS_L0_CONV1_ACC_BITS),
                .CO_BITS(7),
                .W_FILE(`KWS_ROM_CONV1_W),
                .T_FILE(`KWS_ROM_CONV1_T)) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .in_push(push), .in_real(real_f), .in_frame(frame),
        .busy(busy), .out_valid(ov), .out_frame(of));

    reg [`KWS_WORD_BITS-1:0] xin [0:CLIPS*T_IN*NWI-1];
    reg [`KWS_WORD_BITS-1:0] exp [0:CLIPS*T_OUT*NWO-1];
    reg [C_OUT-1:0]          got [0:T_OUT-1];
    integer                  got_n;

    always @(posedge clk) begin
        if (ov && got_n < T_OUT) begin
            got[got_n] <= of;
            got_n      <= got_n + 1;
        end
    end

    integer errors = 0, checked = 0, n, p, j, t;
    reg [C_OUT-1:0] want;

    initial begin
        $dumpfile("tb_conv1.vcd");
        $dumpvars(0, tb_conv1);

        $readmemh(`KWS_GOLD_INPUT, xin,
                  0, CLIPS * T_IN * NWI - 1);
        $readmemh(`KWS_GOLD_CONV1_OUT, exp,
                  0, CLIPS * T_OUT * NWO - 1);

        got_n = 0;
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        for (n = 0; n < CLIPS; n = n + 1) begin
            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;
            got_n = 0;

            for (p = 0; p < T_IN + FLUSH; p = p + 1) begin
                frame = {C_IN{1'b0}};
                if (p < T_IN)
                    for (j = 0; j < NWI; j = j + 1)
                        frame[j*`KWS_WORD_BITS +: `KWS_WORD_BITS] =
                            xin[(n * T_IN + p) * NWI + j];
                @(negedge clk);
                push = 1'b1; real_f = (p < T_IN);
                @(negedge clk);
                push = 1'b0;
                while (busy) @(negedge clk);
            end
            repeat (4) @(negedge clk);

            if (got_n !== T_OUT) begin
                $display("FAIL clip%0d: %0d output frames, expected %0d",
                         n, got_n, T_OUT);
                errors = errors + 1;
            end
            for (t = 0; t < T_OUT; t = t + 1) begin
                want = {C_OUT{1'b0}};
                for (j = 0; j < NWO; j = j + 1)
                    want[j*`KWS_WORD_BITS +: `KWS_WORD_BITS] =
                        exp[(n * T_OUT + t) * NWO + j];
                checked = checked + 1;
                if (got[t] !== want) begin
                    errors = errors + 1;
                    if (errors <= 5)
                        $display("FAIL clip%0d t=%0d\n  got  %h\n  want %h",
                                 n, t, got[t], want);
                end
            end
            $display("ok   clip%0d: %0d frames from %0d pushes (%0d flush)",
                     n, got_n, T_IN + FLUSH, FLUSH);
        end

        $display("\n%0d frames checked, %0d failures", checked, errors);
        $finish;
    end

    initial begin
        // 22,528 cycles a frame x 64 x 2 clips at 10ns is ~29 ms
        #300_000_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule

`default_nettype wire
