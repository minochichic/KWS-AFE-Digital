// kws_affine against the golden vectors, for all three tail sites.
//
// Input  : rtl/gen/xl_g12/golden/<site>_acc.hex   (integer accumulator)
// Expect : rtl/gen/xl_g12/golden/<site>_out.hex   (fixed-point activation)
//
// Three instances rather than one, because the whole claim of kws_affine is
// that the sites differ ONLY in their constants -- conv2_pw's gain comes from a
// binary alpha, conv3's from an int8 scale, conv4's is a bare power of two, and
// by export time all three are just numbers in a ROM. If that claim is wrong,
// one instance fails and the others do not.
//
// EVERY WIDTH COMES FROM parameters.vh. Writing them here would defeat the
// point: the shift is chosen from the trained BN gains and moves on every
// retrain, and a testbench holding a stale shift passes against a stale ROM.
//
// These .hex files are NOT bit-packed. A thresholded layer's `_out.hex` holds
// one word per FRAME with channel c in bit c; a tail layer's holds one word per
// VALUE, because the values are multi-bit and there is nothing to pack.
// golden.json says which, in its `packed` field.
//
//   ./rtl/run_tb.sh affine

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/xl_g12/parameters.vh"

module tb_affine;

    localparam integer CLIPS = 2;
    localparam integer T     = 64;
    localparam integer STEP  = 4;      // sample every 4th frame; 128 ch is plenty

    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    // ---- one driver bus, three consumers -------------------------------- //
    reg               iv = 1'b0;
    reg  [6:0]        ich = 7'd0;
    reg signed [31:0] iacc = 32'sd0;

    wire              ov2, ov3, ov4;
    wire [6:0]        och2, och3;
    wire [3:0]        och4;
    wire signed [`KWS_CONV2_PW_OUT_BITS-1:0] oval2;
    wire signed [`KWS_CONV3_OUT_BITS-1:0]    oval3;
    wire signed [`KWS_CONV4_OUT_BITS-1:0]    oval4;

    kws_affine #(.C(`KWS_CONV2_PW_N_OUT),
                 .ACC_BITS(`KWS_CONV2_PW_ACC_BITS),
                 .GAIN_BITS(`KWS_CONV2_PW_GAIN_BITS),
                 .BIAS_BITS(`KWS_CONV2_PW_BIAS_BITS),
                 .SHIFT(`KWS_CONV2_PW_SHIFT),
                 .OUT_BITS(`KWS_CONV2_PW_OUT_BITS),
                 .RELU(`KWS_CONV2_PW_RELU), .CH_BITS(7),
                 .ROM_FILE("rtl/gen/xl_g12/conv2_pw_bn.hex")) u2 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(iv), .in_ch(ich),
        .in_acc(iacc[`KWS_CONV2_PW_ACC_BITS-1:0]),
        .out_valid(ov2), .out_ch(och2), .out_val(oval2));

    kws_affine #(.C(`KWS_CONV3_N_OUT),
                 .ACC_BITS(`KWS_CONV3_ACC_BITS),
                 .GAIN_BITS(`KWS_CONV3_GAIN_BITS),
                 .BIAS_BITS(`KWS_CONV3_BIAS_BITS),
                 .SHIFT(`KWS_CONV3_SHIFT),
                 .OUT_BITS(`KWS_CONV3_OUT_BITS),
                 .RELU(`KWS_CONV3_RELU), .CH_BITS(7),
                 .ROM_FILE("rtl/gen/xl_g12/conv3_bn.hex")) u3 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(iv), .in_ch(ich), .in_acc(iacc[`KWS_CONV3_ACC_BITS-1:0]),
        .out_valid(ov3), .out_ch(och3), .out_val(oval3));

    kws_affine #(.C(`KWS_CONV4_N_OUT),
                 .ACC_BITS(`KWS_CONV4_ACC_BITS),
                 .GAIN_BITS(`KWS_CONV4_GAIN_BITS),
                 .BIAS_BITS(`KWS_CONV4_BIAS_BITS),
                 .SHIFT(`KWS_CONV4_SHIFT),
                 .OUT_BITS(`KWS_CONV4_OUT_BITS),
                 .RELU(`KWS_CONV4_RELU), .CH_BITS(4),
                 .ROM_FILE("rtl/gen/xl_g12/conv4_bn.hex")) u4 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(iv), .in_ch(ich[3:0]), .in_acc(iacc[`KWS_CONV4_ACC_BITS-1:0]),
        .out_valid(ov4), .out_ch(och4), .out_val(oval4));

    // ---- golden ---------------------------------------------------------- //
    // golden writes full 32-bit two's complement (export/golden.py _int_words),
    // so $signed() is the whole decode -- no sign extension to get wrong.
    localparam integer N2 = CLIPS * `KWS_CONV2_PW_N_OUT * T;
    localparam integer N3 = CLIPS * `KWS_CONV3_N_OUT * T;
    localparam integer N4 = CLIPS * `KWS_CONV4_N_OUT * T;

    reg [31:0] acc2 [0:N2-1];   reg [31:0] exp2 [0:N2-1];
    reg [31:0] acc3 [0:N3-1];   reg [31:0] exp3 [0:N3-1];
    reg [31:0] acc4 [0:N4-1];   reg [31:0] exp4 [0:N4-1];

    integer errors = 0, checked = 0;
    integer n, ch, t, idx;

    // Three registered stages, so an answer arrives three clocks after its
    // push. Driving one value at a time and waiting is slower than pipelining
    // the testbench, and it makes a mismatch name its own channel and frame --
    // worth far more here than simulation speed.
    task drive;
        input [6:0]         c;
        input signed [31:0] v;
        begin
            @(negedge clk);
            iv = 1'b1; ich = c; iacc = v;
            @(negedge clk);
            iv = 1'b0;
            @(negedge clk);
            @(negedge clk);
        end
    endtask

    task expect;
        input [63:0]        name;
        input               got_v;
        input signed [31:0] g;
        input signed [31:0] w;
        input integer       c;
        input integer       i;
        begin
            checked = checked + 1;
            if (!got_v) begin
                $display("FAIL %0s ch=%0d t=%0d: no output", name, c, i);
                errors = errors + 1;
            end else if (g !== w) begin
                errors = errors + 1;
                if (errors <= 8)
                    $display("FAIL %0s ch=%0d t=%0d: got %0d want %0d",
                             name, c, i, g, w);
            end
        end
    endtask

    initial begin
        $dumpfile("tb_affine.vcd");
        $dumpvars(0, tb_affine);

        $readmemh("rtl/gen/xl_g12/golden/conv2_pw_acc.hex", acc2, 0, N2 - 1);
        $readmemh("rtl/gen/xl_g12/golden/conv2_pw_out.hex", exp2, 0, N2 - 1);
        $readmemh("rtl/gen/xl_g12/golden/conv3_acc.hex", acc3, 0, N3 - 1);
        $readmemh("rtl/gen/xl_g12/golden/conv3_out.hex", exp3, 0, N3 - 1);
        $readmemh("rtl/gen/xl_g12/golden/conv4_acc.hex", acc4, 0, N4 - 1);
        $readmemh("rtl/gen/xl_g12/golden/conv4_out.hex", exp4, 0, N4 - 1);

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        // golden order is clip-major, then out_ch, then frame
        for (n = 0; n < CLIPS; n = n + 1)
            for (ch = 0; ch < `KWS_CONV2_PW_N_OUT; ch = ch + 1)
                for (t = 0; t < T; t = t + STEP) begin
                    idx = (n * `KWS_CONV2_PW_N_OUT + ch) * T + t;
                    drive(ch[6:0], $signed(acc2[idx]));
                    expect("conv2_pw", ov2, oval2, $signed(exp2[idx]), ch, t);
                end
        $display("ok   conv2_pw: %0d values", checked);

        for (n = 0; n < CLIPS; n = n + 1)
            for (ch = 0; ch < `KWS_CONV3_N_OUT; ch = ch + 1)
                for (t = 0; t < T; t = t + STEP) begin
                    idx = (n * `KWS_CONV3_N_OUT + ch) * T + t;
                    drive(ch[6:0], $signed(acc3[idx]));
                    expect("conv3", ov3, oval3, $signed(exp3[idx]), ch, t);
                end
        $display("ok   conv3");

        // only 12 classes, so every frame fits
        for (n = 0; n < CLIPS; n = n + 1)
            for (ch = 0; ch < `KWS_CONV4_N_OUT; ch = ch + 1)
                for (t = 0; t < T; t = t + 1) begin
                    idx = (n * `KWS_CONV4_N_OUT + ch) * T + t;
                    drive(ch[6:0], $signed(acc4[idx]));
                    expect("conv4", ov4, oval4, $signed(exp4[idx]), ch, t);
                end
        $display("ok   conv4");

        $display("\n%0d frames checked, %0d failures", checked, errors);
        $finish;
    end

    initial begin
        #900_000_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule

`default_nettype wire
