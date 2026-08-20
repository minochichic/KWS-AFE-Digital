// kws_dense_conv against the golden accumulators, for conv3 and conv4.
//
// conv3:  in  <gen>/golden/conv2_pw_out.hex   (its input activations)
//         out <gen>/golden/conv3_acc.hex
// conv4:  in  <gen>/golden/conv3_out.hex
//         out <gen>/golden/conv4_acc.hex
//
// Two instances, because the two layers differ in exactly the way the module
// claims not to care about: 128 outputs against 12, and a channel count that is
// a power of two against one that is not. If that claim is wrong, one fails.
//
// Note where each instance's IN_BITS comes from: the PREVIOUS site's OUT_BITS.
// That is the dataflow written down -- conv3 consumes what conv2_pw's epilogue
// produced, so its input width is not a number to choose but one to inherit.
//
// A frame at a time: load C_IN activations, pulse start, collect C_OUT
// accumulators. Four frames per clip rather than all 64, because conv3 is
// 16,384 cycles per frame and the coverage from more frames is not worth the
// wall clock.
//
// <gen> is whichever export run_tb.sh selected; the second
// argument picks it and defaults to xl_g12.
//
//   ./rtl/run_tb.sh dense_conv

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module tb_dense_conv;

    localparam integer CLIPS  = 2;
    localparam integer T      = 64;
    localparam integer FRAMES = 4;      // which frames: 0, 1, 17, 63 below

    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    // ---- conv3 ----------------------------------------------------------- //
    reg                                       lv3 = 1'b0, go3 = 1'b0;
    reg  [6:0]                                lc3 = 7'd0;
    reg  signed [`KWS_CONV2_PW_OUT_BITS-1:0]  lx3 = 0;
    wire                                      bsy3, av3;
    wire [6:0]                                ac3;
    wire signed [`KWS_CONV3_ACC_BITS-1:0]     ao3;

    kws_dense_conv #(.C_IN(`KWS_CONV3_N_IN), .C_OUT(`KWS_CONV3_N_OUT),
                     .IN_BITS(`KWS_CONV2_PW_OUT_BITS),
                     .W_BITS(`KWS_CONV3_W_BITS),
                     .ACC_BITS(`KWS_CONV3_ACC_BITS),
                     .CI_BITS(7), .CO_BITS(7),
                     .W_FILE(`KWS_ROM_CONV3_W)) u3 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(lv3), .in_ch(lc3), .in_val(lx3),
        .start(go3), .busy(bsy3),
        .acc_valid(av3), .acc_ch(ac3), .acc_out(ao3));

    // ---- conv4 ----------------------------------------------------------- //
    reg                                       lv4 = 1'b0, go4 = 1'b0;
    reg  [6:0]                                lc4 = 7'd0;
    reg  signed [`KWS_CONV3_OUT_BITS-1:0]     lx4 = 0;
    wire                                      bsy4, av4;
    wire [3:0]                                ac4;
    wire signed [`KWS_CONV4_ACC_BITS-1:0]     ao4;

    kws_dense_conv #(.C_IN(`KWS_CONV4_N_IN), .C_OUT(`KWS_CONV4_N_OUT),
                     .IN_BITS(`KWS_CONV3_OUT_BITS),
                     .W_BITS(`KWS_CONV4_W_BITS),
                     .ACC_BITS(`KWS_CONV4_ACC_BITS),
                     .CI_BITS(7), .CO_BITS(4),
                     .W_FILE(`KWS_ROM_CONV4_W)) u4 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(lv4), .in_ch(lc4), .in_val(lx4),
        .start(go4), .busy(bsy4),
        .acc_valid(av4), .acc_ch(ac4), .acc_out(ao4));

    // ---- golden ---------------------------------------------------------- //
    localparam integer NX3 = CLIPS * `KWS_CONV3_N_IN  * T;
    localparam integer NA3 = CLIPS * `KWS_CONV3_N_OUT * T;
    localparam integer NX4 = CLIPS * `KWS_CONV4_N_IN  * T;
    localparam integer NA4 = CLIPS * `KWS_CONV4_N_OUT * T;

    reg [31:0] x3 [0:NX3-1];   reg [31:0] a3 [0:NA3-1];
    reg [31:0] x4 [0:NX4-1];   reg [31:0] a4 [0:NA4-1];

    // captured accumulators, one slot per output channel
    reg signed [31:0] got3 [0:`KWS_CONV3_N_OUT-1];
    reg signed [31:0] got4 [0:`KWS_CONV4_N_OUT-1];
    reg [`KWS_CONV3_N_OUT-1:0] seen3;
    reg [`KWS_CONV4_N_OUT-1:0] seen4;

    always @(posedge clk) begin
        if (av3) begin got3[ac3] <= ao3; seen3[ac3] <= 1'b1; end
        if (av4) begin got4[ac4] <= ao4; seen4[ac4] <= 1'b1; end
    end

    integer errors = 0, checked = 0;
    integer n, t, i, o, fi;
    integer frame_of [0:FRAMES-1];

    initial begin
        frame_of[0] = 0; frame_of[1] = 1; frame_of[2] = 17; frame_of[3] = 63;
    end

    initial begin
        $dumpfile("tb_dense_conv.vcd");
        $dumpvars(0, tb_dense_conv);

        $readmemh(`KWS_GOLD_CONV2_PW_OUT, x3, 0, NX3 - 1);
        $readmemh(`KWS_GOLD_CONV3_ACC,    a3, 0, NA3 - 1);
        $readmemh(`KWS_GOLD_CONV3_OUT,    x4, 0, NX4 - 1);
        $readmemh(`KWS_GOLD_CONV4_ACC,    a4, 0, NA4 - 1);

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        for (n = 0; n < CLIPS; n = n + 1)
            for (fi = 0; fi < FRAMES; fi = fi + 1) begin
                t = frame_of[fi];

                // ---- conv3 ---------------------------------------------- //
                seen3 = {`KWS_CONV3_N_OUT{1'b0}};
                for (i = 0; i < `KWS_CONV3_N_IN; i = i + 1) begin
                    @(negedge clk);
                    lv3 = 1'b1; lc3 = i[6:0];
                    lx3 = x3[(n * `KWS_CONV3_N_IN + i) * T + t]
                            [`KWS_CONV2_PW_OUT_BITS-1:0];
                end
                @(negedge clk); lv3 = 1'b0;
                @(negedge clk); go3 = 1'b1;
                @(negedge clk); go3 = 1'b0;
                while (bsy3) @(negedge clk);
                repeat (2) @(negedge clk);

                for (o = 0; o < `KWS_CONV3_N_OUT; o = o + 1) begin
                    checked = checked + 1;
                    if (!seen3[o]) begin
                        $display("FAIL conv3 clip%0d t=%0d ch=%0d: never emitted",
                                 n, t, o);
                        errors = errors + 1;
                    end else if (got3[o] !==
                                 $signed(a3[(n * `KWS_CONV3_N_OUT + o) * T + t]))
                    begin
                        errors = errors + 1;
                        if (errors <= 8)
                            $display("FAIL conv3 clip%0d t=%0d ch=%0d: got %0d want %0d",
                                     n, t, o, got3[o],
                                     $signed(a3[(n*`KWS_CONV3_N_OUT+o)*T+t]));
                    end
                end

                // ---- conv4 ---------------------------------------------- //
                seen4 = {`KWS_CONV4_N_OUT{1'b0}};
                for (i = 0; i < `KWS_CONV4_N_IN; i = i + 1) begin
                    @(negedge clk);
                    lv4 = 1'b1; lc4 = i[6:0];
                    lx4 = x4[(n * `KWS_CONV4_N_IN + i) * T + t]
                            [`KWS_CONV3_OUT_BITS-1:0];
                end
                @(negedge clk); lv4 = 1'b0;
                @(negedge clk); go4 = 1'b1;
                @(negedge clk); go4 = 1'b0;
                while (bsy4) @(negedge clk);
                repeat (2) @(negedge clk);

                for (o = 0; o < `KWS_CONV4_N_OUT; o = o + 1) begin
                    checked = checked + 1;
                    if (!seen4[o]) begin
                        $display("FAIL conv4 clip%0d t=%0d ch=%0d: never emitted",
                                 n, t, o);
                        errors = errors + 1;
                    end else if (got4[o] !==
                                 $signed(a4[(n * `KWS_CONV4_N_OUT + o) * T + t]))
                    begin
                        errors = errors + 1;
                        if (errors <= 8)
                            $display("FAIL conv4 clip%0d t=%0d ch=%0d: got %0d want %0d",
                                     n, t, o, got4[o],
                                     $signed(a4[(n*`KWS_CONV4_N_OUT+o)*T+t]));
                    end
                end
            end

        $display("ok   conv3 and conv4, %0d frames each", CLIPS * FRAMES);
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
