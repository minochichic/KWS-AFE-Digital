// kws_top: the whole network, AFE frames in, a class index out.
//
// Input  : <gen>/golden/input.hex            (16 ch x 128, packed +-1)
// Expect : <gen>/golden/predictions_fixed.txt
//
// Every module below has already been checked against its own golden vectors,
// so what this adds is the five-phase sequence: that each plane is filled
// before its reader starts, that conv1's own four flush pushes are generated
// here rather than asked of the caller, that each plane's FLUSH matches its
// CONSUMER's drain (12, 14, 16, 28 -- all different), and that the last frame
// of a clip reaches the pooled argmax.
//
// Widths and paths come from parameters.vh by name, not by position. The layer
// indices in those macro names track the manifest's order, so if a stage were
// inserted the names would move and this file would fail to compile -- which is
// the desired failure, rather than silently instantiating with the wrong ROM.
//
// The slowest thing here: 2.87M cycles a clip.
//
// <gen> is whichever export run_tb.sh selected; the second
// argument picks it and defaults to xl_g12.
//
//   ./rtl/run_tb.sh top

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module tb_top;

    localparam integer CLIPS = 2;
    localparam integer T_IN  = `KWS_T;
    localparam integer T_OUT = T_IN / `KWS_L0_CONV1_STRIDE;
    localparam integer NWI   = (`KWS_N_CH + `KWS_WORD_BITS - 1)
                              / `KWS_WORD_BITS;

    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    reg                  start = 1'b0, iv = 1'b0;
    reg  [`KWS_N_CH-1:0] frame = {`KWS_N_CH{1'b0}};
    wire                 busy, rdy, cls_v;
    wire [3:0]           cls;

    kws_top #(.WORD_BITS(`KWS_WORD_BITS), .T_IN(T_IN), .T_OUT(T_OUT),
              .N_CH(`KWS_N_CH), .C1_OUT(`KWS_L0_CONV1_OUT_CH),
              .C1_K(`KWS_L0_CONV1_KERNEL), .C1_PAD(`KWS_L0_CONV1_PADDING),
              .C1_STRIDE(`KWS_L0_CONV1_STRIDE),
              .C1_ACC(`KWS_L0_CONV1_ACC_BITS),
              .C1_W(`KWS_ROM_CONV1_W),
              .C1_T(`KWS_ROM_CONV1_T),

              .B1_MID(`KWS_L3_B1_S0_PW_OUT_CH), .B1_OUT(`KWS_L5_B1_S1_PW_OUT_CH),
              .B1_K(`KWS_L2_B1_S0_DW_KERNEL), .B1_PAD(`KWS_L2_B1_S0_DW_PADDING),
              .B1_S0DW_A(`KWS_L2_B1_S0_DW_ACC_BITS),
              .B1_S0PW_A(`KWS_L3_B1_S0_PW_ACC_BITS),
              .B1_S1DW_A(`KWS_L4_B1_S1_DW_ACC_BITS),
              .B1_S1PW_A(`KWS_L5_B1_S1_PW_ACC_BITS),
              .B1_SKIP_A(`KWS_L1_B1_SKIP_ACC_BITS),
              .B1_ADD_A(`KWS_L6_B1_ADD_ACC_BITS),
              .B1_S0DW_W(`KWS_ROM_B1_S0_DW_W),
              .B1_S0DW_T(`KWS_ROM_B1_S0_DW_T),
              .B1_S0PW_W(`KWS_ROM_B1_S0_PW_W),
              .B1_S0PW_T(`KWS_ROM_B1_S0_PW_T),
              .B1_S1DW_W(`KWS_ROM_B1_S1_DW_W),
              .B1_S1DW_T(`KWS_ROM_B1_S1_DW_T),
              .B1_S1PW_W(`KWS_ROM_B1_S1_PW_W),
              .B1_SKIP_W(`KWS_ROM_B1_SKIP_W),
              .B1_ADD_T(`KWS_ROM_B1_ADD_T),

              .B2_K(`KWS_L7_B2_S0_DW_KERNEL), .B2_PAD(`KWS_L7_B2_S0_DW_PADDING),
              .B2_S0DW_A(`KWS_L7_B2_S0_DW_ACC_BITS),
              .B2_S0PW_A(`KWS_L8_B2_S0_PW_ACC_BITS),
              .B2_S1DW_A(`KWS_L9_B2_S1_DW_ACC_BITS),
              .B2_S1PW_A(`KWS_L10_B2_S1_PW_ACC_BITS),
              .B2_SKIP_A(`KWS_L11_B2_ADD_ACC_BITS),
              .B2_ADD_A(`KWS_L11_B2_ADD_ACC_BITS),
              .B2_S0DW_W(`KWS_ROM_B2_S0_DW_W),
              .B2_S0DW_T(`KWS_ROM_B2_S0_DW_T),
              .B2_S0PW_W(`KWS_ROM_B2_S0_PW_W),
              .B2_S0PW_T(`KWS_ROM_B2_S0_PW_T),
              .B2_S1DW_W(`KWS_ROM_B2_S1_DW_W),
              .B2_S1DW_T(`KWS_ROM_B2_S1_DW_T),
              .B2_S1PW_W(`KWS_ROM_B2_S1_PW_W),
              .B2_ADD_T(`KWS_ROM_B2_ADD_T),

              .B3_K(`KWS_L12_B3_S0_DW_KERNEL),
              .B3_PAD(`KWS_L12_B3_S0_DW_PADDING),
              .B3_S0DW_A(`KWS_L12_B3_S0_DW_ACC_BITS),
              .B3_S0PW_A(`KWS_L13_B3_S0_PW_ACC_BITS),
              .B3_S1DW_A(`KWS_L14_B3_S1_DW_ACC_BITS),
              .B3_S1PW_A(`KWS_L15_B3_S1_PW_ACC_BITS),
              .B3_SKIP_A(`KWS_L16_B3_ADD_ACC_BITS),
              .B3_ADD_A(`KWS_L16_B3_ADD_ACC_BITS),
              .B3_S0DW_W(`KWS_ROM_B3_S0_DW_W),
              .B3_S0DW_T(`KWS_ROM_B3_S0_DW_T),
              .B3_S0PW_W(`KWS_ROM_B3_S0_PW_W),
              .B3_S0PW_T(`KWS_ROM_B3_S0_PW_T),
              .B3_S1DW_W(`KWS_ROM_B3_S1_DW_W),
              .B3_S1DW_T(`KWS_ROM_B3_S1_DW_T),
              .B3_S1PW_W(`KWS_ROM_B3_S1_PW_W),
              .B3_ADD_T(`KWS_ROM_B3_ADD_T),

              .C2_K(`KWS_L17_CONV2_DW_KERNEL),
              .C2_PAD(`KWS_L17_CONV2_DW_PADDING),
              .C2_DIL(`KWS_L17_CONV2_DW_DILATION),
              .C2_ACC(`KWS_L17_CONV2_DW_ACC_BITS),
              .C2_W(`KWS_ROM_CONV2_DW_W),
              .C2_T(`KWS_ROM_CONV2_DW_T),

              .TL_C2_OUT(`KWS_CONV2_PW_N_OUT), .TL_C2_ACC(`KWS_CONV2_PW_ACC_BITS),
              .TL_C2_W(`KWS_ROM_CONV2_PW_W),
              .TL_A2_G(`KWS_CONV2_PW_GAIN_BITS),
              .TL_A2_B(`KWS_CONV2_PW_BIAS_BITS),
              .TL_A2_S(`KWS_CONV2_PW_SHIFT), .TL_A2_O(`KWS_CONV2_PW_OUT_BITS),
              .TL_A2_F(`KWS_ROM_CONV2_PW_BN),
              .TL_C3_OUT(`KWS_CONV3_N_OUT), .TL_C3_W(`KWS_CONV3_W_BITS),
              .TL_C3_ACC(`KWS_CONV3_ACC_BITS),
              .TL_C3_WF(`KWS_ROM_CONV3_W),
              .TL_A3_G(`KWS_CONV3_GAIN_BITS), .TL_A3_B(`KWS_CONV3_BIAS_BITS),
              .TL_A3_S(`KWS_CONV3_SHIFT), .TL_A3_O(`KWS_CONV3_OUT_BITS),
              .TL_A3_F(`KWS_ROM_CONV3_BN),
              .TL_C4_OUT(`KWS_CONV4_N_OUT), .TL_C4_W(`KWS_CONV4_W_BITS),
              .TL_C4_ACC(`KWS_CONV4_ACC_BITS),
              .TL_C4_WF(`KWS_ROM_CONV4_W),
              .TL_A4_G(`KWS_CONV4_GAIN_BITS), .TL_A4_B(`KWS_CONV4_BIAS_BITS),
              .TL_A4_S(`KWS_CONV4_SHIFT), .TL_A4_O(`KWS_CONV4_OUT_BITS),
              .TL_A4_F(`KWS_ROM_CONV4_BN),
              .TL_POOL(`KWS_CONV4_POOL_BITS), .TL_C4O_B(4)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .in_valid(iv), .in_frame(frame),
        .in_ready(rdy), .busy(busy),
        .class_valid(cls_v), .class_idx(cls));

    reg [`KWS_WORD_BITS-1:0] xin [0:CLIPS*T_IN*NWI-1];
    integer want [0:CLIPS-1];

    reg [3:0] got;
    reg       got_v;
    always @(posedge clk) if (cls_v) begin got <= cls; got_v <= 1'b1; end

    integer errors = 0, n, t, j, fh, code;

    initial begin
        $dumpfile("tb_top.vcd");
        $dumpvars(0, tb_top);

        $readmemh(`KWS_GOLD_INPUT, xin,
                  0, CLIPS * T_IN * NWI - 1);
        fh = $fopen(`KWS_GOLD_PREDICTIONS_FIXED, "r");
        if (fh == 0) begin
            $display("FAIL cannot open predictions_fixed.txt");
            $finish;
        end
        for (n = 0; n < CLIPS; n = n + 1) begin
            code = $fscanf(fh, "%d", want[n]);
            if (code != 1) begin
                $display("FAIL predictions_fixed.txt is short");
                $finish;
            end
        end
        $fclose(fh);

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        for (n = 0; n < CLIPS; n = n + 1) begin
            got_v = 1'b0;
            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;

            // T_IN frames. conv1's four flush pushes are kws_top's business,
            // not the caller's -- the AFE has nothing to say about the kernel.
            for (t = 0; t < T_IN; t = t + 1) begin
                frame = {`KWS_N_CH{1'b0}};
                for (j = 0; j < NWI; j = j + 1)
                    frame[j*`KWS_WORD_BITS +: `KWS_WORD_BITS] =
                        xin[(n * T_IN + t) * NWI + j];
                // wait for the DUT to say it can take one, rather than
                // watching conv1's busy from outside -- that rises a cycle
                // after the push reaches it, so the frame lands in the gap
                while (!rdy) @(negedge clk);
                @(negedge clk); iv = 1'b1;
                @(negedge clk); iv = 1'b0;
            end

            wait (got_v === 1'b1);
            repeat (2) @(negedge clk);

            if (got != want[n]) begin
                $display("FAIL clip%0d: got class %0d, want %0d",
                         n, got, want[n]);
                errors = errors + 1;
            end else begin
                $display("ok   clip%0d: class %0d", n, got);
            end
            while (busy) @(negedge clk);
        end

        $display("\n%0d frames checked, %0d failures", CLIPS, errors);
        $finish;
    end

    initial begin
        // 2.87M cycles a clip at 10ns is ~29 ms; this is 5x for two clips
        #300_000_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule

`default_nettype wire
