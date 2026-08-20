// kws_tail end to end: conv2_dw's +-1 frames in, a class index out.
//
// Input  : <gen>/golden/conv2_dw_out.hex     (packed +-1, 64 channels)
// Expect : <gen>/golden/predictions_fixed.txt
//
// The parts have each been checked against their own golden vectors already,
// so what this adds is the SEAMS -- that conv2_pw's raw accumulator reaches the
// right epilogue, that each dense conv is told its frame is complete at the
// right moment, that the pool accumulates across frames without being cleared
// or double-counted, and that the argmax reads a settled pool.
//
// It compares against predictions_fixed.txt, not predictions.txt: the former is
// what the integer path produces and the latter is the float model's. They
// agree on both clips here, but they are different claims, and checking the
// hardware against the float model would be checking two things at once.
//
// This is the slow one -- 18,700 cycles a frame, 64 frames, two clips.
//
// <gen> is whichever export run_tb.sh selected; the second
// argument picks it and defaults to xl_g12.
//
//   ./rtl/run_tb.sh tail

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module tb_tail;

    localparam integer CLIPS = 2;
    localparam integer T     = 64;
    localparam integer C2_IN = `KWS_CONV2_PW_N_IN;      // 64
    localparam integer NW    = (C2_IN + `KWS_WORD_BITS - 1) / `KWS_WORD_BITS;

    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    reg              start = 1'b0, iv = 1'b0;
    reg  [C2_IN-1:0] frame = {C2_IN{1'b0}};
    wire             busy, cls_v;
    wire [3:0]       cls;

    kws_tail #(.C2_IN(C2_IN), .C2_OUT(`KWS_CONV2_PW_N_OUT),
               .C2_ACC(`KWS_CONV2_PW_ACC_BITS), .WORD_BITS(`KWS_WORD_BITS),
               .C2_W_FILE(`KWS_ROM_CONV2_PW_W),
               .A2_GAIN(`KWS_CONV2_PW_GAIN_BITS),
               .A2_BIAS(`KWS_CONV2_PW_BIAS_BITS),
               .A2_SHIFT(`KWS_CONV2_PW_SHIFT),
               .A2_OUT(`KWS_CONV2_PW_OUT_BITS),
               .A2_FILE(`KWS_ROM_CONV2_PW_BN),

               .C3_OUT(`KWS_CONV3_N_OUT), .C3_W(`KWS_CONV3_W_BITS),
               .C3_ACC(`KWS_CONV3_ACC_BITS),
               .C3_W_FILE(`KWS_ROM_CONV3_W),
               .A3_GAIN(`KWS_CONV3_GAIN_BITS), .A3_BIAS(`KWS_CONV3_BIAS_BITS),
               .A3_SHIFT(`KWS_CONV3_SHIFT), .A3_OUT(`KWS_CONV3_OUT_BITS),
               .A3_FILE(`KWS_ROM_CONV3_BN),

               .C4_OUT(`KWS_CONV4_N_OUT), .C4_W(`KWS_CONV4_W_BITS),
               .C4_ACC(`KWS_CONV4_ACC_BITS),
               .C4_W_FILE(`KWS_ROM_CONV4_W),
               .A4_GAIN(`KWS_CONV4_GAIN_BITS), .A4_BIAS(`KWS_CONV4_BIAS_BITS),
               .A4_SHIFT(`KWS_CONV4_SHIFT), .A4_OUT(`KWS_CONV4_OUT_BITS),
               .A4_FILE(`KWS_ROM_CONV4_BN),

               .T_FRAMES(T), .POOL_BITS(`KWS_CONV4_POOL_BITS),
               .C2O_BITS(7), .C3O_BITS(7), .C4O_BITS(4)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .in_valid(iv), .in_frame(frame), .busy(busy),
        .class_valid(cls_v), .class_idx(cls));

    reg [`KWS_WORD_BITS-1:0] xin [0:CLIPS*T*NW-1];
    integer want [0:CLIPS-1];

    // class_valid pulses for one cycle DURING the argmax scan, and the scan is
    // inside busy -- so the frame loop's `while (busy)` swallows it. Catch it
    // in a register instead of polling after the fact.
    reg [3:0] got;
    reg       got_v;
    always @(posedge clk) if (cls_v) begin got <= cls; got_v <= 1'b1; end

    integer errors = 0, n, t, j, fh, code;

    initial begin
        $dumpfile("tb_tail.vcd");
        $dumpvars(0, tb_tail);

        $readmemh(`KWS_GOLD_CONV2_DW_OUT, xin,
                  0, CLIPS * T * NW - 1);
        // predictions_fixed.txt is one decimal integer per line
        fh = $fopen(`KWS_GOLD_PREDICTIONS_FIXED, "r");
        if (fh == 0) begin
            $display("FAIL cannot open predictions_fixed.txt");
            $finish;
        end
        for (n = 0; n < CLIPS; n = n + 1) begin
            code = $fscanf(fh, "%d", want[n]);
            if (code != 1) begin
                $display("FAIL predictions_fixed.txt has fewer than %0d lines",
                         CLIPS);
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

            for (t = 0; t < T; t = t + 1) begin
                frame = {C2_IN{1'b0}};
                for (j = 0; j < NW; j = j + 1)
                    frame[j*`KWS_WORD_BITS +: `KWS_WORD_BITS] =
                        xin[(n * T + t) * NW + j];
                @(negedge clk); iv = 1'b1;
                @(negedge clk); iv = 1'b0;
                while (busy) @(negedge clk);
            end

            // busy stays high through the argmax scan, so by here it is done
            repeat (4) @(negedge clk);

            if (!got_v) begin
                $display("FAIL clip%0d: no class emitted", n);
                errors = errors + 1;
            end else if (got != want[n]) begin
                $display("FAIL clip%0d: got class %0d, want %0d",
                         n, got, want[n]);
                errors = errors + 1;
            end else begin
                $display("ok   clip%0d: class %0d", n, got);
            end
        end

        $display("\n%0d frames checked, %0d failures", CLIPS, errors);
        $finish;
    end

    initial begin
        // 18,700 cycles a frame x 64 frames x 2 clips at 10ns is ~24 ms;
        // this is 8x that, and stays inside 32 bits
        #200_000_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule

`default_nettype wire
