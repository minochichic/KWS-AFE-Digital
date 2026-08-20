// kws_block against the golden vectors: block input in, block output out.
//
// Input  : <gen>/golden/conv1_out.hex   (b1's block input)
// Expect : <gen>/golden/b1_add_out.hex  (after the residual add)
//
// Everything inside has already passed on its own, so what this adds is the
// structure: that the skip is taken from the right frame, that the add happens
// on integers before any threshold, and that exactly one threshold follows.
//
// Note the push count. Two depthwise stages means the chain lags by 2*PAD, so
// the caller pushes T + 2*PAD times and the outputs start 2*PAD pushes in.
//
// <gen> is whichever export run_tb.sh selected; the second
// argument picks it and defaults to xl_g12.
//
//   ./rtl/run_tb.sh block

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module tb_block;

    localparam integer C_IN  = 128;
    localparam integer C_MID = 64;
    localparam integer C_OUT = 64;
    localparam integer K     = 13;
    localparam integer PAD   = 6;
    localparam integer WB    = 32;
    localparam integer NWI   = C_IN  / WB;
    localparam integer NWO   = C_OUT / WB;
    localparam integer T     = 64;
    localparam integer CLIPS = 2;
    localparam integer LAG   = 2 * PAD;

    reg              clk = 1'b0;
    reg              rst_n = 1'b0;
    reg              start = 1'b0;
    reg              in_push = 1'b0;
    reg              in_real = 1'b0;
    reg  [C_IN-1:0]  in_frame = {C_IN{1'b0}};
    wire             busy;
    wire             out_valid;
    wire [C_OUT-1:0] out_frame;

    kws_block #(.C_IN(C_IN), .C_MID(C_MID), .C_OUT(C_OUT), .K(K), .PAD(PAD),
                .S0_DW_ACC(5), .S0_PW_ACC(9), .S1_DW_ACC(5), .S1_PW_ACC(8),
                .SKIP_ACC(9), .ADD_ACC(9), .WORD_BITS(WB),
                .S0_DW_W(`KWS_ROM_B1_S0_DW_W),
                .S0_DW_T(`KWS_ROM_B1_S0_DW_T),
                .S0_PW_W(`KWS_ROM_B1_S0_PW_W),
                .S0_PW_T(`KWS_ROM_B1_S0_PW_T),
                .S1_DW_W(`KWS_ROM_B1_S1_DW_W),
                .S1_DW_T(`KWS_ROM_B1_S1_DW_T),
                .S1_PW_W(`KWS_ROM_B1_S1_PW_W),
                .SKIP_W (`KWS_ROM_B1_SKIP_W),
                .ADD_T  (`KWS_ROM_B1_ADD_T)) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .in_push(in_push), .in_real(in_real), .in_frame(in_frame),
        .busy(busy), .out_valid(out_valid), .out_frame(out_frame));

    always #5 clk = ~clk;

    reg [WB-1:0] in_mem  [0:CLIPS*T*NWI-1];
    reg [WB-1:0] exp_mem [0:CLIPS*T*NWO-1];

    reg [C_OUT-1:0] got;
    reg             got_v;

    integer errors = 0;
    integer checked = 0;
    integer clip, i, j, t;
    reg [C_OUT-1:0] want;
    reg [C_IN-1:0]  frame;

    task push;
        input          real_f;
        input [C_IN-1:0] fr;
        begin
            @(negedge clk);
            in_push  = 1'b1;
            in_real  = real_f;
            in_frame = fr;
            @(negedge clk);
            in_push  = 1'b0;
            while (busy) @(negedge clk);
            got_v = out_valid;
            got   = out_frame;
        end
    endtask

    initial begin
        $dumpfile("tb_block.vcd");
        $dumpvars(0, tb_block);

        $readmemh(`KWS_GOLD_CONV1_OUT, in_mem);
        $readmemh(`KWS_GOLD_B1_ADD_OUT, exp_mem);

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        for (clip = 0; clip < CLIPS; clip = clip + 1) begin
            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;

            for (i = 0; i < T + LAG; i = i + 1) begin
                frame = {C_IN{1'b0}};
                if (i < T)
                    for (j = 0; j < NWI; j = j + 1)
                        frame[j*WB +: WB] = in_mem[(clip*T + i)*NWI + j];
                push(i < T, frame);

                t = i - LAG;
                if (t >= 0) begin
                    if (!got_v) begin
                        $display("FAIL clip%0d t=%0d: no output", clip, t);
                        errors = errors + 1;
                    end else begin
                        want = {C_OUT{1'b0}};
                        for (j = 0; j < NWO; j = j + 1)
                            want[j*WB +: WB] = exp_mem[(clip*T + t)*NWO + j];
                        checked = checked + 1;
                        if (got !== want) begin
                            errors = errors + 1;
                            if (errors <= 5)
                                $display("FAIL clip%0d t=%0d\n  got  %h\n  want %h",
                                         clip, t, got, want);
                        end
                    end
                end else if (got_v) begin
                    $display("FAIL clip%0d push %0d: output during fill",
                             clip, i);
                    errors = errors + 1;
                end
            end
            $display("ok   clip%0d: %0d frames", clip, T);
        end

        $display("\n%0d frames checked, %0d failures", checked, errors);
        $finish;
    end

    initial begin
        #200_000_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule

`default_nettype wire
