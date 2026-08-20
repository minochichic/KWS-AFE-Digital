// kws_tcs_sub end to end: the block input goes in, the sub-block output comes
// out, and nothing in between is checked.
//
// Input  : <gen>/golden/conv1_out.hex     (b1's block input)
// Expect : <gen>/golden/b1_s0_pw_out.hex
//
// The two halves already pass separately, so this adds exactly one thing: that
// the depthwise output reaches the pointwise unchanged and at the right time.
// A handoff that dropped or duplicated a frame would still leave both halves
// individually correct.
//
// <gen> is whichever export run_tb.sh selected; the second
// argument picks it and defaults to xl_g12.
//
//   ./rtl/run_tb.sh tcs_sub

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module tb_tcs_sub;

    localparam integer C_IN  = 128;
    localparam integer C_OUT = 64;
    localparam integer K     = 13;
    localparam integer PAD   = 6;
    localparam integer DWA   = 5;      // manifest: b1_s0_dw acc_bits
    localparam integer PWA   = 9;      // manifest: b1_s0_pw acc_bits
    localparam integer WB    = 32;
    localparam integer NWI   = C_IN  / WB;
    localparam integer NWO   = C_OUT / WB;
    localparam integer T     = 64;
    localparam integer CLIPS = 2;

    reg              clk = 1'b0;
    reg              rst_n = 1'b0;
    reg              start = 1'b0;
    reg              in_push = 1'b0;
    reg              in_real = 1'b0;
    reg  [C_IN-1:0]  in_frame = {C_IN{1'b0}};
    wire             busy;
    wire             out_valid;
    wire [C_OUT-1:0] out_frame;

    kws_tcs_sub #(.C_IN(C_IN), .C_OUT(C_OUT), .K(K), .PAD(PAD),
                  .DW_ACC(DWA), .PW_ACC(PWA), .WORD_BITS(WB),
                  .DW_W_FILE(`KWS_ROM_B1_S0_DW_W),
                  .DW_T_FILE(`KWS_ROM_B1_S0_DW_T),
                  .PW_W_FILE(`KWS_ROM_B1_S0_PW_W),
                  .PW_T_FILE(`KWS_ROM_B1_S0_PW_T)) dut (
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
        input         real_f;
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
        $dumpfile("tb_tcs_sub.vcd");
        $dumpvars(0, tb_tcs_sub);

        $readmemh(`KWS_GOLD_CONV1_OUT,    in_mem);
        $readmemh(`KWS_GOLD_B1_S0_PW_OUT, exp_mem);

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        for (clip = 0; clip < CLIPS; clip = clip + 1) begin
            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;

            for (i = 0; i < T + PAD; i = i + 1) begin
                frame = {C_IN{1'b0}};
                if (i < T)
                    for (j = 0; j < NWI; j = j + 1)
                        frame[j*WB +: WB] = in_mem[(clip*T + i)*NWI + j];
                push(i < T, frame);

                t = i - PAD;
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
        #40_000_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule

`default_nettype wire
