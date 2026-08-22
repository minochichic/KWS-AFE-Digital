// kws_pw_conv against the golden vectors, chained onto the layer below it.
//
// Input  : <gen>/golden/b1_s0_dw_out.hex   (verified by tb_dw_conv)
// Expect : <gen>/golden/b1_s0_pw_out.hex
// ROMs   : <gen>/b1_s0_pw_{w,t}.hex
//
// The input is the output the depthwise module was just proven to produce, so
// a failure here cannot be blamed on the activations reaching it.
//
// <gen> is whichever export run_tb.sh selected; the second
// argument picks it and defaults to xl_g12.
//
//   ./rtl/run_tb.sh pw_conv

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module tb_pw_conv;

    localparam integer C_IN  = 128;
    localparam integer C_OUT = 64;
    localparam integer ACC   = 9;      // manifest: b1_s0_pw acc_bits
    localparam integer WB    = 32;
    localparam integer NWI   = C_IN  / WB;
    localparam integer NWO   = C_OUT / WB;
    localparam integer T     = 64;
    // The export decides how many clips the vectors hold (export/golden.py
    // --clips), so reading it from the generated header is the only way the
    // two cannot drift. Hardcoding 2 against a default of 8 checked a quarter
    // of the vectors and still printed ok.
    localparam integer CLIPS = `KWS_GOLD_CLIPS;

    reg               clk = 1'b0;
    reg               rst_n = 1'b0;
    reg               in_valid = 1'b0;
    reg  [C_IN-1:0]   in_frame = {C_IN{1'b0}};
    wire              busy;
    wire              out_valid;
    wire [C_OUT-1:0]  out_frame;

    kws_pw_conv #(.C_IN(C_IN), .C_OUT(C_OUT), .ACC_BITS(ACC), .WORD_BITS(WB),
                  .W_FILE(`KWS_ROM_B1_S0_PW_W),
                  .T_FILE(`KWS_ROM_B1_S0_PW_T)) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_frame(in_frame),
        .busy(busy), .out_valid(out_valid), .out_frame(out_frame));

    always #5 clk = ~clk;

    reg [WB-1:0] in_mem  [0:CLIPS*T*NWI-1];
    reg [WB-1:0] exp_mem [0:CLIPS*T*NWO-1];

    // sampled inside the task, at the negedge the wait loop lands on --
    // out_valid rises on the same edge that clears busy (rtl/README.md 4)
    reg [C_OUT-1:0] got;
    reg             got_v;

    integer errors = 0;
    integer checked = 0;
    integer clip, t, j;
    reg [C_OUT-1:0] want;

    task feed;
        input [C_IN-1:0] fr;
        begin
            @(negedge clk);
            in_valid = 1'b1;
            in_frame = fr;
            @(negedge clk);
            in_valid = 1'b0;
            while (busy) @(negedge clk);
            got_v = out_valid;
            got   = out_frame;
        end
    endtask

    initial begin
        $dumpfile("tb_pw_conv.vcd");
        $dumpvars(0, tb_pw_conv);

        $readmemh(`KWS_GOLD_B1_S0_DW_OUT, in_mem);
        $readmemh(`KWS_GOLD_B1_S0_PW_OUT, exp_mem);

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        for (clip = 0; clip < CLIPS; clip = clip + 1) begin
            for (t = 0; t < T; t = t + 1) begin
                // k=1: every frame in, one frame out. No fill, no drain --
                // there is no window along time to hang off an edge.
                for (j = 0; j < NWI; j = j + 1)
                    in_frame[j*WB +: WB] = in_mem[(clip*T + t)*NWI + j];
                feed(in_frame);

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
            end
            $display("ok   clip%0d: %0d frames", clip, T);
        end

        $display("\n%0d frames checked, %0d failures", checked, errors);
        $finish;
    end

    initial begin
        #20_000_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule

`default_nettype wire
