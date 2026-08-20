// kws_dw_conv against the real thing: trained weights, trained thresholds, and
// the activations the network actually produced.
//
// Input  : <gen>/golden/conv1_out.hex   (b1's block input)
// Expect : <gen>/golden/b1_s0_dw_out.hex
// ROMs   : <gen>/b1_s0_dw_{w,t}.hex
//
// Nothing here is synthetic. If this passes, the line buffer, the tap gather,
// the edge shift, n_valid and the fused threshold are all right together --
// and if it fails, kws_bin_mac is already known good, so the fault is in one
// of those five.
//
// A SECOND INSTANCE FOR DILATION. conv2_dw spreads 29 taps over 57 slots, so
// the buffer is wider than the kernel and the taps sit at slot j*DIL. The point
// of running both is that at DIL=1 the two are the same size and nothing can
// tell a stride-aware gather from a plain one -- b1_s0_dw would pass either
// way. Only conv2_dw separates them.
//
// <gen> is whichever export run_tb.sh selected; the second
// argument picks it and defaults to xl_g12.
//
//   ./rtl/run_tb.sh dw_conv

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module tb_dw_conv;

    localparam integer C     = 128;
    localparam integer K     = 13;
    localparam integer PAD   = 6;
    localparam integer ACC   = 5;      // manifest: b1_s0_dw acc_bits
    localparam integer WB    = 32;
    localparam integer NW    = C / WB; // words per frame
    localparam integer T     = 64;
    localparam integer CLIPS = 2;

    reg              clk = 1'b0;
    reg              rst_n = 1'b0;
    reg              start = 1'b0;
    reg              in_push = 1'b0;
    reg              in_real = 1'b0;
    reg  [C-1:0]     in_frame = {C{1'b0}};
    wire             busy;
    wire             out_valid;
    wire [C-1:0]     out_frame;

    kws_dw_conv #(.C(C), .K(K), .PAD(PAD), .ACC_BITS(ACC), .WORD_BITS(WB),
                  .W_FILE(`KWS_ROM_B1_S0_DW_W),
                  .T_FILE(`KWS_ROM_B1_S0_DW_T)) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .in_push(in_push), .in_real(in_real), .in_frame(in_frame),
        .busy(busy), .out_valid(out_valid), .out_frame(out_frame));

    always #5 clk = ~clk;

    reg [WB-1:0] in_mem  [0:CLIPS*T*NW-1];
    reg [WB-1:0] exp_mem [0:CLIPS*T*NW-1];

    // Sampled inside push(), not by a separate always block. out_valid is
    // registered and pulses on the same edge that clears busy, so both are
    // visible at the negedge the wait loop exits on. A posedge capture block
    // would set its flag one edge LATER than the loop returns -- which is why
    // the first version saw "no output" on every single frame.
    reg [C-1:0] got;
    reg         got_v;

    integer errors = 0;
    integer checked = 0;
    integer clip, i, j, t;
    reg [C-1:0] want, frame;

    task push;                        // one frame in, wait until it settles
        input       real_f;
        input [C-1:0] fr;
        begin
            @(negedge clk);
            in_push  = 1'b1;
            in_real  = real_f;
            in_frame = fr;
            @(negedge clk);
            in_push  = 1'b0;
            while (busy) @(negedge clk);
            got_v = out_valid;          // still high on this negedge
            got   = out_frame;
        end
    endtask

    // ---- second DUT: conv2_dw, 29 taps at dilation 2 -------------------- //
    localparam integer DC   = 64;
    localparam integer DK   = 29;
    localparam integer DPAD = 28;
    localparam integer DDIL = 2;
    localparam integer DACC = 6;       // manifest: conv2_dw acc_bits
    localparam integer DNW  = DC / WB;

    reg           d_start = 1'b0, d_push = 1'b0, d_real = 1'b0;
    reg  [DC-1:0] d_frame = {DC{1'b0}};
    wire          d_busy, d_ov;
    wire [DC-1:0] d_of;

    kws_dw_conv #(.C(DC), .K(DK), .PAD(DPAD), .DIL(DDIL), .ACC_BITS(DACC),
                  .WORD_BITS(WB),
                  .W_FILE(`KWS_ROM_CONV2_DW_W),
                  .T_FILE(`KWS_ROM_CONV2_DW_T)) dut_d (
        .clk(clk), .rst_n(rst_n), .start(d_start),
        .in_push(d_push), .in_real(d_real), .in_frame(d_frame),
        .busy(d_busy), .out_valid(d_ov), .out_frame(d_of));

    reg [WB-1:0] d_in  [0:CLIPS*T*DNW-1];
    reg [WB-1:0] d_exp [0:CLIPS*T*DNW-1];
    reg [DC-1:0] d_got, d_want, d_fr;
    reg          d_got_v;

    task d_pushf;
        input        real_f;
        input [DC-1:0] fr;
        begin
            @(negedge clk);
            d_push = 1'b1; d_real = real_f; d_frame = fr;
            @(negedge clk);
            d_push = 1'b0;
            while (d_busy) @(negedge clk);
            d_got_v = d_ov;
            d_got   = d_of;
        end
    endtask

    initial begin
        $dumpfile("tb_dw_conv.vcd");
        $dumpvars(0, tb_dw_conv);

        $readmemh(`KWS_GOLD_CONV1_OUT,    in_mem);
        $readmemh(`KWS_GOLD_B1_S0_DW_OUT, exp_mem);

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        for (clip = 0; clip < CLIPS; clip = clip + 1) begin
            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;

            // T real pushes then PAD flush pushes -- the drain that produces
            // the last PAD outputs (docs/diagrams/24_pipeline_drain.svg)
            for (i = 0; i < T + PAD; i = i + 1) begin
                frame = {C{1'b0}};
                if (i < T)
                    for (j = 0; j < NW; j = j + 1)
                        frame[j*WB +: WB] = in_mem[(clip*T + i)*NW + j];
                push(i < T, frame);

                t = i - PAD;                    // which output this push made
                if (t >= 0) begin
                    if (!got_v) begin
                        $display("FAIL clip%0d t=%0d: no output", clip, t);
                        errors = errors + 1;
                    end else begin
                        want = {C{1'b0}};
                        for (j = 0; j < NW; j = j + 1)
                            want[j*WB +: WB] = exp_mem[(clip*T + t)*NW + j];
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

        // ---- dilated ---------------------------------------------------- //
        $readmemh(`KWS_GOLD_B3_ADD_OUT,   d_in);
        $readmemh(`KWS_GOLD_CONV2_DW_OUT, d_exp);

        for (clip = 0; clip < CLIPS; clip = clip + 1) begin
            @(negedge clk); d_start = 1'b1;
            @(negedge clk); d_start = 1'b0;

            for (i = 0; i < T + DPAD; i = i + 1) begin
                d_fr = {DC{1'b0}};
                if (i < T)
                    for (j = 0; j < DNW; j = j + 1)
                        d_fr[j*WB +: WB] = d_in[(clip*T + i)*DNW + j];
                d_pushf(i < T, d_fr);

                t = i - DPAD;
                if (t >= 0) begin
                    if (!d_got_v) begin
                        $display("FAIL conv2_dw clip%0d t=%0d: no output",
                                 clip, t);
                        errors = errors + 1;
                    end else begin
                        d_want = {DC{1'b0}};
                        for (j = 0; j < DNW; j = j + 1)
                            d_want[j*WB +: WB] = d_exp[(clip*T + t)*DNW + j];
                        checked = checked + 1;
                        if (d_got !== d_want) begin
                            errors = errors + 1;
                            if (errors <= 5)
                                $display("FAIL conv2_dw clip%0d t=%0d\n  got  %h\n  want %h",
                                         clip, t, d_got, d_want);
                        end
                    end
                end else if (d_got_v) begin
                    $display("FAIL conv2_dw clip%0d push %0d: output during fill",
                             clip, i);
                    errors = errors + 1;
                end
            end
            $display("ok   conv2_dw clip%0d: %0d frames (29 taps, dilation 2)",
                     clip, T);
        end

        $display("\n%0d frames checked, %0d failures", checked, errors);
        $finish;
    end

    // a runaway FSM should not hang the run
    initial begin
        #60_000_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule

`default_nettype wire
