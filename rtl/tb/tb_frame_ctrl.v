// kws_frame_ctrl from the comparator wires to the tensor the network was
// trained on.
//
// Expect : <gen>/golden/input.hex
//
// This closes the last gap in the chain. Every other testbench starts from a
// golden vector; this one starts from sixteen wires wiggling and has to arrive
// at exactly the same file the AFE simulation produced -- padding, framing and
// the discretisation rule all at once.
//
// THE PULSES ARE DELIBERATELY SCATTERED. Each channel fires for ONE cycle at
// its own offset inside the window, so reproducing the frame requires a real OR
// over the whole window. Sampling at any single instant, or ANDing, or latching
// only the last value would all fail -- whereas driving every channel together
// at one instant would pass for any of them.
//
// FRAME_CYCLES IS SMALL HERE. The real one is a million (10 ms at 100 MHz),
// which would be 10^8 cycles of simulation for one clip. The ratio between the
// window and the pulse offsets is what is under test, not its absolute size.
//
// <gen> is whichever export run_tb.sh selected; the second argument picks it
// and defaults to xl_g12.
//
//   ./rtl/run_tb.sh frame_ctrl

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module tb_frame_ctrl;

    localparam integer N_CH     = `KWS_N_CH;
    localparam integer T        = `KWS_T;
    localparam integer NATIVE_T = `KWS_NATIVE_T;
    localparam integer PAD_L    = `KWS_PAD_LEFT;
    localparam integer FC       = 24;      // stand-in for 1,000,000
    localparam integer CLIPS    = 2;
    localparam integer NW       = (N_CH + `KWS_WORD_BITS - 1) / `KWS_WORD_BITS;

    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    reg              start = 1'b0;
    reg  [N_CH-1:0]  cmp = {N_CH{1'b0}};
    wire             ov, busy;
    wire [N_CH-1:0]  of;

    kws_frame_ctrl #(.N_CH(N_CH), .FRAME_CYCLES(FC), .NATIVE_T(NATIVE_T),
                     .T(T), .PAD_LEFT(PAD_L), .CMP_INVERT(0)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .cmp(cmp),
        .out_ready(1'b1), .out_valid(ov), .out_frame(of), .busy(busy));

    reg [`KWS_WORD_BITS-1:0] gold [0:CLIPS*T*NW-1];
    reg [N_CH-1:0]           got  [0:T-1];
    integer                  rx;

    always @(posedge clk) begin
        if (ov && rx < T) begin
            got[rx] <= of;
            rx      <= rx + 1;
        end
    end

    // the golden frame for (clip, t), unpacked from the word layout
    function [N_CH-1:0] want_frame;
        input integer n;
        input integer t;
        integer j;
        reg [`KWS_WORD_BITS*NW-1:0] w;
        begin
            w = {(`KWS_WORD_BITS*NW){1'b0}};
            for (j = 0; j < NW; j = j + 1)
                w[j*`KWS_WORD_BITS +: `KWS_WORD_BITS] =
                    gold[(n * T + t) * NW + j];
            want_frame = w[N_CH-1:0];
        end
    endfunction

    // Where channel c fires inside a window. Spread, and kept clear of both
    // edges so a one-cycle error in the testbench's own window alignment
    // cannot slide a pulse into the neighbouring frame and turn a testbench
    // bug into a DUT failure.
    function integer offset_of;
        input integer c;
        begin
            offset_of = 2 + ((c * 5) % (FC - 4));
        end
    endfunction

    integer errors = 0, checked = 0, n, t, c, k;
    reg [N_CH-1:0] w;

    initial begin
        $dumpfile("tb_frame_ctrl.vcd");
        $dumpvars(0, tb_frame_ctrl);

        $readmemh(`KWS_GOLD_INPUT, gold, 0, CLIPS * T * NW - 1);

        rx = 0;
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        for (n = 0; n < CLIPS; n = n + 1) begin
            rx = 0;
            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;

            // PAD_L cycles of padding go out first, then window 0 opens on the
            // next cycle -- see the note in kws_frame_ctrl on the sequence.
            repeat (PAD_L) @(negedge clk);

            for (t = 0; t < NATIVE_T; t = t + 1) begin
                w = want_frame(n, PAD_L + t);
                for (k = 0; k < FC; k = k + 1) begin
                    cmp = {N_CH{1'b0}};
                    for (c = 0; c < N_CH; c = c + 1)
                        if (w[c] && offset_of(c) == k) cmp[c] = 1'b1;
                    @(negedge clk);
                end
            end
            cmp = {N_CH{1'b0}};

            // the trailing padding, plus the two-flop delay on the last window
            while (busy) @(negedge clk);
            repeat (8) @(negedge clk);

            if (rx !== T) begin
                $display("FAIL clip%0d: %0d frames out, expected %0d", n, rx, T);
                errors = errors + 1;
            end
            for (t = 0; t < T; t = t + 1) begin
                checked = checked + 1;
                if (got[t] !== want_frame(n, t)) begin
                    errors = errors + 1;
                    if (errors <= 6)
                        $display("FAIL clip%0d t=%0d: got %h want %h%0s",
                                 n, t, got[t], want_frame(n, t),
                                 (t < PAD_L || t >= PAD_L + NATIVE_T)
                                 ? "  (padding)" : "");
                end
            end
            $display("ok   clip%0d: %0d frames (%0d pad, %0d real, %0d pad)",
                     n, rx, PAD_L, NATIVE_T, T - PAD_L - NATIVE_T);
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
