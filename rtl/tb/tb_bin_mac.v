// Self-checking testbench for kws_bin_mac.
//
// Vectors come from rtl/tb/make_mac_vectors.py, which packs them with
// export/pack.py -- the same code that writes the weight ROMs. A testbench
// that packed its own vectors could agree with itself and still disagree with
// the hardware's weights, which is the failure worth designing against.
//
//   iverilog -g2005 -o /tmp/tb rtl/kws_bin_mac.v rtl/tb/tb_bin_mac.v && vvp /tmp/tb
//
// Add -DKWS_ASSERT to also arm the accumulator-bound check inside the DUT.

`timescale 1ns/1ps
`default_nettype none

// generated alongside the vectors, so the expected values and the
// packed bits can never come from different runs
`include "rtl/tb/vectors/expect.vh"

module tb_bin_mac;

    localparam integer WORD_BITS = 32;
    localparam integer ACC_BITS  = 16;
    localparam integer CNT_BITS  = 16;
    localparam integer MAXW      = 8;      // words in the widest case (128/32)

    reg                       clk = 1'b0;
    reg                       rst_n = 1'b0;
    reg                       start = 1'b0;
    reg  [CNT_BITS-1:0]       n_valid = 0;
    reg                       in_valid = 1'b0;
    reg  [WORD_BITS-1:0]      act = 0, wgt = 0;
    wire                      out_valid;
    wire signed [ACC_BITS-1:0] acc;

    kws_bin_mac #(.WORD_BITS(WORD_BITS), .ACC_BITS(ACC_BITS),
                  .CNT_BITS(CNT_BITS)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .n_valid(n_valid),
        .in_valid(in_valid), .act(act), .wgt(wgt),
        .out_valid(out_valid), .acc(acc));

    always #5 clk = ~clk;

    reg [WORD_BITS-1:0] a_mem [0:MAXW-1];
    reg [WORD_BITS-1:0] w_mem [0:MAXW-1];

    integer errors = 0;
    integer ran    = 0;

    // one case: load, stream, compare
    task run_case;
        input [8*8-1:0] name;
        input integer   n;
        input integer   nw;
        input integer   expect;
        integer i;
        begin
            @(negedge clk);
            n_valid  = n[CNT_BITS-1:0];
            start    = 1'b1;
            @(negedge clk);
            start    = 1'b0;

            for (i = 0; i < nw; i = i + 1) begin
                act      = a_mem[i];
                wgt      = w_mem[i];
                in_valid = 1'b1;
                @(negedge clk);
            end
            in_valid = 1'b0;

            // out_valid is registered, so it lands the cycle after the last word
            if (!out_valid) @(negedge clk);

            ran = ran + 1;
            if (acc !== expect[ACC_BITS-1:0]) begin
                $display("FAIL %0s: n=%0d got %0d want %0d", name, n, acc, expect);
                errors = errors + 1;
            end else begin
                $display("ok   %0s: n=%0d acc=%0d", name, n, acc);
            end
        end
    endtask

    // A pattern with no bits set at all: acc must be -N, not 0. Catches a DUT
    // that forgets the -N term, which a balanced random vector would hide.
    task run_all_minus_one;
        integer i;
        begin
            @(negedge clk);
            n_valid = 16'd64; start = 1'b1;
            @(negedge clk); start = 1'b0;
            for (i = 0; i < 2; i = i + 1) begin
                act = {WORD_BITS{1'b0}};          // all -1
                wgt = {WORD_BITS{1'b1}};          // all +1
                in_valid = 1'b1;
                @(negedge clk);
            end
            in_valid = 1'b0;
            if (!out_valid) @(negedge clk);
            ran = ran + 1;
            if (acc !== -16'sd64) begin
                $display("FAIL all-oppose: got %0d want -64", acc);
                errors = errors + 1;
            end else $display("ok   all-oppose: acc=%0d", acc);
        end
    endtask

    initial begin
        $dumpfile("tb_bin_mac.vcd");
        $dumpvars(0, tb_bin_mac);

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        // Kept explicit rather than parsed from cases.txt: $fscanf across
        // simulators is a portability tarpit, and these must match the file.
        $readmemh("rtl/tb/vectors/k13_act.hex", a_mem);
        $readmemh("rtl/tb/vectors/k13_wgt.hex", w_mem);
        run_case("k13", 13, `K13_NWORDS, `K13_EXPECT);

        $readmemh("rtl/tb/vectors/k29_act.hex", a_mem);
        $readmemh("rtl/tb/vectors/k29_wgt.hex", w_mem);
        run_case("k29", 29, `K29_NWORDS, `K29_EXPECT);

        $readmemh("rtl/tb/vectors/w31_act.hex", a_mem);
        $readmemh("rtl/tb/vectors/w31_wgt.hex", w_mem);
        run_case("w31", 31, `W31_NWORDS, `W31_EXPECT);

        $readmemh("rtl/tb/vectors/w32_act.hex", a_mem);
        $readmemh("rtl/tb/vectors/w32_wgt.hex", w_mem);
        run_case("w32", 32, `W32_NWORDS, `W32_EXPECT);

        $readmemh("rtl/tb/vectors/w33_act.hex", a_mem);
        $readmemh("rtl/tb/vectors/w33_wgt.hex", w_mem);
        run_case("w33", 33, `W33_NWORDS, `W33_EXPECT);

        $readmemh("rtl/tb/vectors/c64_act.hex", a_mem);
        $readmemh("rtl/tb/vectors/c64_wgt.hex", w_mem);
        run_case("c64", 64, `C64_NWORDS, `C64_EXPECT);

        $readmemh("rtl/tb/vectors/c128_act.hex", a_mem);
        $readmemh("rtl/tb/vectors/c128_wgt.hex", w_mem);
        run_case("c128", 128, `C128_NWORDS, `C128_EXPECT);

        run_all_minus_one;

        $display("\n%0d cases, %0d failures", ran, errors);
        if (errors != 0) $fatal(1);
        $finish;
    end

endmodule

`default_nettype wire
