// tb_bench_mac -- the microbenchmark is alive and computes what it claims.
//
//   iverilog -g2012 -DKIND_UNDER_TEST=1 -s tb_bench_mac -o /tmp/b \
//            rtl/kws_bin_mac.v rtl/bench/bench_mac.v rtl/bench/tb_bench_mac.v
//   (cd rtl/bench && vvp /tmp/b)        -- bench_rom.mem is read from the cwd
//
// Copy 0 / lane 0's accumulator is recomputed from the operands the DUT shows
// at the engine's input (bench_mac mon_*) and compared when each vector
// closes. The operands are whatever the KIND feeds -- stimulus bits, or ROM
// and frame-register words for the memory kinds. For those, the weight word at
// the engine must also be the ROM's word for the address read (one cycle back
// for block RAM): an all-zero or misaddressed ROM fails. Which activation
// meets which weight is NOT checked -- the data is random, so the pairing
// changes no cost. `out` must toggle.

`timescale 1ns/1ps
`default_nettype none

`ifndef KIND_UNDER_TEST
`define KIND_UNDER_TEST 0
`endif

module tb_bench_mac;
    localparam integer KIND = `KIND_UNDER_TEST;
    localparam         BIN  = (KIND == 0 || KIND == 7 || KIND == 8 || KIND == 12);
    localparam         PM1  = (KIND == 3 || KIND == 11);
    reg clk = 0, rst_n = 0;
    wire out;
    always #10 clk = ~clk;

    bench_mac #(.KIND(KIND)) dut (.clk(clk), .rst_n(rst_n), .out(out));

    integer fails = 0, checked = 0, toggles = 0;
    reg     out_q = 0;
    reg     armed = 0;
    reg signed [31:0] ref_acc;

    always @(posedge clk) begin
        if (rst_n && out !== out_q) toggles = toggles + 1;
        out_q <= out;
    end

    // one term of lane 0, from the operands at the engine's input
    wire signed [31:0] term =
        BIN ? 2 * $countones(~(dut.mon_act[31:0] ^ dut.mon_wgt[31:0])) - 32
      : PM1 ? (dut.mon_act[0] ? $signed(dut.mon_wgt[7:0]) : -$signed(dut.mon_wgt[7:0]))
            : $signed(dut.mon_act[7:0]) * $signed(dut.mon_wgt[7:0]);

    // A vector closes on the start that opens the next one; the accumulator
    // still holds the finished vector there (sampled before the edge's update).
    always @(posedge clk) begin
        if (!rst_n) begin
            ref_acc <= 0;
        end else if (dut.mon_start) begin
            if (armed && KIND != 5) begin
                if (dut.mon_acc !== ref_acc[23:0]) begin
                    fails = fails + 1;
                    $display("FAIL acc %0d want %0d", $signed(dut.mon_acc), ref_acc);
                end
                checked = checked + 1;
            end else if (armed) begin
                checked = checked + 1;       // stimulus only: it runs
            end
            armed   <= 1;
            ref_acc <= 0;
        end else if (dut.mon_valid) begin
            ref_acc <= ref_acc + term;
        end
    end

    // memory kinds: the weight at the engine is the file's word (copy 0 for
    // binary, which uses bits [31:0])
    localparam         MEM  = (KIND >= 8 && KIND <= 12);
    reg  [255:0] golden [0:255];
    initial $readmemh("bench_rom.mem", golden);
    reg  [7:0]   addr_q;
    always @(posedge clk) addr_q <= dut.addr;
    wire [7:0]   raddr  = (KIND == 10) ? addr_q : dut.addr;
    wire [255:0] want_w = golden[raddr];
    integer rom_checked = 0;
    always @(posedge clk) if (MEM && rst_n && dut.mon_valid) begin
        if (BIN ? (dut.mon_wgt[31:0] !== want_w[31:0]) : (dut.mon_wgt !== want_w)) begin
            fails = fails + 1;
            if (fails < 5) $display("FAIL rom word at addr %0d", raddr);
        end
        rom_checked = rom_checked + 1;
    end

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (65 * 20 + 3) @(posedge clk);
        if (toggles == 0) begin
            $display("FAIL out never toggled -- logic would be optimised away");
            fails = fails + 1;
        end
        // a missing file leaves both the DUT's ROM and `golden` X, and X !== X
        // is false -- so check the file arrived at all
        if (MEM && (^golden[0] === 1'bx || ^golden[255] === 1'bx || golden[0] == 256'd0)) begin
            $display("FAIL bench_rom.mem not loaded (run vvp from rtl/bench)");
            fails = fails + 1;
        end
        if (MEM && rom_checked < 1000) begin
            $display("FAIL only %0d ROM words checked", rom_checked);
            fails = fails + 1;
        end
        $display("KIND %0d: %0d vectors checked, out toggled %0d times", KIND, checked, toggles);
        $display("%0d frames checked, %0d failures", checked, fails);
        $finish;
    end
endmodule

`default_nettype wire
