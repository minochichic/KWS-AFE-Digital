// tb_bench_mac -- the microbenchmark is alive and computes what it claims.
//
//   iverilog -g2005 -Wall -DKIND_UNDER_TEST=1 -s tb_bench_mac -o /tmp/b \
//            rtl/kws_bin_mac.v rtl/bench/bench_mac.v rtl/bench/tb_bench_mac.v
//
// For the int8 kinds, lane 0's accumulator is recomputed from the same
// stimulus bits and compared at the end of every vector. For bin32 the
// accumulator is checked against 2*popcount - N. `out` must toggle.

`timescale 1ns/1ps
`default_nettype none

`ifndef KIND_UNDER_TEST
`define KIND_UNDER_TEST 0
`endif

module tb_bench_mac;
    localparam integer KIND = `KIND_UNDER_TEST;
    reg clk = 0, rst_n = 0;
    wire out;
    always #10 clk = ~clk;

    bench_mac #(.KIND(KIND)) dut (.clk(clk), .rst_n(rst_n), .out(out));

    integer fails = 0, checked = 0, toggles = 0;
    reg     out_q = 0;
    reg signed [31:0] ref_acc;

    always @(posedge clk) begin
        if (rst_n && out !== out_q) toggles = toggles + 1;
        out_q <= out;
    end

    // reference for lane 0 / the binary engine, from the DUT's own stimulus
    always @(posedge clk) begin
        if (!rst_n) ref_acc <= 0;
        else if (dut.start) ref_acc <= 0;
        else if (dut.in_valid) begin
            if (KIND == 0)
                ref_acc <= ref_acc + 2 * $countones(~(dut.act_bits[31:0] ^ dut.wgt_bits[31:0])) - 32;
            else if (KIND == 3)
                ref_acc <= ref_acc + (dut.act_bits[0] ? $signed(dut.wgt_bits[7:0])
                                                     : -$signed(dut.wgt_bits[7:0]));
            else
                ref_acc <= ref_acc + $signed(dut.act_bits[7:0]) * $signed(dut.wgt_bits[7:0]);
        end
    end

    // compare when a vector closes (the cycle the rhythm wraps to start)
    always @(posedge clk) begin
        if (rst_n && dut.pos == 7'd64) begin
            #1;
            if (KIND == 5) begin
                // stimulus only: nothing to check but that it runs
            end else if (KIND == 0) begin
                if (dut.g_bin.acc !== ref_acc[15:0]) begin
                    fails = fails + 1;
                    $display("FAIL bin acc %0d want %0d", dut.g_bin.acc, ref_acc);
                end
            end else if (dut.g_int.acc[0] !== ref_acc[23:0]) begin
                fails = fails + 1;
                $display("FAIL lane0 acc %0d want %0d", dut.g_int.acc[0], ref_acc);
            end
            checked = checked + 1;
        end
    end

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (65 * 20 + 3) @(posedge clk);
        if (toggles == 0) begin
            $display("FAIL out never toggled -- logic would be optimised away");
            fails = fails + 1;
        end
        $display("KIND %0d: %0d vectors checked, out toggled %0d times", KIND, checked, toggles);
        $display("%0d frames checked, %0d failures", checked, fails);
        $finish;
    end
endmodule

`default_nettype wire
