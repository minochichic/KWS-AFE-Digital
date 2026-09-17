// tb_stream_selftest: the streaming harness + deployable core on a few cases.
//
// Vectors : `KWS_SST_FRAME_FILE / `KWS_SST_EXP_FILE (export/slice_stream_selftest.py)
//
// One case is 300 frames. At the real FRAME_CYCLES (500,000) that is 150 M
// cycles, so the testbench uses a shorter frame: `KWS_SST_FRAME_CYCLES
// (default 320,000). The only constraint is that a 10-frame hop (3.2 M cycles)
// is longer than one network pass (~2.87 M cycles) plus the snapshot copy;
// otherwise windows are skipped and overrun_count rises -- which this bench
// reports as a failure, not a pass.
//
// Every window's keyword_idx / keyword_margin is printed so it can be diffed
// against stream_trace.csv; the pass line compares the full 40-bit case word.

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

`ifndef KWS_SST_FRAME_CYCLES
`define KWS_SST_FRAME_CYCLES 320000
`endif

module tb_stream_selftest;

    localparam integer CASES = `KWS_SST_CASES;
    localparam integer FC    = `KWS_SST_FRAME_CYCLES;

    reg clk = 1'b0, rst_n = 1'b0, go = 1'b0;
    always #10 clk = ~clk;

    wire        done, any_fail;
    wire [15:0] total, match, hit, wrong, outside, quiet_false, first_fail_case;
    wire [39:0] first_fail_got, first_fail_exp;
    wire [4:0]  window_now;
    wire [7:0]  overrun_count;

    kws_stream_selftest_top #(
        .CASES(CASES), .FRAME_CYCLES(FC),
        .FRAME_FILE(`KWS_SST_FRAME_FILE), .EXP_FILE(`KWS_SST_EXP_FILE)
    ) dut (
        .clk(clk), .rst_n(rst_n), .go(go),
        .done(done), .total(total), .match(match), .hit(hit), .wrong(wrong),
        .outside(outside), .quiet_false(quiet_false), .any_fail(any_fail),
        .first_fail_case(first_fail_case), .first_fail_got(first_fail_got),
        .first_fail_exp(first_fail_exp), .window_now(window_now),
        .overrun_count(overrun_count)
    );

    // per-window trace, in the column order of stream_trace.csv
    always @(posedge clk) begin
        if (dut.score_valid)
            $display("  case %0d window %0d idx %0d margin %0d",
                     dut.u_st.ci, dut.u_st.window_now, dut.keyword_idx,
                     $signed(dut.keyword_margin));
        if (dut.det_valid)
            $display("  case %0d DETECT class %0d on window %0d",
                     dut.u_st.ci, dut.det_idx, dut.u_st.window_now - 1);
        if (dut.u_st.st == 3'd3)
            $display("  case %0d got %010h exp %010h %s", dut.u_st.ci, dut.u_st.got,
                     dut.u_st.exp_q, dut.u_st.case_ok ? "match" : "MISMATCH");
    end

    // watchdog: 300 frames + 1 s of tail per case, times 2
    localparam integer LIMIT = CASES * (FC * 310) * 2;
    integer cyc = 0;
    always @(posedge clk) begin
        cyc = cyc + 1;
        if (cyc > LIMIT) begin
            $display("FAIL watchdog: total=%0d of %0d, window %0d", total, CASES, window_now);
            $finish;
        end
    end

    initial begin
`ifdef KWS_DUMP
        $dumpfile("tb_stream_selftest.vcd");
        $dumpvars(0, tb_stream_selftest);
`endif
        $display("== tb_stream_selftest: %0d cases, FRAME_CYCLES %0d ==", CASES, FC);
        repeat (5) @(negedge clk);
        rst_n = 1'b1;
        repeat (5) @(negedge clk);
        @(negedge clk) go = 1'b1;
        @(negedge clk) go = 1'b0;
        wait (done === 1'b1);
        repeat (2) @(negedge clk);

        $display("");
        $display("total %0d match %0d  hit %0d wrong %0d outside %0d quiet_false %0d  overrun %0d",
                 total, match, hit, wrong, outside, quiet_false, overrun_count);
        if (any_fail === 1'b1)
            $display("FAIL first mismatch case %0d: got %010h exp %010h",
                     first_fail_case, first_fail_got, first_fail_exp);
        else if (match !== CASES[15:0])
            $display("FAIL match %0d of %0d", match, CASES);
        else
            $display("PASS all %0d cases reproduce the integer streaming reference", CASES);
        // run_xsim.ps1 looks for ", 0 failures"
        $display("%0d cases checked, %0d failures", total, total - match);
        $finish;
    end

endmodule

`default_nettype wire
