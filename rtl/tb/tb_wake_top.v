// tb_wake_top -- the STM32 -> FPGA wake link, end to end, against golden vectors.
//
//   ./rtl/run_tb.sh wake_top bd_base rtl/gen/bd_base/selftest
//
// Plays the STM32's part bit by bit on uart_rx: packets built from the
// export's input.hex (the 100 native frames of each clip; the testbench checks
// that the 14 + 14 pad frames there are all-zero, i.e. -1, which is what the
// link adds back). Decodes uart_tx and checks every result packet's CRC and
// class against predictions_fixed.txt -- the same oracle the chip self-test
// uses (docs/selftest_report.md 1).
//
// Before the good clips, the faults the link must survive:
//   1. a packet with a wrong CRC         -> err, no inference
//   2. a packet with N != NATIVE_T       -> err, no inference
//   3. half a packet, then WAKE          -> parser resyncs, next packet is fine
// No inference may start for 1-3: the first result must be clip 0's.
//
// The first TB_CLIPS (default 3) golden clips; +define+TB_CLIPS=N for more.

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

`ifndef TB_CLIPS
`define TB_CLIPS 3
`endif
`ifndef TB_BAUD
`define TB_BAUD 115200
`endif

module tb_wake_top;

    localparam integer CLK_HZ  = 50000000;
    localparam integer BAUD    = `TB_BAUD;
    localparam integer CPB     = CLK_HZ / BAUD;
    localparam integer T       = `KWS_T;
    localparam integer NAT     = `KWS_NATIVE_T;
    localparam integer PADL    = `KWS_PAD_LEFT;
    localparam integer CLIPS   = `KWS_GOLD_CLIPS;
    // each clip costs ~2.9 M network cycles (minutes in iverilog), so this
    // runs the first TB_CLIPS of them and SAYS SO in the pass line -- the full
    // set is the chip self-test's job (docs/selftest_report.md)
    localparam integer RUN     = (`TB_CLIPS < CLIPS) ? `TB_CLIPS : CLIPS;

    reg  clk = 1'b0, rst_n = 1'b0, uart_rx = 1'b1, wake = 1'b0;
    wire uart_tx, busy, class_valid, link_err;
    wire [3:0] class_idx;

    always #10 clk = ~clk;                    // 50 MHz

    kws_wake_top #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) dut (
        .clk(clk), .rst_n(rst_n), .uart_rx(uart_rx), .uart_tx(uart_tx),
        .wake(wake), .busy(busy), .class_valid(class_valid),
        .class_idx(class_idx), .link_err(link_err));

    // ---- golden vectors ------------------------------------------------------ //
    reg [31:0] frames [0:CLIPS*T-1];
    integer    want   [0:CLIPS-1];
    integer    fails = 0, checked = 0;

    // ---- reference CRC (the same algorithm, written independently) --------- //
    function [7:0] crc8(input [7:0] c, input [7:0] d);
        integer i;
        reg [7:0] x;
        begin
            x = c ^ d;
            for (i = 0; i < 8; i = i + 1)
                x = x[7] ? {x[6:0], 1'b0} ^ 8'h07 : {x[6:0], 1'b0};
            crc8 = x;
        end
    endfunction

    // ---- STM32 side: send ----------------------------------------------------- //
    task send_byte(input [7:0] b);
        integer i;
        begin
            uart_rx = 1'b0; repeat (CPB) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin
                uart_rx = b[i]; repeat (CPB) @(posedge clk);
            end
            uart_rx = 1'b1; repeat (CPB) @(posedge clk);
        end
    endtask

    // n_bytes < full length sends a truncated packet
    task send_packet(input integer clip, input [7:0] n, input bad_crc,
                     input integer n_frames);
        integer f;
        reg [7:0]  crc;
        reg [15:0] w;
        begin
            crc = 8'd0;
            send_byte(8'hA5); crc = crc8(crc, 8'hA5);
            send_byte(n);     crc = crc8(crc, n);
            for (f = 0; f < n_frames; f = f + 1) begin
                w = frames[clip*T + PADL + f][15:0];
                send_byte(w[7:0]);  crc = crc8(crc, w[7:0]);
                send_byte(w[15:8]); crc = crc8(crc, w[15:8]);
            end
            if (n_frames == n) send_byte(bad_crc ? ~crc : crc);
        end
    endtask

    task pulse_wake;
        begin
            wake = 1'b1; repeat (20) @(posedge clk); wake = 1'b0;
            repeat (20) @(posedge clk);
        end
    endtask

    // ---- STM32 side: receive -------------------------------------------------- //
    reg [7:0] rb [0:7];
    integer   rn = 0, results = 0;
    integer   got_class, got_kw, got_margin;
    reg       got_ok;

    task recv_byte(output [7:0] b);
        integer i;
        begin
            @(negedge uart_tx);
            repeat (CPB / 2) @(posedge clk);
            if (uart_tx !== 1'b0) begin $display("FAIL rx start bit glitch at %0t", $time); fails = fails + 1; end
            for (i = 0; i < 8; i = i + 1) begin
                repeat (CPB) @(posedge clk);
                b[i] = uart_tx;
            end
            repeat (CPB) @(posedge clk);
            if (uart_tx !== 1'b1) begin $display("FAIL rx stop bit low at %0t", $time); fails = fails + 1; end
        end
    endtask

    initial begin : rx_loop
        reg [7:0] b, crc;
        integer i;
        forever begin
            for (i = 0; i < 8; i = i + 1) begin
                recv_byte(b);
                rb[i] = b;
            end
            crc = 8'd0;
            for (i = 0; i < 7; i = i + 1) crc = crc8(crc, rb[i]);
            got_ok     = (rb[0] == 8'h5A) && (rb[7] == crc);
            got_class  = rb[1];
            got_kw     = rb[2];
            got_margin = $signed({rb[6], rb[5], rb[4], rb[3]});
            results    = results + 1;
        end
    end

    task expect_result(input integer clip);
        integer start_n, guard;
        begin
            start_n = results;
            guard   = 0;
            while (results == start_n && guard < 6000000) begin
                @(posedge clk); guard = guard + 1;
            end
            checked = checked + 1;
            if (results == start_n) begin
                $display("FAIL clip %0d: no result packet", clip); fails = fails + 1;
            end else if (!got_ok) begin
                $display("FAIL clip %0d: result packet bad sync/CRC", clip); fails = fails + 1;
            end else if (got_class != want[clip]) begin
                $display("FAIL clip %0d: class %0d, want %0d", clip, got_class, want[clip]);
                fails = fails + 1;
            end else
                $display("ok   clip %0d: class %0d (keyword %0d, margin %0d) at %0t",
                         clip, got_class, got_kw, got_margin, $time);
        end
    endtask

    // ---- the run ---------------------------------------------------------------- //
    initial begin : main
        integer c, f, fd, v, r;
        reg [7:0] chk;
        // tie this testbench's CRC to the published one (CRC-8/SMBUS check
        // value), so DUT and testbench cannot agree on a wrong algorithm
        chk = 8'd0;
        for (c = 0; c < 9; c = c + 1) chk = crc8(chk, 8'h31 + c[7:0]);   // "123456789"
        if (chk != 8'hF4) begin
            $display("FAIL crc8(\"123456789\") = %02x, want f4", chk); fails = fails + 1;
        end
        $readmemh(`KWS_GOLD_INPUT, frames);
        fd = $fopen(`KWS_GOLD_PREDICTIONS_FIXED, "r");
        if (fd == 0) begin $display("FAIL cannot open predictions_fixed"); $finish; end
        for (c = 0; c < CLIPS; c = c + 1) begin
            r = $fscanf(fd, "%d\n", v); want[c] = v;
        end
        $fclose(fd);
        // the link adds the padding back, so it had better be -1 (zero) in the
        // vectors -- otherwise this test would pass while dropping real data
        for (c = 0; c < RUN; c = c + 1)
            for (f = 0; f < T; f = f + 1)
                if ((f < PADL || f >= PADL + NAT) && frames[c*T + f] != 32'd0) begin
                    $display("FAIL clip %0d frame %0d is padding but not -1", c, f);
                    fails = fails + 1;
                end

        $display("== tb_wake_top: first %0d of %0d clips, %0d baud (%0d clk/bit), N %0d, T %0d ==",
                 RUN, CLIPS, BAUD, CPB, NAT, T);
        repeat (10) @(posedge clk); rst_n = 1'b1; repeat (10) @(posedge clk);

        // 1. bad CRC
        pulse_wake;
        send_packet(0, NAT[7:0], 1'b1, NAT);
        repeat (2000) @(posedge clk);
        if (!link_err) begin $display("FAIL bad CRC not flagged"); fails = fails + 1; end
        else $display("ok   bad CRC -> link_err");

        // 2. wrong N (the length byte is checked before any payload)
        pulse_wake;
        send_packet(0, NAT[7:0] - 8'd1, 1'b0, 0);
        repeat (200) @(posedge clk);
        if (!link_err) begin $display("FAIL wrong N not flagged"); fails = fails + 1; end
        else $display("ok   wrong N -> link_err");

        // 3. half a packet, then WAKE resyncs
        pulse_wake;
        send_packet(0, NAT[7:0], 1'b0, 10);
        pulse_wake;

        if (busy) begin $display("FAIL network started on a rejected packet"); fails = fails + 1; end

        // good clips, one result each, in order
        for (c = 0; c < RUN; c = c + 1) begin
            if (results != c) begin
                $display("FAIL %0d result packets before clip %0d", results, c);
                fails = fails + 1;
            end
            send_packet(c, NAT[7:0], 1'b0, NAT);
            expect_result(c);
        end
        if (link_err) begin $display("FAIL link_err still set after good packets"); fails = fails + 1; end

        $display("");
        $display("first %0d of %0d golden clips (TB_CLIPS)", RUN, CLIPS);
        $display("%0d clips checked, %0d failures", checked, fails);
        $finish;
    end

endmodule

`default_nettype wire
