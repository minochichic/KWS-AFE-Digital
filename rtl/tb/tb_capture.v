// Unit test for the free-running comparator capture primitive.
//
// This test deliberately has no `start` signal. It proves that capture can run
// continuously whenever enable is high, and that disabling it discards a
// partial frame before the next enable interval begins.

`timescale 1ns/1ps
`default_nettype none

module tb_capture;

    localparam integer N_CH = 4;
    localparam integer FC   = 12;

    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    reg             enable = 1'b0;
    reg  [N_CH-1:0] cmp = {N_CH{1'b0}};
    wire            frame_valid;
    wire [N_CH-1:0] frame;

    kws_capture #(
        .N_CH(N_CH), .FRAME_CYCLES(FC), .CMP_INVERT(0)
    ) dut (
        .clk(clk), .rst_n(rst_n), .enable(enable), .cmp(cmp),
        .frame_valid(frame_valid), .frame(frame)
    );

    reg [N_CH-1:0] got [0:7];
    integer seen = 0;

    always @(posedge clk) begin
        if (frame_valid) begin
            got[seen] <= frame;
            seen      <= seen + 1;
        end
    end

    task drive_window;
        input [N_CH-1:0] bits;
        integer k;
        begin
            for (k = 0; k < FC; k = k + 1) begin
                // Keep the pulse clear of both edges. Two synchronizer clocks
                // still leave ample time for it to enter this frame.
                cmp = (k == 2) ? bits : {N_CH{1'b0}};
                @(negedge clk);
            end
            cmp = {N_CH{1'b0}};
        end
    endtask

    integer errors = 0;

    initial begin
        $dumpfile("tb_capture.vcd");
        $dumpvars(0, tb_capture);

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (3) @(negedge clk);

        // Comparator activity while disabled must neither emit a frame nor
        // leak into the next enabled interval.
        cmp = 4'b1111;
        repeat (4) @(negedge clk);
        cmp = 4'b0000;
        repeat (4) @(negedge clk);
        if (seen !== 0) begin
            $display("FAIL disabled capture emitted %0d frame(s)", seen);
            errors = errors + 1;
        end

        // No start pulse: enable alone produces back-to-back frames.
        enable = 1'b1;
        drive_window(4'b0101);
        drive_window(4'b1010);

        // Begin a third frame, then abort it. Its sticky bits must be cleared.
        cmp = 4'b1111;
        repeat (5) @(negedge clk);
        enable = 1'b0;
        cmp = 4'b0000;
        repeat (FC + 3) @(negedge clk);
        if (seen !== 2) begin
            $display("FAIL partial disabled frame changed count to %0d", seen);
            errors = errors + 1;
        end

        // Re-enable starts a fresh full window; aborted 1111 must not leak.
        enable = 1'b1;
        drive_window(4'b0011);
        enable = 1'b0;
        repeat (3) @(negedge clk);

        if (seen !== 3) begin
            $display("FAIL got %0d frames, expected 3", seen);
            errors = errors + 1;
        end else begin
            if (got[0] !== 4'b0101) begin
                $display("FAIL frame0 got %b want 0101", got[0]);
                errors = errors + 1;
            end
            if (got[1] !== 4'b1010) begin
                $display("FAIL frame1 got %b want 1010", got[1]);
                errors = errors + 1;
            end
            if (got[2] !== 4'b0011) begin
                $display("FAIL frame2 got %b want 0011", got[2]);
                errors = errors + 1;
            end
        end

        $display("\n3 frames checked, %0d failures", errors);
        $finish;
    end

    initial begin
        #100_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule

`default_nettype wire
