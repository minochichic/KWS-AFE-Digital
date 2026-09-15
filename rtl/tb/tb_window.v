// Sliding-window unit test.
//
// Twelve captured frames produce three overlapping histories:
//   clip 0 = 1..8, clip 1 = 3..10, clip 2 = 5..12.
// Later frames arrive while earlier clips are being processed, so passing
// clips 1 and 2 also proves capture continues while the consumer is busy.

`timescale 1ns/1ps
`default_nettype none

module tb_window;

    localparam integer N_CH      = 4;
    // Deliberately not divisible by HOP: the first complete history must
    // launch on frame 7, then continue every two frames.
    localparam integer NATIVE_T  = 7;
    localparam integer PAD_LEFT  = 2;
    localparam integer PAD_RIGHT = 2;
    localparam integer T         = 11;
    localparam integer FC        = 12;
    localparam integer HOP       = 4;

    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    reg [N_CH-1:0] cmp = {N_CH{1'b0}};
    reg force_start = 1'b0;
    reg out_ready = 1'b1;
    reg launch_ready = 1'b1;
    wire clip_start, out_valid, busy, overrun;
    wire [N_CH-1:0] out_frame;
    wire [7:0] overrun_count;

    kws_window #(
        .N_CH(N_CH), .FRAME_CYCLES(FC), .NATIVE_T(NATIVE_T),
        .PAD_LEFT(PAD_LEFT), .PAD_RIGHT(PAD_RIGHT), .T(T),
        .TRIGGER_FRAMES(HOP), .CMP_INVERT(0)
    ) dut (
        .clk(clk), .rst_n(rst_n), .cmp(cmp),
        .force_start(force_start), .launch_ready(launch_ready),
        .clip_start(clip_start),
        .out_valid(out_valid), .out_frame(out_frame),
        .out_ready(out_ready), .busy(busy), .overrun(overrun),
        .overrun_count(overrun_count)
    );

    reg [N_CH-1:0] got [0:4*T-1];
    integer starts = 0, transferred = 0, overruns = 0;

    always @(posedge clk) begin
        if (clip_start) starts <= starts + 1;
        if (overrun) overruns <= overruns + 1;
        if (out_valid && out_ready) begin
            got[transferred] <= out_frame;
            transferred      <= transferred + 1;
        end
    end

    task drive_capture_frame;
        input [N_CH-1:0] bits;
        integer k;
        begin
            for (k = 0; k < FC; k = k + 1) begin
                cmp = (k == 2) ? bits : {N_CH{1'b0}};
                @(negedge clk);
            end
            cmp = {N_CH{1'b0}};
        end
    endtask

    function [N_CH-1:0] expected;
        input integer clip;
        input integer t;
        integer first;
        begin
            first = 1 + clip * HOP;
            if (t < PAD_LEFT || t >= PAD_LEFT + NATIVE_T)
                expected = {N_CH{1'b0}};
            else
                expected = first + t - PAD_LEFT;
        end
    endfunction

    integer errors = 0, i, clip, t, timeout;

    initial begin
        $dumpfile("tb_window.vcd");
        $dumpvars(0, tb_window);

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (3) @(negedge clk);

        for (i = 1; i <= 15; i = i + 1)
            drive_capture_frame(i[N_CH-1:0]);

        timeout = 0;
        while (transferred < 3*T && timeout < 300) begin
            @(negedge clk);
            timeout = timeout + 1;
        end

        if (starts !== 3) begin
            $display("FAIL got %0d clip_start pulses, expected 3", starts);
            errors = errors + 1;
        end
        if (transferred !== 3*T) begin
            $display("FAIL got %0d output frames, expected %0d", transferred, 3*T);
            errors = errors + 1;
        end

        for (clip = 0; clip < 3; clip = clip + 1) begin
            for (t = 0; t < T; t = t + 1) begin
                if (got[clip*T+t] !== expected(clip, t)) begin
                    errors = errors + 1;
                    if (errors <= 8)
                        $display("FAIL clip%0d t=%0d got %h want %h",
                                 clip, t, got[clip*T+t], expected(clip, t));
                end
            end
        end

        // A forced request is accepted while idle. A second pulse during its
        // copy/replay is skipped and counted exactly once.
        while (busy) @(negedge clk);
        force_start = 1'b1;
        @(negedge clk);
        force_start = 1'b0;
        // Give the DUT one sampled-low cycle so these are two pulses rather
        // than one request held high for two clocks.
        @(negedge clk);
        if (!busy) begin
            $display("FAIL forced request did not enter busy state");
            errors = errors + 1;
        end
        force_start = 1'b1;
        @(negedge clk);
        force_start = 1'b0;
        while (busy) @(negedge clk);
        repeat (2) @(negedge clk);

        if (overrun_count !== 8'd1 || overruns !== 1) begin
            $display("FAIL overrun count/pulses got %0d/%0d want 1/1",
                     overrun_count, overruns);
            errors = errors + 1;
        end

        // Window replay may be idle while the folded network is still in a
        // later phase. Such a request must not restart that network.
        launch_ready = 1'b0;
        force_start = 1'b1;
        @(negedge clk);
        force_start = 1'b0;
        @(negedge clk);
        if (busy || overrun_count !== 8'd2 || overruns !== 2) begin
            $display("FAIL launch interlock busy=%b count/pulses=%0d/%0d want 0/2/2",
                     busy, overrun_count, overruns);
            errors = errors + 1;
        end

        $display("\n%0d frames checked, %0d failures", 3*T, errors);
        $finish;
    end

    initial begin
        #100_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule

`default_nettype wire
