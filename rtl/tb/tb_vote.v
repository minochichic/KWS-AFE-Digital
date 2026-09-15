// Unit test for the exported-margin + consecutive-vote decision policy.

`timescale 1ns/1ps
`default_nettype none

module tb_vote;

    localparam integer MARGIN = 10;
    localparam integer N      = 3;
    localparam integer CD     = 4;

    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    reg score_valid = 1'b0, window_gap = 1'b0;
    reg [3:0] keyword_idx = 4'd0;
    reg signed [7:0] keyword_margin = 8'sd0;
    wire detection_valid, margin_accept, armed;
    wire [3:0] detection_idx;

    kws_vote #(
        .CLASS_BITS(4), .MARGIN_BITS(8), .MARGIN_INT(MARGIN),
        .REQUIRED(N), .COOLDOWN_WINDOWS(CD)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .score_valid(score_valid), .keyword_idx(keyword_idx),
        .keyword_margin(keyword_margin), .window_gap(window_gap),
        .detection_valid(detection_valid), .detection_idx(detection_idx),
        .margin_accept(margin_accept), .armed(armed)
    );

    integer detections = 0, errors = 0;
    reg [3:0] last_detection = 4'd0;

    always @(posedge clk) begin
        if (detection_valid) begin
            detections     <= detections + 1;
            last_detection <= detection_idx;
        end
    end

    task step;
        input [3:0] idx;
        input signed [7:0] margin;
        begin
            keyword_idx    = idx;
            keyword_margin = margin;
            score_valid    = 1'b1;
            @(negedge clk);
            score_valid    = 1'b0;
            @(negedge clk);
        end
    endtask

    task reset_dut;
        begin
            rst_n = 1'b0;
            repeat (2) @(negedge clk);
            rst_n = 1'b1;
            repeat (2) @(negedge clk);
        end
    endtask

    initial begin
        $dumpfile("tb_vote.vcd");
        $dumpvars(0, tb_vote);

        reset_dut();

        // Rejection breaks two correct votes; equality with MARGIN passes.
        step(4'd2, 8'sd20);
        step(4'd2, 8'sd20);
        step(4'd2, 8'sd9);
        step(4'd2, 8'sd10);
        step(4'd2, 8'sd20);
        step(4'd2, 8'sd20);
        if (detections !== 1 || last_detection !== 4'd2) begin
            $display("FAIL margin/streak detection count=%0d idx=%0d",
                     detections, last_detection);
            errors = errors + 1;
        end

        // Cooldown alone is insufficient; a rejected/quiet result is also
        // required. Once both hold, the expiry-cycle keyword counts as vote 1.
        step(4'd2, 8'sd20);  // cooldown 3, not armed
        step(4'd2, 8'sd20);  // cooldown 2, not armed
        step(4'd2, 8'sd9);   // cooldown 1, quiet seen
        step(4'd3, 8'sd20);  // cooldown 0, re-arm and vote 1
        step(4'd3, 8'sd20);
        step(4'd3, 8'sd20);
        if (detections !== 2 || last_detection !== 4'd3) begin
            $display("FAIL re-arm detection count=%0d idx=%0d",
                     detections, last_detection);
            errors = errors + 1;
        end

        reset_dut();
        detections = 0;
        last_detection = 0;

        // A missing scheduled result breaks adjacency.
        step(4'd1, 8'sd20);
        step(4'd1, 8'sd20);
        window_gap = 1'b1;
        @(negedge clk);
        window_gap = 1'b0;
        step(4'd1, 8'sd20);
        if (detections !== 0) begin
            $display("FAIL gap did not break streak");
            errors = errors + 1;
        end
        step(4'd1, 8'sd20);
        step(4'd1, 8'sd20);
        if (detections !== 1 || last_detection !== 4'd1) begin
            $display("FAIL post-gap detection count=%0d idx=%0d",
                     detections, last_detection);
            errors = errors + 1;
        end

        $display("\n3 vote scenarios checked, %0d failures", errors);
        $finish;
    end

    initial begin
        #100_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule

`default_nettype wire
