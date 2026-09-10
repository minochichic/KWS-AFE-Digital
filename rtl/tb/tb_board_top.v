// Board-level end-to-end test:
//
//   asynchronous comparator pulses -> frame controller -> neural network
//                                  -> registered class output
//
// Each asserted input bit is presented as a one-clock pulse at a channel-
// specific offset. This checks the synchronizers and sticky-OR behavior instead
// of merely holding the complete frame on cmp. Every frame delivered to the
// network is compared with golden/input.hex, and the final class is compared
// with predictions_fixed.txt.

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module tb_board_top;

    localparam integer CLIPS    = `KWS_GOLD_CLIPS;
    localparam integer N_CH     = `KWS_N_CH;
    localparam integer T        = `KWS_T;
    localparam integer NATIVE_T = `KWS_NATIVE_T;
    localparam integer PAD_L    = `KWS_PAD_LEFT;
    localparam integer NW       = (N_CH + `KWS_WORD_BITS - 1)
                                  / `KWS_WORD_BITS;

    // Conv1 needs 128 * 16 * 11 = 22,528 term cycles after every second
    // accepted frame. Keeping a simulated window above that value preserves
    // the real-time contract while avoiding the real 500,000-cycle window.
    localparam integer FRAME_CYCLES = 24000;
    localparam integer SIM_CLK_HZ =
        (FRAME_CYCLES / `KWS_FRAME_MS) * 1000;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg              rst_n = 1'b0;
    reg              start = 1'b0;
    reg  [N_CH-1:0]  cmp   = {N_CH{1'b0}};
    wire             busy;
    wire             class_valid;
    wire [3:0]       class_idx;

    kws_board_top #(
        .CLK_HZ    (SIM_CLK_HZ),
        .CMP_INVERT(0)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .cmp        (cmp),
        .busy       (busy),
        .class_valid(class_valid),
        .class_idx  (class_idx)
    );

    reg [`KWS_WORD_BITS-1:0] gold [0:CLIPS*T*NW-1];
    integer want_class [0:CLIPS-1];

    function [N_CH-1:0] want_frame;
        input integer n;
        input integer t;
        integer j;
        reg [`KWS_WORD_BITS*NW-1:0] words;
        begin
            words = {(`KWS_WORD_BITS*NW){1'b0}};
            for (j = 0; j < NW; j = j + 1)
                words[j*`KWS_WORD_BITS +: `KWS_WORD_BITS] =
                    gold[(n * T + t) * NW + j];
            want_frame = words[N_CH-1:0];
        end
    endfunction

    // Keep pulses away from both frame edges so the two synchronizer cycles
    // cannot move a pulse into the neighboring frame.
    function integer offset_of;
        input integer c;
        begin
            offset_of = 3 + ((c * 997) % (FRAME_CYCLES - 8));
        end
    endfunction

    integer errors = 0;
    integer active_clip = -1;
    integer frames_seen = 0;
    integer n, t, c, k, fh, code;
    reg [N_CH-1:0] requested;
    reg [3:0] got_class;
    reg       got_valid = 1'b0;

    // Check the exact tensor at the integration boundary. fc_valid is the
    // frame controller's accepted pulse, so this also counts all padding.
    always @(posedge clk) begin
        if (dut.fc_valid && dut.top_ready && active_clip >= 0) begin
            if (frames_seen >= T) begin
                $display("FAIL clip%0d: emitted more than %0d frames",
                         active_clip, T);
                errors = errors + 1;
            end else if (dut.fc_frame !== want_frame(active_clip, frames_seen)) begin
                $display("FAIL clip%0d frame%0d: got %h want %h",
                         active_clip, frames_seen, dut.fc_frame,
                         want_frame(active_clip, frames_seen));
                errors = errors + 1;
            end
            frames_seen = frames_seen + 1;
        end

        if (class_valid) begin
            got_class = class_idx;
            got_valid = 1'b1;
        end
    end

    initial begin
        $readmemh(`KWS_GOLD_INPUT, gold, 0, CLIPS * T * NW - 1);
        fh = $fopen(`KWS_GOLD_PREDICTIONS_FIXED, "r");
        if (fh == 0) begin
            $display("FAIL cannot open predictions_fixed.txt");
            $finish;
        end
        for (n = 0; n < CLIPS; n = n + 1) begin
            code = $fscanf(fh, "%d", want_class[n]);
            if (code != 1) begin
                $display("FAIL predictions_fixed.txt is short");
                $finish;
            end
        end
        $fclose(fh);

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        wait (dut.rst_n_s === 1'b1);
        repeat (2) @(negedge clk);

        for (n = 0; n < CLIPS; n = n + 1) begin
            active_clip = n;
            frames_seen = 0;
            got_valid   = 1'b0;

            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;

            // Left padding is back-pressured by Conv1. Begin the first real
            // window only after the frame controller has entered S_RUN.
            wait (dut.u_fc.st === 3'd2);
            @(negedge clk);

            for (t = 0; t < NATIVE_T; t = t + 1) begin
                requested = want_frame(n, PAD_L + t);
                for (k = 0; k < FRAME_CYCLES; k = k + 1) begin
                    cmp = {N_CH{1'b0}};
                    for (c = 0; c < N_CH; c = c + 1)
                        if (requested[c] && offset_of(c) == k)
                            cmp[c] = 1'b1;
                    @(negedge clk);
                end
            end
            cmp = {N_CH{1'b0}};

            wait (got_valid === 1'b1);
            while (busy) @(negedge clk);
            repeat (2) @(negedge clk);

            if (frames_seen != T) begin
                $display("FAIL clip%0d: emitted %0d frames, expected %0d",
                         n, frames_seen, T);
                errors = errors + 1;
            end
            if (got_class != want_class[n]) begin
                $display("FAIL clip%0d: got class %0d, want %0d",
                         n, got_class, want_class[n]);
                errors = errors + 1;
            end else begin
                $display("ok   clip%0d: %0d frames, class %0d",
                         n, frames_seen, got_class);
            end
        end

        active_clip = -1;
        $display("\n%0d clips checked, %0d failures", CLIPS, errors);
        $finish;
    end

    initial begin
        #500_000_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule

`default_nettype wire
