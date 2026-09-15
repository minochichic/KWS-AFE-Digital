// Continuous comparator capture and sliding-window replay.
//
// Comparator activity is reduced to one N_CH-bit frame every FRAME_CYCLES by
// kws_capture. The newest NATIVE_T frames live in a circular history buffer.
// Every TRIGGER_FRAMES captured frames, that history is copied into a stable
// snapshot and replayed as
//
//     PAD_LEFT zero frames + NATIVE_T captured frames + PAD_RIGHT zero frames.
//
// Capture never waits for replay or for the neural network. If a new automatic
// or forced request arrives while a snapshot is being copied/replayed, or while
// the downstream network is still busy, that request is skipped and
// overrun_count increments. The next periodic request remains on the original
// capture-frame cadence.

`timescale 1ns/1ps
`default_nettype none

module kws_window #(
    parameter integer N_CH           = 16,
    parameter integer FRAME_CYCLES   = 500000,
    parameter integer NATIVE_T       = 100,
    parameter integer PAD_LEFT       = 14,
    parameter integer PAD_RIGHT      = 14,
    parameter integer T              = 128,
    parameter integer TRIGGER_FRAMES = 1,
    parameter integer CMP_INVERT     = 0
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire [N_CH-1:0] cmp,
    // Synchronous one-cycle diagnostic/manual request. It uses the most recent
    // complete history and does not disturb the automatic trigger phase.
    input  wire            force_start,
    // A snapshot may launch only when the downstream network has completed
    // its previous clip. Replay can finish long before the folded network's
    // later phases, so local `busy` alone is not sufficient.
    input  wire            launch_ready,

    output wire            clip_start,
    output wire            out_valid,
    output reg  [N_CH-1:0] out_frame,
    input  wire            out_ready,
    output wire            busy,
    output reg             overrun,
    output reg  [7:0]      overrun_count
);

    wire            capture_valid;
    wire [N_CH-1:0] capture_frame;

    kws_capture #(
        .N_CH(N_CH), .FRAME_CYCLES(FRAME_CYCLES),
        .CMP_INVERT(CMP_INVERT)
    ) u_capture (
        .clk(clk), .rst_n(rst_n), .enable(1'b1), .cmp(cmp),
        .frame_valid(capture_valid), .frame(capture_frame)
    );

    localparam integer PTR_BITS = (NATIVE_T <= 2) ? 1 : $clog2(NATIVE_T);
    localparam integer HC_BITS  = (NATIVE_T <= 1) ? 1 : $clog2(NATIVE_T + 1);
    localparam integer HP_BITS  = (TRIGGER_FRAMES <= 2)
                                   ? 1 : $clog2(TRIGGER_FRAMES);
    localparam integer OUT_BITS = (T <= 2) ? 1 : $clog2(T);

    localparam integer PTR_LAST_I = NATIVE_T - 1;
    localparam integer HC_FULL_I  = NATIVE_T;
    localparam integer HC_PRE_I   = NATIVE_T - 1;
    localparam integer HOP_LAST_I = TRIGGER_FRAMES - 1;
    localparam integer OUT_LAST_I = T - 1;
    localparam integer REAL_END_I = PAD_LEFT + NATIVE_T;

    localparam [PTR_BITS-1:0] PTR_LAST = PTR_LAST_I[PTR_BITS-1:0];
    localparam [HC_BITS-1:0]  HC_FULL  = HC_FULL_I[HC_BITS-1:0];
    localparam [HC_BITS-1:0]  HC_PRE   = HC_PRE_I[HC_BITS-1:0];
    localparam [HP_BITS-1:0]  HOP_LAST = HOP_LAST_I[HP_BITS-1:0];
    localparam [OUT_BITS-1:0] OUT_LAST = OUT_LAST_I[OUT_BITS-1:0];

    reg [N_CH-1:0] history  [0:NATIVE_T-1];
    reg [N_CH-1:0] snapshot [0:NATIVE_T-1];

    reg [PTR_BITS-1:0] wr_ptr;
    reg [HC_BITS-1:0]  history_count;
    reg [HP_BITS-1:0]  hop_count;

    wire [PTR_BITS-1:0] wr_next = (wr_ptr == PTR_LAST)
                                        ? {PTR_BITS{1'b0}} : wr_ptr + 1'b1;
    wire history_full = (history_count == HC_FULL);
    // The capture closing this cycle can be the NATIVE_T-th history frame.
    // Launch that first complete history immediately, then count the selected
    // hop from that point. This remains correct when NATIVE_T is not an exact
    // multiple of TRIGGER_FRAMES.
    wire first_history_ready = capture_valid && !history_full
                                             && history_count == HC_PRE;
    wire history_ready_after_capture = history_full || first_history_ready;
    wire auto_request = first_history_ready
                     || (capture_valid && history_full
                                      && hop_count == HOP_LAST);
    wire request = auto_request || force_start;
    wire [PTR_BITS-1:0] oldest_after_capture = capture_valid ? wr_next : wr_ptr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr        <= {PTR_BITS{1'b0}};
            history_count <= {HC_BITS{1'b0}};
            hop_count     <= {HP_BITS{1'b0}};
        end else if (capture_valid) begin
            history[wr_ptr] <= capture_frame;
            wr_ptr          <= wr_next;
            if (!history_full) begin
                history_count <= history_count + 1'b1;
                hop_count     <= {HP_BITS{1'b0}};
            end else if (hop_count == HOP_LAST)
                hop_count <= {HP_BITS{1'b0}};
            else
                hop_count <= hop_count + 1'b1;
        end
    end

    localparam [2:0] S_IDLE   = 3'd0,
                     S_COPY   = 3'd1,
                     S_START  = 3'd2,
                     S_REPLAY = 3'd3;
    reg [2:0] st;

    reg [PTR_BITS-1:0] copy_src;
    reg [PTR_BITS-1:0] copy_idx;
    reg [OUT_BITS-1:0] replay_idx;

    wire [PTR_BITS-1:0] copy_src_next = (copy_src == PTR_LAST)
                                        ? {PTR_BITS{1'b0}} : copy_src + 1'b1;

    assign clip_start = (st == S_START);
    assign out_valid  = (st == S_REPLAY);
    assign busy       = (st != S_IDLE);

    always @* begin
        out_frame = {N_CH{1'b0}};
        if ((st == S_REPLAY)
                && (replay_idx >= PAD_LEFT)
                && (replay_idx < REAL_END_I)) begin
            out_frame = snapshot[replay_idx - PAD_LEFT];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st            <= S_IDLE;
            copy_src      <= {PTR_BITS{1'b0}};
            copy_idx      <= {PTR_BITS{1'b0}};
            replay_idx    <= {OUT_BITS{1'b0}};
            overrun       <= 1'b0;
            overrun_count <= 8'd0;
        end else begin
            overrun <= 1'b0;
            // Request acceptance is independent of capture bookkeeping above.
            // A request on the same edge as the final history write sees that
            // write on the following S_COPY cycle.
            if (request && history_ready_after_capture) begin
                if (st == S_IDLE && launch_ready) begin
                    copy_src <= oldest_after_capture;
                    copy_idx <= {PTR_BITS{1'b0}};
                    st       <= S_COPY;
                end else if (overrun_count != 8'hff) begin
                    overrun       <= 1'b1;
                    overrun_count <= overrun_count + 1'b1;
                end else begin
                    // Keep the event pulse alive even after the diagnostic
                    // counter saturates.
                    overrun <= 1'b1;
                end
            end

            case (st)
                S_COPY: begin
                    snapshot[copy_idx] <= history[copy_src];
                    if (copy_idx == PTR_LAST) begin
                        replay_idx <= {OUT_BITS{1'b0}};
                        st         <= S_START;
                    end else begin
                        copy_idx <= copy_idx + 1'b1;
                        copy_src <= copy_src_next;
                    end
                end

                S_START: begin
                    st <= S_REPLAY;
                end

                S_REPLAY: begin
                    if (out_ready) begin
                        if (replay_idx == OUT_LAST) begin
                            replay_idx <= {OUT_BITS{1'b0}};
                            st         <= S_IDLE;
                        end else begin
                            replay_idx <= replay_idx + 1'b1;
                        end
                    end
                end

                default: begin
                    // S_IDLE has no sequential work; capture keeps running in
                    // its independent always block.
                end
            endcase
        end
    end

`ifdef KWS_ASSERT
    initial begin
        if (NATIVE_T <= 0 || T != PAD_LEFT + NATIVE_T + PAD_RIGHT) begin
            $display("ASSERT %m: T=%0d must equal PAD_LEFT+NATIVE_T+PAD_RIGHT=%0d",
                     T, PAD_LEFT + NATIVE_T + PAD_RIGHT);
            $finish;
        end
        if (TRIGGER_FRAMES <= 0) begin
            $display("ASSERT %m: TRIGGER_FRAMES must be positive");
            $finish;
        end
        // The sequential snapshot copy must finish before the next capture can
        // overwrite another history location. Real hardware has 500,000 clocks
        // per frame and only 100 words to copy.
        if (FRAME_CYCLES <= NATIVE_T) begin
            $display("ASSERT %m: FRAME_CYCLES=%0d must exceed NATIVE_T=%0d",
                     FRAME_CYCLES, NATIVE_T);
            $finish;
        end
    end
`endif

endmodule

`default_nettype wire
