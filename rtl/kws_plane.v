// One activation plane: store a layer's output, then stream it into the next.
//
// This is the testbench driver turned into hardware. Everything the loop in
// tb_block.v does -- hold the frames, push them one at a time, wait out the
// consumer's busy, append the flushes its line buffers need -- happens here.
//
// WHY A PLANE AT ALL (docs/diagrams/28_plane_buffer.svg). Streaming is right
// INSIDE a layer and wrong between them: every junction needs its arrival times
// matched by hand, and kws_block cost two bugs doing that for one junction.
// A plane makes "six frames ago" an address instead of something to hold, so
// there is no timing to match -- layer L finishes writing before layer L+1
// starts reading, and ping-pong alternates two of these.
//
// THE FLUSH COUNT IS THE CONSUMER'S PROPERTY, NOT THIS MODULE'S. A depthwise
// stage lags its output by PAD frames, so a consumer with one costs PAD flush
// pushes and kws_block, with two in series, costs 2*PAD. Passed in as FLUSH
// rather than derived, because this module cannot see inside its consumer --
// and it is a static number from the manifest, not a runtime schedule
// (docs/diagrams/29_drain_propagate.svg).
//
// GAP CYCLES AFTER EACH PUSH. The consumer's `busy` is registered, so it is not
// high in the cycle right after a push. Sampling rd_ready immediately would
// read stale low and push again into a consumer that is about to start work --
// the same class of mistake as the one-cycle hole in kws_tcs_sub's busy. Two
// idle cycles per frame out of ~1600 settles it.

`timescale 1ns/1ps
`default_nettype none

module kws_plane #(
    parameter integer C     = 128,   // channels per frame
    parameter integer T     = 64,    // frames in a plane
    parameter integer FLUSH = 12     // drain pushes the CONSUMER needs
) (
    input  wire            clk,
    input  wire            rst_n,

    // ---- write side: the producing layer's output ---------------------- //
    input  wire            wr_start,   // begin filling (clears the pointer)
    input  wire            wr_valid,
    input  wire [C-1:0]    wr_frame,
    output wire            wr_full,

    // ---- read side: drives the consuming layer ------------------------ //
    input  wire            rd_start,   // begin streaming out
    input  wire            rd_ready,   // consumer's !busy
    output reg             rd_push,
    output reg             rd_real,
    output reg  [C-1:0]    rd_frame,
    output reg             rd_done
);

    localparam integer TOTAL   = T + FLUSH;
    localparam integer AW      = (T <= 2) ? 1 : $clog2(T);
    // one wider than the frame count, so the counter can reach TOTAL without
    // wrapping and every comparison below has room
    localparam integer CW_RAW  = (TOTAL <= 2) ? 1 : $clog2(TOTAL + 1);
    localparam integer CW      = (CW_RAW > AW) ? CW_RAW : AW + 1;

    localparam integer T_I     = T;
    localparam [CW-1:0] T_C     = T_I[CW-1:0];
    localparam integer TOTAL_I = TOTAL;
    localparam [CW-1:0] TOTAL_C = TOTAL_I[CW-1:0];

    // Synchronous read, which is what infers BRAM rather than a wall of flops:
    // C*T is 8192 bits at 128x64, and as registers that is 8192 flip-flops for
    // something a single block RAM holds.
    reg [C-1:0] mem [0:T-1];

    reg [CW-1:0] wp;                 // write pointer
    assign wr_full = (wp >= T_C);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          wp <= {CW{1'b0}};
        else if (wr_start)   wp <= {CW{1'b0}};
        else if (wr_valid && !wr_full) begin
            mem[wp[AW-1:0]] <= wr_frame;
            wp <= wp + {{(CW-1){1'b0}}, 1'b1};
        end
    end

    // ---- read FSM ------------------------------------------------------- //
    localparam [2:0] S_IDLE = 3'd0, S_WAIT = 3'd1, S_FETCH = 3'd2,
                     S_PUSH = 3'd3, S_GAP1 = 3'd4, S_GAP2 = 3'd5;
    reg [2:0]    st;
    reg [CW-1:0] rp;                 // which frame is being handed over
    reg [C-1:0]  rdata;

    wire is_real = (rp < T_C);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st       <= S_IDLE;
            rp       <= {CW{1'b0}};
            rdata    <= {C{1'b0}};
            rd_push  <= 1'b0;
            rd_real  <= 1'b0;
            rd_frame <= {C{1'b0}};
            rd_done  <= 1'b0;
        end else begin
            rd_push <= 1'b0;
            rd_done <= 1'b0;
            case (st)
            S_IDLE:
                if (rd_start) begin
                    rp <= {CW{1'b0}};
                    st <= S_WAIT;
                end
            S_WAIT:
                if (rd_ready) st <= S_FETCH;
            S_FETCH: begin
                // the flush tail has nothing stored; a zero frame is pushed
                // with rd_real low so the consumer masks it out anyway
                rdata <= is_real ? mem[rp[AW-1:0]] : {C{1'b0}};
                st    <= S_PUSH;
            end
            S_PUSH: begin
                rd_frame <= rdata;
                rd_real  <= is_real;
                rd_push  <= 1'b1;
                st       <= S_GAP1;
            end
            S_GAP1: st <= S_GAP2;    // consumer's busy is registered; let it rise
            S_GAP2: begin
                if (rp + {{(CW-1){1'b0}}, 1'b1} >= TOTAL_C) begin
                    rd_done <= 1'b1;
                    st      <= S_IDLE;
                end else begin
                    rp <= rp + {{(CW-1){1'b0}}, 1'b1};
                    st <= S_WAIT;
                end
            end
            default: st <= S_IDLE;
            endcase
        end
    end

`ifdef KWS_ASSERT
    // Reading a plane that was never filled means the producer and the schedule
    // disagree, and the symptom would be plausible garbage rather than an error.
    always @(posedge clk) if (rd_start && !wr_full) begin
        $display("ASSERT %m: read started with only %0d of %0d frames written",
                 wp, T);
        $finish;
    end
    // Ping-pong depends on these never overlapping on one instance.
    always @(posedge clk) if (wr_valid && (st != S_IDLE)) begin
        $display("ASSERT %m: written while streaming out");
        $finish;
    end
    always @(posedge clk) if (rd_push && !rd_ready) begin
        $display("ASSERT %m: pushed while the consumer is busy");
        $finish;
    end
`endif

endmodule

`default_nettype wire
