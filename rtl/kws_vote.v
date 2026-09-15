// Streaming keyword decision policy.
//
// Each neural-network result supplies the best keyword class and the signed
// difference between that keyword's pooled score and the best quiet score
// (silence/unknown). A result participates only when margin >= MARGIN_INT.
// REQUIRED adjacent accepted results must name the same keyword.
//
// After a detection, both conditions are required before re-arming:
//   1. COOLDOWN_WINDOWS classified windows have elapsed, and
//   2. at least one margin-rejected (quiet) result has been observed.

`timescale 1ns/1ps
`default_nettype none

module kws_vote #(
    parameter integer CLASS_BITS       = 4,
    parameter integer MARGIN_BITS      = 22,
    parameter integer MARGIN_INT       = 0,
    parameter integer REQUIRED         = 1,
    parameter integer COOLDOWN_WINDOWS = 0
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          score_valid,
    input  wire [CLASS_BITS-1:0]         keyword_idx,
    input  wire signed [MARGIN_BITS-1:0] keyword_margin,
    // Assert when a scheduled classified window was skipped. It immediately
    // breaks adjacency; cooldown remains conservative because no result was
    // consumed on the missing window.
    input  wire                          window_gap,

    output reg                           detection_valid,
    output reg  [CLASS_BITS-1:0]         detection_idx,
    output wire                          margin_accept,
    output reg                           armed
);

    localparam integer STREAK_BITS = (REQUIRED <= 1)
                                      ? 1 : $clog2(REQUIRED + 1);
    localparam integer CD_BITS = (COOLDOWN_WINDOWS <= 1)
                                  ? 1 : $clog2(COOLDOWN_WINDOWS + 1);
    localparam integer REQUIRED_I = REQUIRED;
    localparam integer COOLDOWN_I = COOLDOWN_WINDOWS;
    localparam [STREAK_BITS-1:0] REQUIRED_COUNT =
        REQUIRED_I[STREAK_BITS-1:0];
    localparam [CD_BITS-1:0] COOLDOWN_COUNT = COOLDOWN_I[CD_BITS-1:0];
    localparam signed [MARGIN_BITS-1:0] MARGIN_LIMIT = MARGIN_INT;

    reg [CLASS_BITS-1:0] candidate;
    reg [STREAK_BITS-1:0] streak;
    reg [CD_BITS-1:0] cooldown;
    reg quiet_seen;

    assign margin_accept = ($signed(keyword_margin) >= $signed(MARGIN_LIMIT));

    wire [CD_BITS-1:0] cooldown_after = (cooldown == {CD_BITS{1'b0}})
        ? {CD_BITS{1'b0}} : cooldown - 1'b1;
    wire quiet_after = quiet_seen || !margin_accept;
    // Python ConsecutiveVote re-arms before processing the current result. If
    // cooldown expires on an accepted result after a prior quiet result, that
    // accepted result is therefore streak element one.
    wire effective_armed = armed
        || (!armed && quiet_after && cooldown_after == {CD_BITS{1'b0}});

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            candidate       <= {CLASS_BITS{1'b0}};
            streak          <= {STREAK_BITS{1'b0}};
            cooldown        <= {CD_BITS{1'b0}};
            quiet_seen      <= 1'b1;
            armed           <= 1'b1;
            detection_valid <= 1'b0;
            detection_idx   <= {CLASS_BITS{1'b0}};
        end else begin
            detection_valid <= 1'b0;

            if (window_gap) begin
                candidate <= {CLASS_BITS{1'b0}};
                streak    <= {STREAK_BITS{1'b0}};
            end

            if (score_valid) begin
                cooldown <= cooldown_after;
                if (!armed && !margin_accept) quiet_seen <= 1'b1;
                if (!armed && quiet_after
                        && cooldown_after == {CD_BITS{1'b0}})
                    armed <= 1'b1;

                if (!margin_accept || !effective_armed || window_gap) begin
                    candidate <= {CLASS_BITS{1'b0}};
                    streak    <= {STREAK_BITS{1'b0}};
                end else if ((streak != {STREAK_BITS{1'b0}})
                             && keyword_idx == candidate) begin
                    if (streak + 1'b1 >= REQUIRED_COUNT) begin
                        detection_valid <= 1'b1;
                        detection_idx   <= keyword_idx;
                        candidate       <= {CLASS_BITS{1'b0}};
                        streak          <= {STREAK_BITS{1'b0}};
                        armed           <= 1'b0;
                        quiet_seen      <= 1'b0;
                        cooldown        <= COOLDOWN_COUNT;
                    end else begin
                        streak <= streak + 1'b1;
                    end
                end else if (REQUIRED == 1) begin
                    detection_valid <= 1'b1;
                    detection_idx   <= keyword_idx;
                    candidate       <= {CLASS_BITS{1'b0}};
                    streak          <= {STREAK_BITS{1'b0}};
                    armed           <= 1'b0;
                    quiet_seen      <= 1'b0;
                    cooldown        <= COOLDOWN_COUNT;
                end else begin
                    candidate <= keyword_idx;
                    streak    <= {{(STREAK_BITS-1){1'b0}}, 1'b1};
                end
            end
        end
    end

`ifdef KWS_ASSERT
    initial begin
        if (REQUIRED <= 0 || COOLDOWN_WINDOWS < 0) begin
            $display("ASSERT %m: REQUIRED must be positive and cooldown nonnegative");
            $finish;
        end
        if (MARGIN_INT < -(1 << (MARGIN_BITS-1))
                || MARGIN_INT >= (1 << (MARGIN_BITS-1))) begin
            $display("ASSERT %m: MARGIN_INT=%0d does not fit %0d signed bits",
                     MARGIN_INT, MARGIN_BITS);
            $finish;
        end
    end
`endif

endmodule

`default_nettype wire
