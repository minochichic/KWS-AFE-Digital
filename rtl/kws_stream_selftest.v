// Streaming self-test harness: replay stored comparator streams into the
// deployable always-on core and score every case on the chip.
//
//   frame ROM (cases x 300 frames x 16 bit) --> cmp[15:0] --> kws_stream_core
//                                                (capture, window, network, vote)
//   per window: {keyword_idx, keyword_margin} --> CRC-16
//   per detection: {window, class} --> two slots
//   per case: {crc, target, slot1, slot0} == expected ROM word ?
//
// The reference is experiments/eval_streaming_int.py, which mirrors kws_tail's
// keyword/margin and kws_vote register by register. Word layout and CRC are
// defined there (docs/alwayson_chip_plan.md).
//
// ---- frame alignment (the delicate part) ------------------------------------ //
//
// kws_capture passes cmp through two synchronizer flops and ORs everything seen
// between two frame_valid pulses. Its frame counter restarts when its reset is
// released. This harness owns that reset (core_rst_n) and runs its own counter
// `hc` from the same release, so hc == capture.fc on every cycle.
//
// cmp is changed on the edge where hc == FRAME_CYCLES-3. After the two
// synchronizer stages the new value first reaches the sticky OR on the cycle
// hc == 0 of the next capture window, and the old value is still what the
// closing edge (hc == FRAME_CYCLES-1) of the current window sees. No bit of one
// frame leaks into the next. Frame 0 is placed on cmp while the core is still in
// reset; the synchronizers read 0 for two cycles after release, and 0 cannot
// set a sticky bit, so frame 0 is exact too.
//
// Every case starts from reset, as the Python reference starts every stream
// with empty history and a fresh vote.

`timescale 1ns/1ps
`default_nettype none

module kws_stream_selftest #(
    parameter integer N_CH         = 16,
    parameter integer FRAME_CYCLES = 500000,
    parameter integer CASES        = 240,
    parameter integer FRAMES       = 300,   // per case
    parameter integer WINDOWS      = 21,    // per case
    parameter integer HOP          = 10,    // frames between windows
    parameter integer NATIVE_T     = 100,
    parameter integer N_KEYWORDS   = 10,
    parameter integer MARGIN_BITS  = 22,
    parameter         FRAME_FILE   = "",
    parameter         EXP_FILE     = ""
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          go,

    // to / from kws_stream_core
    output reg                           core_rst_n,
    output reg  [N_CH-1:0]               cmp,
    input  wire                          obs_score_valid,
    input  wire [3:0]                    obs_keyword_idx,
    input  wire signed [MARGIN_BITS-1:0] obs_keyword_margin,
    input  wire                          det_valid,
    input  wire [3:0]                    det_idx,
    input  wire [7:0]                    overrun_count,

    // results
    output reg                           done,
    output reg  [15:0]                   total,
    output reg  [15:0]                   match,
    output reg  [15:0]                   hit,
    output reg  [15:0]                   wrong,
    output reg  [15:0]                   outside,
    output reg  [15:0]                   quiet_false,
    output reg                           any_fail,
    output reg  [15:0]                   first_fail_case,
    output reg  [39:0]                   first_fail_got,
    output reg  [39:0]                   first_fail_exp,
    output reg  [4:0]                    window_now
);

    localparam integer ROM_WORDS = CASES * FRAMES;
    localparam integer FAW = (ROM_WORDS <= 2) ? 1 : $clog2(ROM_WORDS);
    localparam integer FCW = (FRAME_CYCLES <= 2) ? 1 : $clog2(FRAME_CYCLES);

    // ---- ROMs (synchronous read -> block RAM) ----
    (* rom_style = "block" *) reg [15:0] frame_rom [0:ROM_WORDS-1];
    (* rom_style = "block" *) reg [39:0] exp_rom   [0:CASES-1];
    initial begin
        if (FRAME_FILE != "") $readmemh(FRAME_FILE, frame_rom);
        if (EXP_FILE   != "") $readmemh(EXP_FILE,   exp_rom);
    end

    reg  [15:0]    ci;           // case
    reg  [15:0]    nf;           // next frame of this case to put on cmp
    wire [15:0]    nf_clamped = (nf < FRAMES) ? nf : FRAMES - 1;
    wire [FAW-1:0] frame_addr = ci * FRAMES + nf_clamped;
    reg  [15:0]    frame_q;
    reg  [39:0]    exp_q;
    always @(posedge clk) begin
        frame_q <= frame_rom[frame_addr];
        exp_q   <= exp_rom[ci];
    end

    // ---- CRC-16/CCITT-FALSE over {idx[3:0], margin[MARGIN_BITS-1:0]}, MSB first ----
    function [15:0] crc_word(input [15:0] c, input [3:0] k, input [MARGIN_BITS-1:0] m);
        integer i;
        reg [MARGIN_BITS+3:0] w;
        reg fb;
        begin
            w = {k, m};
            crc_word = c;
            for (i = MARGIN_BITS + 3; i >= 0; i = i - 1) begin
                fb = crc_word[15] ^ w[i];
                crc_word = {crc_word[14:0], 1'b0};
                if (fb) crc_word = crc_word ^ 16'h1021;
            end
        end
    endfunction

    // A detection on window w overlaps the inserted target clip [NATIVE_T, 2*NATIVE_T)
    function overlaps(input [4:0] w);
        overlaps = (w != 5'd0) && ((w * HOP) < (2 * NATIVE_T));
    endfunction

    localparam [2:0] S_IDLE  = 3'd0,
                     S_LOAD  = 3'd1,
                     S_RUN   = 3'd2,
                     S_CHECK = 3'd3,
                     S_DONE  = 3'd4;
    reg [2:0] st;
    reg [1:0] load_wait;

    reg [FCW-1:0] hc;
    reg [15:0]    crc;
    reg [9:0]     slot0, slot1;
    reg [1:0]     n_det;
    reg           extra_det;

    // ---- per-case outcome, from the chip's own detections ----
    wire [3:0]  target  = exp_q[23:20];
    wire [39:0] got     = {crc, target, slot1, slot0};
    wire        v0 = slot0[9], v1 = slot1[9];
    wire        c0 = v0 && overlaps(slot0[8:4]) && (target < N_KEYWORDS) && (slot0[3:0] == target);
    wire        c1 = v1 && overlaps(slot1[8:4]) && (target < N_KEYWORDS) && (slot1[3:0] == target);
    wire        o0 = v0 && !overlaps(slot0[8:4]);
    wire        o1 = v1 && !overlaps(slot1[8:4]);
    wire        case_ok = (got == exp_q) && !extra_det && (overrun_count == 8'd0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; core_rst_n <= 1'b0; cmp <= {N_CH{1'b0}};
            ci <= 16'd0; nf <= 16'd0; load_wait <= 2'd0; hc <= {FCW{1'b0}};
            crc <= 16'hFFFF; slot0 <= 10'd0; slot1 <= 10'd0; n_det <= 2'd0; extra_det <= 1'b0;
            window_now <= 5'd0;
            done <= 1'b0; total <= 16'd0; match <= 16'd0; hit <= 16'd0; wrong <= 16'd0;
            outside <= 16'd0; quiet_false <= 16'd0; any_fail <= 1'b0;
            first_fail_case <= 16'd0; first_fail_got <= 40'd0; first_fail_exp <= 40'd0;
        end else begin
            // capture-phase counter, released together with the core
            if (!core_rst_n)                    hc <= {FCW{1'b0}};
            else if (hc == FRAME_CYCLES - 1)    hc <= {FCW{1'b0}};
            else                                hc <= hc + 1'b1;

            case (st)
            S_IDLE:
                if (go) begin
                    ci <= 16'd0; done <= 1'b0; total <= 16'd0; match <= 16'd0;
                    hit <= 16'd0; wrong <= 16'd0; outside <= 16'd0; quiet_false <= 16'd0;
                    any_fail <= 1'b0;
                    st <= S_LOAD; load_wait <= 2'd0; nf <= 16'd0; core_rst_n <= 1'b0;
                end

            S_LOAD: begin
                // core held in reset; frame 0 settles on cmp before release
                core_rst_n <= 1'b0;
                crc <= 16'hFFFF; slot0 <= 10'd0; slot1 <= 10'd0; n_det <= 2'd0;
                extra_det <= 1'b0; window_now <= 5'd0;
                if (load_wait != 2'd3) begin
                    load_wait <= load_wait + 1'b1;        // address -> frame_q settle
                end else begin
                    cmp <= frame_q[N_CH-1:0];
                    nf  <= 16'd1;
                    core_rst_n <= 1'b1;
                    st <= S_RUN;
                end
            end

            S_RUN: begin
                if (core_rst_n && hc == FRAME_CYCLES - 3) begin
                    cmp <= (nf < FRAMES) ? frame_q[N_CH-1:0] : {N_CH{1'b0}};
                    if (nf < FRAMES) nf <= nf + 1'b1;
                end
                if (obs_score_valid) begin
                    crc <= crc_word(crc, obs_keyword_idx, obs_keyword_margin);
                    window_now <= window_now + 1'b1;
                end
                if (det_valid) begin
                    // det_valid follows its score_valid by one cycle, so the
                    // window index has already advanced
                    if (n_det == 2'd0)      slot0 <= {1'b1, window_now - 5'd1, det_idx};
                    else if (n_det == 2'd1) slot1 <= {1'b1, window_now - 5'd1, det_idx};
                    else                    extra_det <= 1'b1;
                    if (n_det != 2'd3) n_det <= n_det + 1'b1;
                end
                if (window_now == WINDOWS[4:0] && !obs_score_valid)
                    st <= S_CHECK;   // the cycle that can carry the last det_valid is this one
            end

            S_CHECK: begin
                core_rst_n <= 1'b0;
                total <= total + 1'b1;
                if (case_ok) begin
                    match <= match + 1'b1;
                end else if (!any_fail) begin
                    any_fail <= 1'b1;
                    first_fail_case <= ci;
                    first_fail_got  <= got;
                    first_fail_exp  <= exp_q;
                end
                if (target < N_KEYWORDS && (c0 || c1)) hit <= hit + 1'b1;
                if (target >= N_KEYWORDS && (v0 || v1)) quiet_false <= quiet_false + 1'b1;
                wrong   <= wrong + ((v0 && !o0 && !c0) ? 16'd1 : 16'd0)
                                 + ((v1 && !o1 && !c1) ? 16'd1 : 16'd0);
                outside <= outside + (o0 ? 16'd1 : 16'd0) + (o1 ? 16'd1 : 16'd0);
                if (ci == CASES - 1) begin
                    st <= S_DONE; done <= 1'b1;
                end else begin
                    ci <= ci + 1'b1; nf <= 16'd0; load_wait <= 2'd0; st <= S_LOAD;
                end
            end

            S_DONE:
                if (go) st <= S_IDLE;

            default: st <= S_IDLE;
            endcase
        end
    end

`ifdef KWS_ASSERT
    initial if (FRAME_CYCLES < 4 || WINDOWS > 31) begin
        $display("ASSERT %m: FRAME_CYCLES must be >= 4 and WINDOWS <= 31");
        $finish;
    end
    // the frame index must never run past the stream while windows are pending
    always @(posedge clk) if (st == S_RUN && obs_score_valid && window_now == WINDOWS[4:0]) begin
        $display("ASSERT %m: more than %0d windows scored in case %0d", WINDOWS, ci);
        $finish;
    end
`endif

endmodule

`default_nettype wire
