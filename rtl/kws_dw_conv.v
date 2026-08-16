// Depthwise binary conv over time: one channel, K taps, no channel mixing.
//
// Reads a [C, T] plane stored frame-major (one word = one frame's channels,
// channel c in bit c). A depthwise tap run is horizontal in that picture, so it
// lands across K different words -- hence the line buffer: hold K frames in
// registers and the gather becomes combinational instead of K memory reads.
//
// PADDING IS NOT -1 (rtl/README.md 1). conv1d pads with 0, which contributes
// nothing, while every term of 2*popcount - N is assumed +-1. So a padded tap
// must leave N as well, and n_valid is per-frame at the edges.
//
// And n_valid alone is not enough: kws_bin_mac masks the LOW n_valid bits, but
// during fill the valid taps sit HIGH (t=0 uses taps 6..12). The valid run is
// always contiguous, so shifting activation AND weight together by the number
// of leading pads lands it at bit 0. Shifting only the activation pairs tap 6
// with weight 0 -- still a sum of +-1, so still a plausible number, wrong only
// at the edges.
//
// Both facts come out of one register: `valid`, a K-bit shift register running
// alongside the data. shift = trailing zeros, n_valid = popcount. No frame
// counter and no separate left/right edge arithmetic -- the two ends differ
// only in which side of `valid` the zeros sit on, and that falls out.
//
// TIMING. Output t needs input t+PAD, so the buffer runs PAD pushes ahead: PAD
// pushes produce nothing at the start, and PAD flush pushes drain the tail.
// See docs/diagrams/24_pipeline_drain.svg.

`timescale 1ns/1ps
`default_nettype none

module kws_dw_conv #(
    parameter integer C         = 128,   // channels
    parameter integer K         = 13,    // kernel taps  (K <= WORD_BITS)
    parameter integer PAD       = 6,     // (K-1)/2 for "same"
    parameter integer ACC_BITS  = 8,     // from parameters.vh
    parameter integer CNT_BITS  = 8,
    parameter integer WORD_BITS = 32,
    parameter W_FILE = "",               // C words, tap j in bit j
    parameter T_FILE = ""                // C thresholds, then C polarity words
) (
    input  wire              clk,
    input  wire              rst_n,

    input  wire              start,      // new clip: empty the buffer
    input  wire              in_push,    // advance one frame
    input  wire              in_real,    // 1 = in_frame is data, 0 = drain
    input  wire [C-1:0]      in_frame,

    output wire              busy,       // do not push while high
    output reg               out_valid,
    output reg  [C-1:0]      out_frame
);

    localparam integer CH_BITS = (C <= 2) ? 1 : $clog2(C);
    // Sized explicitly. `= C - 1` is a 32-bit expression landing in a
    // CH_BITS target, which is correct only because C-1 happens to fit --
    // a fact about today's C, not about the code.
    localparam integer          C_MINUS_1 = C - 1;
    localparam [CH_BITS-1:0]    LAST_CH   = C_MINUS_1[CH_BITS-1:0];
    // Threshold comparison width. 33, not 32, so BOTH sign-extensions below are
    // at least one bit -- a zero-width replication is not legal Verilog.
    localparam integer TH_BITS = (ACC_BITS + 1 > 33) ? ACC_BITS + 1 : 33;

    localparam [1:0] S_IDLE = 2'd0, S_START = 2'd1,
                     S_FEED = 2'd2, S_TAKE  = 2'd3;

    reg [1:0]         st;
    reg [CH_BITS-1:0] ch;
    assign busy = (st != S_IDLE);

    // ---- ROMs --------------------------------------------------------- //
    reg [WORD_BITS-1:0] w_rom [0:C-1];
    reg [31:0]          t_rom [0:2*C-1];
    initial begin
        if (W_FILE != "") $readmemh(W_FILE, w_rom);
        if (T_FILE != "") $readmemh(T_FILE, t_rom);
    end

    // ---- line buffer --------------------------------------------------- //
    // slot 0 is the oldest frame and is tap 0; a push moves everything down and
    // the new frame enters at slot K-1.
    reg [C-1:0] fbuf [0:K-1];
    reg [K-1:0] valid;

    wire [K-1:0] valid_next = {in_real, valid[K-1:1]};
    // The centre tap is slot K-1-PAD. An output exists exactly when that slot
    // holds a real frame: true after PAD pushes, false again PAD pushes past
    // the last real one. That single bit is the whole fill/drain condition.
    wire emit_now  = valid[K-1-PAD];
    wire emit_next = valid_next[K-1-PAD];

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid <= {K{1'b0}};
            for (i = 0; i < K; i = i + 1) fbuf[i] <= {C{1'b0}};
        end else if (start) begin
            valid <= {K{1'b0}};
            for (i = 0; i < K; i = i + 1) fbuf[i] <= {C{1'b0}};
        end else if (in_push && !busy) begin
            for (i = 0; i < K - 1; i = i + 1) fbuf[i] <= fbuf[i + 1];
            fbuf[K-1] <= in_frame;
            valid     <= valid_next;
        end
    end

    // ---- what `valid` tells us ---------------------------------------- //
    function [CNT_BITS-1:0] popcnt;
        input [K-1:0] v;
        integer j;
        begin
            popcnt = {CNT_BITS{1'b0}};
            for (j = 0; j < K; j = j + 1)
                popcnt = popcnt + {{(CNT_BITS-1){1'b0}}, v[j]};
        end
    endfunction

    // trailing zeros = pads sitting below the valid run = the shift
    function [CNT_BITS-1:0] tzc;
        input [K-1:0] v;
        integer j;
        reg hit;
        begin
            tzc = {CNT_BITS{1'b0}};
            hit = 1'b0;
            for (j = 0; j < K; j = j + 1) begin
                if (v[j]) hit = 1'b1;
                else if (!hit) tzc = tzc + {{(CNT_BITS-1){1'b0}}, 1'b1};
            end
        end
    endfunction

    wire [CNT_BITS-1:0] sh = tzc(valid);
    wire [CNT_BITS-1:0] nv = popcnt(valid);

    // ---- gather + align ------------------------------------------------ //
    // tap j of channel `ch` lives in slot j; all K slots are registers we
    // already hold, so this costs no cycles.
    // A generate loop rather than @(*) over the array: an implicit sensitivity
    // list on a memory makes the simulator sensitive to every word, which it
    // warns about, and per-bit assigns say exactly what this is.
    wire [K-1:0] taps;
    genvar gv;
    generate
        for (gv = 0; gv < K; gv = gv + 1) begin : g_taps
            assign taps[gv] = fbuf[gv][ch];
        end
    endgenerate

    // Take only the bits that mean something. The weight ROM holds K taps in a
    // 32-bit word and the polarity ROM one flag; reading the full words and
    // ignoring the rest reads as "these bits are dead", which is what lint
    // objected to and is worth stating rather than suppressing.
    wire [K-1:0]         w_taps  = w_rom[ch][K-1:0];
    wire signed [31:0]   thr     = t_rom[{1'b0, ch}];
    wire                 take_ge = t_rom[{1'b1, ch}][0];

    wire [WORD_BITS-1:0] taps_w = {{(WORD_BITS-K){1'b0}}, taps};
    wire [WORD_BITS-1:0] wgt_w  = {{(WORD_BITS-K){1'b0}}, w_taps};
    // the same shift on both, or the taps stop meeting their weights
    wire [WORD_BITS-1:0] act_sh = taps_w >> sh;
    wire [WORD_BITS-1:0] wgt_sh = wgt_w  >> sh;

    // ---- MAC ----------------------------------------------------------- //
    reg                        mac_start, mac_feed;
    wire                       mac_done;
    wire signed [ACC_BITS-1:0] mac_acc;

    kws_bin_mac #(.WORD_BITS(WORD_BITS), .ACC_BITS(ACC_BITS),
                  .CNT_BITS(CNT_BITS)) u_mac (
        .clk(clk), .rst_n(rst_n),
        .start(mac_start), .n_valid(nv),
        .in_valid(mac_feed), .act(act_sh), .wgt(wgt_sh),
        .out_valid(mac_done), .acc(mac_acc));

    // ---- threshold: sign(BN(alpha*n)) as one integer compare ----------- //
    wire signed [TH_BITS-1:0] acc_x =
        {{(TH_BITS-ACC_BITS){mac_acc[ACC_BITS-1]}}, mac_acc};
    wire signed [TH_BITS-1:0] thr_x = {{(TH_BITS-32){thr[31]}}, thr};
    wire                      fired = ((acc_x >= thr_x) == take_ge);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st        <= S_IDLE;
            ch        <= {CH_BITS{1'b0}};
            out_valid <= 1'b0;
            out_frame <= {C{1'b0}};
            mac_start <= 1'b0;
            mac_feed  <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            mac_start <= 1'b0;
            mac_feed  <= 1'b0;
            case (st)
            S_IDLE:
                if (start) begin
                    ch <= {CH_BITS{1'b0}};
                end else if (in_push && emit_next) begin
                    ch <= {CH_BITS{1'b0}};
                    st <= S_START;
                end
            S_START: begin
                mac_start <= 1'b1;
                st        <= S_FEED;
            end
            S_FEED: begin
                mac_feed <= 1'b1;          // K <= WORD_BITS: one word is all
                st       <= S_TAKE;
            end
            S_TAKE:
                if (mac_done) begin
                    out_frame[ch] <= fired;
                    if (ch == LAST_CH) begin
                        st        <= S_IDLE;
                        out_valid <= 1'b1;
                    end else begin
                        ch <= ch + {{(CH_BITS-1){1'b0}}, 1'b1};
                        st <= S_START;
                    end
                end
            default: st <= S_IDLE;
            endcase
        end
    end

`ifdef KWS_ASSERT
    always @(posedge clk) if (in_push && busy) begin
        $display("ASSERT %m: pushed while busy -- that frame would be dropped");
        $finish;
    end
    // `valid` must be one contiguous run of ones, or the shift means nothing.
    // Fill puts the zeros low and drain puts them high; anything else says the
    // caller interleaved real and flush pushes.
    wire [K-1:0] v_sh  = valid >> sh;
    wire [K-1:0] v_exp = ~({K{1'b1}} << nv);
    always @(posedge clk) if (emit_now) begin
        if (v_sh != v_exp) begin
            $display("ASSERT %m: valid=%b is not one contiguous run", valid);
            $finish;
        end
    end
`endif

endmodule

`default_nettype wire
