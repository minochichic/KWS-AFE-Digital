// Pointwise binary conv: one frame, all channels mixed, no time dimension.
//
// This is where bit packing actually pays. A pointwise output sums over every
// input channel, and the channels are exactly what a word holds (frame-major,
// rtl/README.md 2) -- so the activation vector IS the stored word and streams
// straight into the MAC. 128 channels are four words, four cycles, no gather
// and no transpose.
//
// Simpler than kws_dw_conv in every way that module was hard:
//   * k=1, so no line buffer and no tap run
//   * padding=0, so n_valid is C_IN on every frame and never varies
//   * nothing to shift, because there are no edges in the channel direction
// The edge machinery in dw exists because its taps run along TIME, which is
// finite. The channel axis is not a window; it is the whole vector.
//
// Weight ROM layout: pack_pm1([C_OUT, C_IN]) along C_IN, so output channel o
// owns words [o*NW .. o*NW+NW-1] and word j carries input channels 32j..32j+31.

`timescale 1ns/1ps
`default_nettype none

module kws_pw_conv #(
    parameter integer C_IN      = 128,
    parameter integer C_OUT     = 64,
    parameter integer ACC_BITS  = 9,     // from parameters.vh
    parameter integer CNT_BITS  = 16,
    parameter integer WORD_BITS = 32,
    parameter W_FILE = "",               // C_OUT*NW words
    parameter T_FILE = "",               // C_OUT thresholds, then C_OUT polarity
    // width of the channel index; a localparam cannot be used in a port
    // declaration, so it is a parameter with a derived default
    parameter integer CO_BITS_P = (C_OUT <= 2) ? 1 : $clog2(C_OUT)
) (
    input  wire              clk,
    input  wire              rst_n,

    input  wire              in_valid,   // one frame; held until busy drops
    input  wire [C_IN-1:0]   in_frame,

    output wire              busy,
    output reg               out_valid,
    output reg  [C_OUT-1:0]  out_frame,

    // Raw accumulator, streamed one channel at a time. The last pointwise of a
    // residual block does NOT end in a threshold (manifest epilogue "none") --
    // its integer accumulator is added to the skip's before a single threshold
    // is applied. Exposing it here rather than adding the residual arithmetic
    // to this module keeps that arithmetic where it belongs, in kws_block, and
    // leaves the thresholded path untouched for callers that want it.
    output reg                       acc_valid,
    output reg  [CO_BITS_P-1:0]      acc_ch,
    output reg  signed [ACC_BITS-1:0] acc_out
);

    localparam integer NW      = C_IN / WORD_BITS;   // words per frame
    localparam integer CO_BITS = CO_BITS_P;
    localparam integer NW_BITS = (NW    <= 2) ? 1 : $clog2(NW);
    // sized explicitly rather than letting a 32-bit expression fall into a
    // narrow target -- that only ever works by accident of today's parameters
    localparam integer          CO_M1_I  = C_OUT - 1;
    localparam [CO_BITS-1:0]    LAST_CO  = CO_M1_I[CO_BITS-1:0];
    localparam integer          NW_M1_I  = NW - 1;
    localparam [NW_BITS-1:0]    LAST_W   = NW_M1_I[NW_BITS-1:0];
    // 33, not 32, so both sign-extensions below are at least one bit wide
    localparam integer TH_BITS = (ACC_BITS + 1 > 33) ? ACC_BITS + 1 : 33;

    localparam [1:0] S_IDLE = 2'd0, S_START = 2'd1,
                     S_FEED = 2'd2, S_TAKE  = 2'd3;

    // The weight ROM is walked strictly in order -- output channel 0's words,
    // then channel 1's, and so on -- so a counter replaces `co * NW + wi`.
    // That address expression also mixed three different widths, which lint
    // objected to; there is nothing to widen if there is no arithmetic.
    localparam integer WA_RAW  = (C_OUT * NW <= 2) ? 1 : $clog2(C_OUT * NW);
    // at least one bit wider than either index, so nothing here can need a
    // zero-width replication
    localparam integer WA_BITS = (WA_RAW > CO_BITS)
                                 ? ((WA_RAW > NW_BITS) ? WA_RAW : NW_BITS + 1)
                                 : CO_BITS + 1;

    reg [1:0]          st;
    reg [CO_BITS-1:0]  co;          // which output channel
    reg [NW_BITS-1:0]  wi;          // which word of the frame
    reg [WA_BITS-1:0]  wa;          // weight ROM address, monotonic per frame
    reg [C_IN-1:0]     act;         // latched, so the caller may drop in_frame
    assign busy = (st != S_IDLE);

    reg [WORD_BITS-1:0] w_rom [0:C_OUT*NW-1];
    reg [31:0]          t_rom [0:2*C_OUT-1];
    initial begin
        if (W_FILE != "") $readmemh(W_FILE, w_rom);
        if (T_FILE != "") $readmemh(T_FILE, t_rom);
    end

    // word `wi` of the frame, and the weight word the address counter is on
    wire [WORD_BITS-1:0] act_w = act[wi*WORD_BITS +: WORD_BITS];
    wire [WORD_BITS-1:0] wgt_w = w_rom[wa];

    wire [CNT_BITS-1:0] n_terms = C_IN[CNT_BITS-1:0];

    // Combinational, not registered. A registered strobe rises one cycle after
    // its state, and pw advances its word/address counters inside S_FEED -- so
    // the MAC would have sampled word 1 against address base+1 and skipped word
    // 0 entirely. Driving both from the state keeps the strobe, the activation
    // word and the weight word in the same cycle by construction.
    wire                       mac_start = (st == S_START);
    wire                       mac_feed  = (st == S_FEED);
    wire                       mac_done;
    wire signed [ACC_BITS-1:0] mac_acc;

    kws_bin_mac #(.WORD_BITS(WORD_BITS), .ACC_BITS(ACC_BITS),
                  .CNT_BITS(CNT_BITS)) u_mac (
        .clk(clk), .rst_n(rst_n),
        .start(mac_start), .n_valid(n_terms),
        .in_valid(mac_feed), .act(act_w), .wgt(wgt_w),
        .out_valid(mac_done), .acc(mac_acc));

    wire signed [31:0] thr     = t_rom[{1'b0, co}];
    wire               take_ge = t_rom[{1'b1, co}][0];
    wire signed [TH_BITS-1:0] acc_x =
        {{(TH_BITS-ACC_BITS){mac_acc[ACC_BITS-1]}}, mac_acc};
    wire signed [TH_BITS-1:0] thr_x = {{(TH_BITS-32){thr[31]}}, thr};
    wire                      fired = ((acc_x >= thr_x) == take_ge);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st        <= S_IDLE;
            co        <= {CO_BITS{1'b0}};
            wi        <= {NW_BITS{1'b0}};
            wa        <= {WA_BITS{1'b0}};
            act       <= {C_IN{1'b0}};
            out_valid <= 1'b0;
            out_frame <= {C_OUT{1'b0}};
            acc_valid <= 1'b0;
            acc_ch    <= {CO_BITS{1'b0}};
            acc_out   <= {ACC_BITS{1'b0}};
        end else begin
            out_valid <= 1'b0;
            acc_valid <= 1'b0;
            case (st)
            S_IDLE:
                if (in_valid) begin
                    act <= in_frame;
                    co  <= {CO_BITS{1'b0}};
                    wa  <= {WA_BITS{1'b0}};   // frame restarts at the ROM base
                    st  <= S_START;
                end
            S_START: begin
                wi <= {NW_BITS{1'b0}};
                st <= S_FEED;
            end
            S_FEED: begin
                wa <= wa + {{(WA_BITS-1){1'b0}}, 1'b1};
                if (wi == LAST_W) st <= S_TAKE;
                else              wi <= wi + {{(NW_BITS-1){1'b0}}, 1'b1};
            end
            S_TAKE:
                if (mac_done) begin
                    out_frame[co] <= fired;
                    acc_valid     <= 1'b1;      // same cycle as the bit above
                    acc_ch        <= co;
                    acc_out       <= mac_acc;
                    if (co == LAST_CO) begin
                        st        <= S_IDLE;
                        out_valid <= 1'b1;
                    end else begin
                        co <= co + {{(CO_BITS-1){1'b0}}, 1'b1};
                        st <= S_START;
                    end
                end
            default: st <= S_IDLE;
            endcase
        end
    end

`ifdef KWS_ASSERT
    always @(posedge clk) if (in_valid && busy) begin
        $display("ASSERT %m: frame offered while busy -- it would be dropped");
        $finish;
    end
    // C_IN must be a whole number of words; a partial one would need the MAC's
    // tail mask and this module never asks for it.
    initial if (NW * WORD_BITS != C_IN) begin
        $display("ASSERT %m: C_IN=%0d is not a multiple of WORD_BITS=%0d",
                 C_IN, WORD_BITS);
        $finish;
    end
`endif

endmodule

`default_nettype wire
