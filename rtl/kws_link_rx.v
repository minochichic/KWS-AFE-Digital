// Wake link, receive side: one STM32 packet in, one clip of frames out.
//
//   STM32 -> FPGA   0xA5 | N | frame[0..N-1] (2 bytes each, LE, bit k = ch k) | CRC-8
//
// (docs/wake_stm32_plan.md 3). This module takes the place kws_frame_ctrl has
// on the always-on board: its output side is the same ready/valid stream of T
// frames into kws_top, so the network underneath -- the part verified 600/600
// on the chip -- does not change.
//
// THE PADDING IS ADDED HERE, NOT BY THE STM32. The firmware sends only the
// N = 100 native frames; the 14 frames of -1 either side are model geometry and
// come from parameters.vh like everything else the export decides. If the
// firmware knew T, every retrain that moved it would need a reflash.
//
// THE WHOLE PACKET IS CHECKED BEFORE ONE FRAME IS PUSHED. Streaming frames into
// the network as they arrive would start an inference that a bad CRC at the
// end could no longer cancel. 100 x 16 bits is small enough to buffer.
//
// A packet whose N is not NATIVE_T is dropped (err_len): the firmware and the
// export disagree about the clip length, and a padded guess would be a
// different input from the one the weights were trained on.
//
// `resync` (the WAKE line's rising edge) throws away a half-received packet,
// so a lost byte costs one packet, not every packet after it. It does not
// touch a clip already being played into the network.

`timescale 1ns/1ps
`default_nettype none

module kws_link_rx #(
    parameter integer N_CH     = 16,
    parameter integer NATIVE_T = 100,
    parameter integer T        = 128,
    parameter integer PAD_LEFT = 14,
    parameter [7:0]   SYNC     = 8'hA5
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire            resync,

    input  wire            rx_valid,
    input  wire [7:0]      rx_byte,

    output reg             net_start,
    output wire            net_valid,
    output wire [N_CH-1:0] net_frame,
    input  wire            net_ready,
    input  wire            net_busy,

    output reg             pkt_ok,      // 1-cycle pulses, for status/LEDs
    output reg             err_len,
    output reg             err_crc,
    output wire            busy
);

    localparam integer IW = (T <= 2) ? 1 : $clog2(T + 1);

    localparam [3:0] L_SYNC = 4'd0, L_LEN  = 4'd1, L_LO   = 4'd2,
                     L_HI   = 4'd3, L_CRC  = 4'd4, L_WAIT = 4'd5,
                     L_START= 4'd6, L_GAP  = 4'd7, L_FEED = 4'd8;
    reg [3:0]      st;
    reg [IW-1:0]   idx;             // payload frame index, then output frame index
    reg [7:0]      lo;
    reg [7:0]      crc;
    reg [N_CH-1:0] mem [0:NATIVE_T-1];

    wire [7:0] crc_next;
    kws_crc8 u_crc (.crc(crc), .data(rx_byte), .next(crc_next));

    wire parsing = (st == L_LEN) || (st == L_LO) || (st == L_HI) || (st == L_CRC);
    assign busy  = (st != L_SYNC);

    // ---- feed: T frames, the native ones framed by -1 (an all-zero word) --- //
    wire in_clip = (idx >= PAD_LEFT) && (idx < PAD_LEFT + NATIVE_T);
    wire [IW-1:0] mi = idx - PAD_LEFT[IW-1:0];
    assign net_frame = in_clip ? mem[mi] : {N_CH{1'b0}};
    // combinational valid, same reason as kws_selftest: kws_top asserts on
    // in_valid && !in_ready, so valid may only be high in a ready cycle
    assign net_valid = (st == L_FEED) && net_ready;

    always @(posedge clk) begin
        if (rx_valid && st == L_HI)
            mem[idx[IW-1:0]] <= {rx_byte, lo};
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= L_SYNC; idx <= {IW{1'b0}}; lo <= 8'd0; crc <= 8'd0;
            net_start <= 1'b0; pkt_ok <= 1'b0; err_len <= 1'b0; err_crc <= 1'b0;
        end else begin
            net_start <= 1'b0;
            pkt_ok    <= 1'b0;
            err_len   <= 1'b0;
            err_crc   <= 1'b0;
            if (resync && parsing) begin
                st <= L_SYNC;
            end else case (st)
            L_SYNC:
                if (rx_valid && rx_byte == SYNC) begin
                    crc <= crc_next;          // from 0: crc is held at 0 here
                    st  <= L_LEN;
                end
            L_LEN:
                if (rx_valid) begin
                    if (rx_byte == NATIVE_T[7:0]) begin
                        crc <= crc_next;
                        idx <= {IW{1'b0}};
                        st  <= L_LO;
                    end else begin
                        err_len <= 1'b1;
                        crc     <= 8'd0;
                        st      <= L_SYNC;
                    end
                end
            L_LO:
                if (rx_valid) begin
                    lo  <= rx_byte;
                    crc <= crc_next;
                    st  <= L_HI;
                end
            L_HI:
                if (rx_valid) begin
                    crc <= crc_next;
                    idx <= idx + 1'b1;
                    st  <= (idx == NATIVE_T - 1) ? L_CRC : L_LO;
                end
            L_CRC:
                if (rx_valid) begin
                    if (rx_byte == crc) begin pkt_ok  <= 1'b1; st <= L_WAIT; end
                    else                begin err_crc <= 1'b1; st <= L_SYNC; end
                    crc <= 8'd0;
                end
            L_WAIT:                           // previous clip still in the net
                if (!net_busy) st <= L_START;
            L_START: begin
                net_start <= 1'b1;
                idx       <= {IW{1'b0}};
                st        <= L_GAP;
            end
            L_GAP:                            // start lands a cycle before frame 0
                st <= L_FEED;
            L_FEED:
                if (net_valid) begin
                    if (idx == T - 1) st <= L_SYNC;
                    idx <= idx + 1'b1;
                end
            default: st <= L_SYNC;
            endcase
            // the CRC register restarts from 0 for the next packet whenever the
            // parser falls back to SYNC from anywhere
            if (resync && parsing) crc <= 8'd0;
        end
    end

`ifdef KWS_ASSERT
    initial if (N_CH != 16) begin
        $display("ASSERT %m: the packet carries 2 bytes per frame, N_CH must be 16, got %0d", N_CH);
        $finish;
    end
    initial if (NATIVE_T > 255 || PAD_LEFT + NATIVE_T > T) begin
        $display("ASSERT %m: NATIVE_T %0d / PAD_LEFT %0d / T %0d do not fit the packet", NATIVE_T, PAD_LEFT, T);
        $finish;
    end
`endif

endmodule

`default_nettype wire
