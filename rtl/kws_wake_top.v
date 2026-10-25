// Board top for the wake-up system (docs/wake_stm32_plan.md, option B):
// the STM32 wakes, captures one second and sends it; this classifies it.
//
//   STM32 J4 ── uart_rx ──▶ kws_uart_rx ─▶ kws_link_rx ─▶ kws_top_synth ─┐
//            ◀─ uart_tx ─── kws_uart_tx ◀─ kws_link_tx ◀─────────────────┘
//            ── wake ────▶ (rising edge resyncs the packet parser)
//
// This is kws_board_top with kws_frame_ctrl swapped for the link: the network
// is the same instance of kws_top_synth, fed the same ready/valid frames. The
// comparators are not wired to the FPGA at all on this board -- they go to the
// STM32 (J5 -> PB0..15).
//
// Like kws_board_top this module computes nothing; it absorbs the board.

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module kws_wake_top #(
    parameter integer CLK_HZ = 50000000,   // kit clock switch F (hanback_kit.md 3.2)
    parameter integer BAUD   = 115200
) (
    input  wire       clk,
    input  wire       rst_n,        // pin, async, internal pull-up
    input  wire       uart_rx,      // from STM32 UART_TX, async, idle high
    output wire       uart_tx,      // to STM32 UART_RX
    input  wire       wake,         // from STM32 WAKE_FPGA, async
    output wire       busy,
    output wire       class_valid,
    output wire [3:0] class_idx,
    output reg        link_err      // sticky: a packet was dropped (len / crc / framing)
);

    localparam integer CLKS_PER_BIT = CLK_HZ / BAUD;

    // ---- reset: async assert, sync release (kws_board_top 1) -------------- //
    (* ASYNC_REG = "TRUE" *) reg [1:0] rst_sync;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) rst_sync <= 2'b00;
        else        rst_sync <= {rst_sync[0], 1'b1};
    wire rst_n_s = rst_sync[1];

    // ---- wake: sync + rising edge ------------------------------------------ //
    (* ASYNC_REG = "TRUE" *) reg [1:0] wake_sync;
    reg wake_q;
    always @(posedge clk or negedge rst_n_s)
        if (!rst_n_s) begin wake_sync <= 2'b00; wake_q <= 1'b0; end
        else begin wake_sync <= {wake_sync[0], wake}; wake_q <= wake_sync[1]; end
    wire wake_rise = wake_sync[1] && !wake_q;

    // ---- link in ------------------------------------------------------------ //
    wire       rx_valid, rx_ferr;
    wire [7:0] rx_byte;
    kws_uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_urx (
        .clk(clk), .rst_n(rst_n_s), .rx(uart_rx),
        .valid(rx_valid), .data(rx_byte), .frame_err(rx_ferr));

    wire                 net_start, net_valid, net_ready, net_busy;
    wire [`KWS_N_CH-1:0] net_frame;
    wire                 pkt_ok, err_len, err_crc, lrx_busy;
    kws_link_rx #(
        .N_CH(`KWS_N_CH), .NATIVE_T(`KWS_NATIVE_T),
        .T(`KWS_T), .PAD_LEFT(`KWS_PAD_LEFT)
    ) u_lrx (
        .clk(clk), .rst_n(rst_n_s), .resync(wake_rise),
        .rx_valid(rx_valid), .rx_byte(rx_byte),
        .net_start(net_start), .net_valid(net_valid), .net_frame(net_frame),
        .net_ready(net_ready), .net_busy(net_busy),
        .pkt_ok(pkt_ok), .err_len(err_len), .err_crc(err_crc), .busy(lrx_busy));

    // ---- the network, unchanged -------------------------------------------- //
    wire       cls_valid, sc_valid;
    wire [3:0] cls_idx, kw_idx;
    wire signed [`KWS_CONV4_POOL_BITS:0] kw_margin;
    kws_top_synth u_net (
        .clk(clk), .rst_n(rst_n_s), .start(net_start),
        .in_valid(net_valid), .in_frame(net_frame), .in_ready(net_ready),
        .busy(net_busy),
        .class_valid(cls_valid), .class_idx(cls_idx),
        .score_valid(sc_valid), .keyword_idx(kw_idx), .keyword_margin(kw_margin));

    // ---- link out ------------------------------------------------------------ //
    wire       tx_send, tx_ready, tx_dropped;
    wire [7:0] tx_data;
    kws_link_tx #(.CLS_BITS(4), .M_BITS(`KWS_CONV4_POOL_BITS + 1)) u_ltx (
        .clk(clk), .rst_n(rst_n_s),
        .res_valid(cls_valid), .class_idx(cls_idx),
        .keyword_idx(kw_idx), .margin(kw_margin),
        .tx_send(tx_send), .tx_data(tx_data), .tx_ready(tx_ready),
        .dropped(tx_dropped));
    kws_uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_utx (
        .clk(clk), .rst_n(rst_n_s), .send(tx_send), .data(tx_data),
        .ready(tx_ready), .tx(uart_tx));

    // ---- outward: busy, latched class (kws_board_top 4), sticky error ------- //
    assign busy = lrx_busy || net_busy;

    reg [3:0] class_idx_r;
    reg       class_valid_r;
    always @(posedge clk or negedge rst_n_s) begin
        if (!rst_n_s) begin
            class_idx_r <= 4'd0; class_valid_r <= 1'b0; link_err <= 1'b0;
        end else begin
            class_valid_r <= cls_valid;
            if (cls_valid) class_idx_r <= cls_idx;
            if (err_len || err_crc || rx_ferr || tx_dropped) link_err <= 1'b1;
            else if (pkt_ok) link_err <= 1'b0;
        end
    end
    assign class_idx   = class_idx_r;
    assign class_valid = class_valid_r;

`ifdef KWS_ASSERT
    initial begin
        if (CLKS_PER_BIT < 8) begin
            $display("ASSERT %m: CLK_HZ %0d / BAUD %0d = %0d clocks per bit, need >= 8",
                     CLK_HZ, BAUD, CLKS_PER_BIT);
            $finish;
        end
        $display("kws_wake_top: CLK_HZ %0d, BAUD %0d, %0d clocks/bit",
                 CLK_HZ, BAUD, CLKS_PER_BIT);
    end
    // score and class come out of kws_tail in the same cycle; link_tx latches
    // the keyword fields on class_valid, so check that stays true
    always @(posedge clk) if (cls_valid !== sc_valid) begin
        $display("ASSERT %m: class_valid and score_valid diverged"); $finish;
    end
`endif

endmodule

`default_nettype wire
