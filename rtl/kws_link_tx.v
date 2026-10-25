// Wake link, transmit side: one classification out as one 8-byte packet.
//
//   FPGA -> STM32   0x5A | class_idx | keyword_idx | margin (4 bytes, LE, signed) | CRC-8
//
// margin is kws_tail's keyword_margin (pool-sum units, never divided) sign-
// extended to 32 bits, so the firmware reads it as an int32_t without knowing
// the pool width. The CRC covers every byte before it, 0x5A included.
//
// A result that arrives while the previous packet is still going out is
// dropped and counted (`dropped`). At 115200 baud a packet takes ~0.7 ms and
// an inference 57 ms, so this should never fire; if it does, something is
// starting inferences the link did not ask for.

`timescale 1ns/1ps
`default_nettype none

module kws_link_tx #(
    parameter integer CLS_BITS = 4,
    parameter integer M_BITS   = 22,     // keyword_margin width (POOL_BITS + 1)
    parameter [7:0]   SYNC     = 8'h5A
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     res_valid,
    input  wire [CLS_BITS-1:0]      class_idx,
    input  wire [CLS_BITS-1:0]      keyword_idx,
    input  wire signed [M_BITS-1:0] margin,

    output wire                     tx_send,
    output wire [7:0]               tx_data,
    input  wire                     tx_ready,

    output reg                      dropped
);

    reg        active;
    reg [2:0]  bi;                      // byte index 0..7
    reg [7:0]  buf_c, buf_k;
    reg [31:0] buf_m;
    reg [7:0]  crc;

    wire [7:0] cur =
        (bi == 3'd0) ? SYNC  :
        (bi == 3'd1) ? buf_c :
        (bi == 3'd2) ? buf_k :
        (bi == 3'd3) ? buf_m[7:0]   :
        (bi == 3'd4) ? buf_m[15:8]  :
        (bi == 3'd5) ? buf_m[23:16] :
        (bi == 3'd6) ? buf_m[31:24] : crc;

    wire [7:0] crc_next;
    kws_crc8 u_crc (.crc(crc), .data(cur), .next(crc_next));

    assign tx_send = active && tx_ready;
    assign tx_data = cur;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0; bi <= 3'd0; crc <= 8'd0; dropped <= 1'b0;
            buf_c <= 8'd0; buf_k <= 8'd0; buf_m <= 32'd0;
        end else begin
            dropped <= 1'b0;
            if (res_valid) begin
                if (active) dropped <= 1'b1;
                else begin
                    active <= 1'b1;
                    bi     <= 3'd0;
                    crc    <= 8'd0;
                    buf_c  <= {{(8-CLS_BITS){1'b0}}, class_idx};
                    buf_k  <= {{(8-CLS_BITS){1'b0}}, keyword_idx};
                    buf_m  <= {{(32-M_BITS){margin[M_BITS-1]}}, margin};
                end
            end
            if (tx_send) begin
                crc <= crc_next;
                bi  <= bi + 3'd1;
                if (bi == 3'd7) active <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
