// UART transmitter, 8N1, LSB first. The FPGA -> STM32 half of the wake link.
//
// `send` is taken only while `ready` is high; the byte is copied, so the
// caller may change `data` the next cycle.

`timescale 1ns/1ps
`default_nettype none

module kws_uart_tx #(
    parameter integer CLKS_PER_BIT = 434
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       send,
    input  wire [7:0] data,
    output wire       ready,
    output reg        tx           // idle high
);

    localparam integer CW = (CLKS_PER_BIT <= 2) ? 2 : $clog2(CLKS_PER_BIT + 1);
    localparam [CW-1:0] FULL = CLKS_PER_BIT - 1;

    reg          busy;
    reg [CW-1:0] cnt;
    reg [3:0]    bit_i;            // 0 = start, 1..8 = data, 9 = stop
    reg [9:0]    sh;               // {stop, data, start}, shifted out LSB first

    assign ready = !busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; cnt <= {CW{1'b0}}; bit_i <= 4'd0; sh <= 10'h3FF;
            tx <= 1'b1;
        end else if (!busy) begin
            tx <= 1'b1;
            if (send) begin
                busy  <= 1'b1;
                sh    <= {1'b1, data, 1'b0};
                tx    <= 1'b0;                 // start bit goes out now
                cnt   <= {CW{1'b0}};
                bit_i <= 4'd0;
            end
        end else if (cnt == FULL) begin
            cnt <= {CW{1'b0}};
            if (bit_i == 4'd9) begin
                busy <= 1'b0;
                tx   <= 1'b1;
            end else begin
                bit_i <= bit_i + 4'd1;
                sh    <= {1'b1, sh[9:1]};
                tx    <= sh[1];
            end
        end else cnt <= cnt + 1'b1;
    end

endmodule

`default_nettype wire
