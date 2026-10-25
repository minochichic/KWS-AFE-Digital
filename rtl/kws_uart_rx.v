// UART receiver, 8N1, LSB first. The STM32 -> FPGA half of the wake link
// (docs/wake_stm32_plan.md 3).
//
// One sample per bit, taken in the middle of the bit. The start bit is checked
// again at its middle so a glitch on an idle line is dropped rather than read
// as 0x00. A low stop bit is a framing error: the byte is thrown away, not
// delivered, because a packet with one wrong byte must fail its CRC anyway and
// a missing byte fails it more reliably than a wrong one.
//
// CLKS_PER_BIT = CLK_HZ / BAUD, rounded down. 50 MHz / 115200 = 434.03, a
// 0.007% error; a UART tolerates a few percent over ten bits.

`timescale 1ns/1ps
`default_nettype none

module kws_uart_rx #(
    parameter integer CLKS_PER_BIT = 434
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,          // ASYNCHRONOUS, idle high
    output reg        valid,       // 1 cycle per received byte
    output reg  [7:0] data,
    output reg        frame_err    // 1 cycle: stop bit was low, byte dropped
);

    localparam integer CW = (CLKS_PER_BIT <= 2) ? 2 : $clog2(CLKS_PER_BIT + 1);
    localparam [CW-1:0] FULL = CLKS_PER_BIT - 1;
    localparam [CW-1:0] HALF = CLKS_PER_BIT / 2 - 1;

    // the line idles high, so the synchroniser does too -- a reset to 0 would
    // look like a start bit on the first cycle out of reset
    (* ASYNC_REG = "TRUE" *) reg [1:0] sync;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) sync <= 2'b11;
        else        sync <= {sync[0], rx};
    wire r = sync[1];

    localparam [1:0] S_IDLE = 2'd0, S_START = 2'd1, S_DATA = 2'd2, S_STOP = 2'd3;
    reg [1:0]    st;
    reg [CW-1:0] cnt;
    reg [2:0]    bit_i;
    reg [7:0]    sh;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; cnt <= {CW{1'b0}}; bit_i <= 3'd0; sh <= 8'd0;
            valid <= 1'b0; data <= 8'd0; frame_err <= 1'b0;
        end else begin
            valid     <= 1'b0;
            frame_err <= 1'b0;
            case (st)
            S_IDLE:
                if (!r) begin st <= S_START; cnt <= {CW{1'b0}}; end
            S_START:
                if (cnt == HALF) begin
                    cnt   <= {CW{1'b0}};
                    bit_i <= 3'd0;
                    st    <= r ? S_IDLE : S_DATA;     // glitch -> back to idle
                end else cnt <= cnt + 1'b1;
            S_DATA:
                if (cnt == FULL) begin
                    cnt <= {CW{1'b0}};
                    sh  <= {r, sh[7:1]};
                    if (bit_i == 3'd7) st <= S_STOP;
                    bit_i <= bit_i + 3'd1;
                end else cnt <= cnt + 1'b1;
            S_STOP:
                if (cnt == FULL) begin
                    cnt <= {CW{1'b0}};
                    st  <= S_IDLE;
                    if (r) begin valid <= 1'b1; data <= sh; end
                    else   frame_err <= 1'b1;
                end else cnt <= cnt + 1'b1;
            endcase
        end
    end

endmodule

`default_nettype wire
