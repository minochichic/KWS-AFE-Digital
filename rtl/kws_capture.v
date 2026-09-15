// Free-running comparator capture at the analog/digital boundary.
//
//   cmp[k] -> [2FF sync] -> [sticky OR] -> [frame timer] -> frame[k]
//
// `enable` controls the capture window, not the synchronizer. While disabled,
// the timer and sticky bits are held at zero. While enabled, frame_valid pulses
// once every FRAME_CYCLES clocks and frame contains the OR of every synchronized
// comparator value seen in that window, including the closing clock edge.
//
// Keeping this independent of clip replay is what permits continuous capture:
// kws_frame_ctrl enables it only during its legacy S_RUN phase, while the future
// kws_window keeps it enabled permanently and stores every 10 ms frame.

`timescale 1ns/1ps
`default_nettype none

module kws_capture #(
    parameter integer N_CH         = 16,
    parameter integer FRAME_CYCLES = 500000,
    parameter integer CMP_INVERT   = 0
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire            enable,
    input  wire [N_CH-1:0] cmp,

    output wire            frame_valid,
    output wire [N_CH-1:0] frame
);

    // The only logic allowed to touch the asynchronous comparator pins.
    (* ASYNC_REG = "TRUE" *) reg [N_CH-1:0] sync1;
    (* ASYNC_REG = "TRUE" *) reg [N_CH-1:0] sync2;

    wire [N_CH-1:0] cmp_in = (CMP_INVERT != 0) ? ~cmp : cmp;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sync1 <= {N_CH{1'b0}};
            sync2 <= {N_CH{1'b0}};
        end else begin
            sync1 <= cmp_in;
            sync2 <= sync1;
        end
    end

    localparam integer FC_BITS = (FRAME_CYCLES <= 2) ? 1 : $clog2(FRAME_CYCLES);
    localparam integer FC_LAST_I = FRAME_CYCLES - 1;
    localparam [FC_BITS-1:0] FC_LAST = FC_LAST_I[FC_BITS-1:0];

    reg [FC_BITS-1:0] fc;
    reg [N_CH-1:0] sticky;

    assign frame_valid = enable && (fc == FC_LAST);
    // The closing edge belongs to the window being emitted.
    assign frame = sticky | sync2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fc <= {FC_BITS{1'b0}};
        end else if (!enable || frame_valid) begin
            fc <= {FC_BITS{1'b0}};
        end else begin
            fc <= fc + {{(FC_BITS-1){1'b0}}, 1'b1};
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || !enable) begin
            sticky <= {N_CH{1'b0}};
        end else if (frame_valid) begin
            sticky <= {N_CH{1'b0}};
        end else begin
            sticky <= sticky | sync2;
        end
    end

`ifdef KWS_ASSERT
    initial if (FRAME_CYCLES <= 0) begin
        $display("ASSERT %m: FRAME_CYCLES must be positive, got %0d",
                 FRAME_CYCLES);
        $finish;
    end
`endif

endmodule

`default_nettype wire
