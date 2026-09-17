// Deployable continuous board wrapper.
//
// This top exists only for an export that carries a validation-selected
// streaming policy. That makes accidentally synthesizing an old tag with a
// neutral/default margin impossible: export.emit must first write every
// KWS_STREAM_* macro into parameters.vh.

`include "rtl/gen/active.vh"

`ifdef KWS_STREAM_MARGIN_INT
module kws_stream_top #(
    parameter integer CLK_HZ     = 50000000,
    parameter integer CMP_INVERT = 0
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,
    input  wire [`KWS_N_CH-1:0] cmp,
    output wire                 busy,
    output wire                 class_valid,
    output wire [3:0]           class_idx
);

    localparam integer FRAME_CYCLES = (CLK_HZ / 1000) * `KWS_FRAME_MS;
    localparam integer PAD_RIGHT = `KWS_T - `KWS_PAD_LEFT - `KWS_NATIVE_T;

    // Reset: asynchronous assertion, two-clock synchronous release.
    (* ASYNC_REG = "TRUE" *) reg [1:0] rst_sync;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rst_sync <= 2'b00;
        else        rst_sync <= {rst_sync[0], 1'b1};
    end
    wire rst_n_s = rst_sync[1];

    // The board start input remains a manual diagnostic override. Continuous
    // capture itself begins after reset and does not depend on this pulse.
    (* ASYNC_REG = "TRUE" *) reg [1:0] start_sync;
    reg start_q;
    always @(posedge clk or negedge rst_n_s) begin
        if (!rst_n_s) begin
            start_sync <= 2'b00;
            start_q    <= 1'b0;
        end else begin
            start_sync <= {start_sync[0], start};
            start_q    <= start_sync[1];
        end
    end
    wire force_start = start_sync[1] && !start_q;

    wire [7:0] overrun_count_nc;
    kws_stream_core #(
        .N_CH(`KWS_N_CH), .FRAME_CYCLES(FRAME_CYCLES),
        .NATIVE_T(`KWS_NATIVE_T), .PAD_LEFT(`KWS_PAD_LEFT),
        .PAD_RIGHT(PAD_RIGHT), .T(`KWS_T),
        .TRIGGER_FRAMES(`KWS_STREAM_HOP_FRAMES),
        .REQUIRED(`KWS_STREAM_REQUIRED),
        .COOLDOWN_WINDOWS(`KWS_STREAM_COOLDOWN_WINDOWS),
        .MARGIN_BITS(`KWS_CONV4_POOL_BITS + 1),
        .MARGIN_INT(`KWS_STREAM_MARGIN_INT), .CMP_INVERT(CMP_INVERT)
    ) u_stream (
        .clk(clk), .rst_n(rst_n_s), .force_start(force_start), .cmp(cmp),
        .busy(busy), .detection_valid(class_valid),
        .detection_idx(class_idx), .overrun_count(overrun_count_nc),
        .obs_score_valid(), .obs_keyword_idx(), .obs_keyword_margin()
    );

`ifdef KWS_ASSERT
    initial begin
        $display("kws_stream_top: CLK_HZ=%0d frame=%0d cycles, hop=%0d, N=%0d, cooldown=%0d, margin_int=%0d",
                 CLK_HZ, FRAME_CYCLES, `KWS_STREAM_HOP_FRAMES,
                 `KWS_STREAM_REQUIRED, `KWS_STREAM_COOLDOWN_WINDOWS,
                 `KWS_STREAM_MARGIN_INT);
    end
`endif

endmodule
`endif
