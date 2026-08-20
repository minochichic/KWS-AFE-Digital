// kws_plane driving kws_block, with no help from the testbench.
//
// Input  : <gen>/golden/conv1_out.hex  -> written into the plane
// Expect : <gen>/golden/b1_add_out.hex
//
// The point is not that the block still works -- tb_block already showed that.
// It is that the plane can REPLACE the driver loop: hold the frames, hand them
// over one at a time, wait out busy, and append the twelve flushes the block's
// two depthwise stages need. The testbench here only fills the plane and checks
// what comes out; it never pushes a frame or counts a flush.
//
// If this passes, kws_top is a sequence of these rather than a schedule.
//
// <gen> is whichever export run_tb.sh selected; the second
// argument picks it and defaults to xl_g12.
//
//   ./rtl/run_tb.sh plane

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module tb_plane;

    localparam integer C_IN  = 128;
    localparam integer C_MID = 64;
    localparam integer C_OUT = 64;
    localparam integer K     = 13;
    localparam integer PAD   = 6;
    localparam integer WB    = 32;
    localparam integer NWI   = C_IN  / WB;
    localparam integer NWO   = C_OUT / WB;
    localparam integer T     = 64;
    localparam integer CLIPS = 2;
    localparam integer FLUSH = 2 * PAD;      // kws_block has two depthwise stages

    reg  clk = 1'b0;
    reg  rst_n = 1'b0;
    always #5 clk = ~clk;

    // ---- plane ---------------------------------------------------------- //
    reg              wr_start = 1'b0, wr_valid = 1'b0;
    reg  [C_IN-1:0]  wr_frame = {C_IN{1'b0}};
    wire             wr_full;
    reg              rd_start = 1'b0;
    wire             pl_push, pl_real, pl_done;
    wire [C_IN-1:0]  pl_frame;
    wire             blk_busy;

    kws_plane #(.C(C_IN), .T(T), .FLUSH(FLUSH)) u_plane (
        .clk(clk), .rst_n(rst_n),
        .wr_start(wr_start), .wr_valid(wr_valid), .wr_frame(wr_frame),
        .wr_full(wr_full),
        .rd_start(rd_start), .rd_ready(!blk_busy),
        .rd_push(pl_push), .rd_real(pl_real), .rd_frame(pl_frame),
        .rd_done(pl_done));

    // ---- the consumer --------------------------------------------------- //
    reg              blk_start = 1'b0;
    wire             blk_ov;
    wire [C_OUT-1:0] blk_of;

    kws_block #(.C_IN(C_IN), .C_MID(C_MID), .C_OUT(C_OUT), .K(K), .PAD(PAD),
                .S0_DW_ACC(5), .S0_PW_ACC(9), .S1_DW_ACC(5), .S1_PW_ACC(8),
                .SKIP_ACC(9), .ADD_ACC(9), .WORD_BITS(WB),
                .S0_DW_W(`KWS_ROM_B1_S0_DW_W),
                .S0_DW_T(`KWS_ROM_B1_S0_DW_T),
                .S0_PW_W(`KWS_ROM_B1_S0_PW_W),
                .S0_PW_T(`KWS_ROM_B1_S0_PW_T),
                .S1_DW_W(`KWS_ROM_B1_S1_DW_W),
                .S1_DW_T(`KWS_ROM_B1_S1_DW_T),
                .S1_PW_W(`KWS_ROM_B1_S1_PW_W),
                .SKIP_W (`KWS_ROM_B1_SKIP_W),
                .ADD_T  (`KWS_ROM_B1_ADD_T)) u_blk (
        .clk(clk), .rst_n(rst_n), .start(blk_start),
        .in_push(pl_push), .in_real(pl_real), .in_frame(pl_frame),
        .busy(blk_busy), .out_valid(blk_ov), .out_frame(blk_of));

    // ---- collect what the block produces -------------------------------- //
    reg [WB-1:0]     in_mem  [0:CLIPS*T*NWI-1];
    reg [WB-1:0]     exp_mem [0:CLIPS*T*NWO-1];
    reg [C_OUT-1:0]  got_mem [0:T-1];
    integer          got_n;

    always @(posedge clk) begin
        if (blk_ov && got_n < T) begin
            got_mem[got_n] <= blk_of;
            got_n          <= got_n + 1;
        end
    end

    integer errors = 0;
    integer checked = 0;
    integer clip, t, j;
    reg [C_OUT-1:0] want;

    initial begin
        $dumpfile("tb_plane.vcd");
        $dumpvars(0, tb_plane);

        $readmemh(`KWS_GOLD_CONV1_OUT, in_mem);
        $readmemh(`KWS_GOLD_B1_ADD_OUT, exp_mem);

        got_n = 0;
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        for (clip = 0; clip < CLIPS; clip = clip + 1) begin
            // fill the plane. No pushing, no flush counting -- just frames.
            @(negedge clk); wr_start = 1'b1;
            @(negedge clk); wr_start = 1'b0;
            for (t = 0; t < T; t = t + 1) begin
                for (j = 0; j < NWI; j = j + 1)
                    wr_frame[j*WB +: WB] = in_mem[(clip*T + t)*NWI + j];
                wr_valid = 1'b1;
                @(negedge clk);
            end
            wr_valid = 1'b0;
            @(negedge clk);
            if (!wr_full) begin
                $display("FAIL clip%0d: plane not full after %0d writes",
                         clip, T);
                errors = errors + 1;
            end

            // hand it over. From here the testbench does nothing.
            got_n = 0;
            @(negedge clk); blk_start = 1'b1;
            @(negedge clk); blk_start = 1'b0;
            @(negedge clk); rd_start = 1'b1;
            @(negedge clk); rd_start = 1'b0;

            wait (pl_done === 1'b1);
            // the last frame's output lands a few cycles after the last push
            repeat (4000) @(negedge clk);

            if (got_n !== T) begin
                $display("FAIL clip%0d: %0d frames out, expected %0d",
                         clip, got_n, T);
                errors = errors + 1;
            end
            for (t = 0; t < T; t = t + 1) begin
                want = {C_OUT{1'b0}};
                for (j = 0; j < NWO; j = j + 1)
                    want[j*WB +: WB] = exp_mem[(clip*T + t)*NWO + j];
                checked = checked + 1;
                if (got_mem[t] !== want) begin
                    errors = errors + 1;
                    if (errors <= 5)
                        $display("FAIL clip%0d t=%0d\n  got  %h\n  want %h",
                                 clip, t, got_mem[t], want);
                end
            end
            $display("ok   clip%0d: %0d frames, plane drove all of it",
                     clip, got_n);
        end

        $display("\n%0d frames checked, %0d failures", checked, errors);
        $finish;
    end

    initial begin
        #400_000_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule

`default_nettype wire
