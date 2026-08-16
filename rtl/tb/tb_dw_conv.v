// kws_dw_conv against the real thing: trained weights, trained thresholds, and
// the activations the network actually produced.
//
// Input  : rtl/gen/xl_g12/golden/conv1_out.hex   (b1's block input)
// Expect : rtl/gen/xl_g12/golden/b1_s0_dw_out.hex
// ROMs   : rtl/gen/xl_g12/b1_s0_dw_{w,t}.hex
//
// Nothing here is synthetic. If this passes, the line buffer, the tap gather,
// the edge shift, n_valid and the fused threshold are all right together --
// and if it fails, kws_bin_mac is already known good, so the fault is in one
// of those five.
//
//   ./rtl/run_tb.sh dw_conv

`timescale 1ns/1ps
`default_nettype none

module tb_dw_conv;

    localparam integer C     = 128;
    localparam integer K     = 13;
    localparam integer PAD   = 6;
    localparam integer ACC   = 5;      // manifest: b1_s0_dw acc_bits
    localparam integer WB    = 32;
    localparam integer NW    = C / WB; // words per frame
    localparam integer T     = 64;
    localparam integer CLIPS = 2;

    reg              clk = 1'b0;
    reg              rst_n = 1'b0;
    reg              start = 1'b0;
    reg              in_push = 1'b0;
    reg              in_real = 1'b0;
    reg  [C-1:0]     in_frame = {C{1'b0}};
    wire             busy;
    wire             out_valid;
    wire [C-1:0]     out_frame;

    kws_dw_conv #(.C(C), .K(K), .PAD(PAD), .ACC_BITS(ACC), .WORD_BITS(WB),
                  .W_FILE("rtl/gen/xl_g12/b1_s0_dw_w.hex"),
                  .T_FILE("rtl/gen/xl_g12/b1_s0_dw_t.hex")) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .in_push(in_push), .in_real(in_real), .in_frame(in_frame),
        .busy(busy), .out_valid(out_valid), .out_frame(out_frame));

    always #5 clk = ~clk;

    reg [WB-1:0] in_mem  [0:CLIPS*T*NW-1];
    reg [WB-1:0] exp_mem [0:CLIPS*T*NW-1];

    // latch the result so the driver can check it after busy drops
    reg [C-1:0] got;
    reg         got_v;
    always @(posedge clk) begin
        if (out_valid) begin
            got   <= out_frame;
            got_v <= 1'b1;
        end
    end

    integer errors = 0;
    integer checked = 0;
    integer clip, i, j, t;
    reg [C-1:0] want, frame;

    task push;                        // one frame in, wait until it settles
        input       real_f;
        input [C-1:0] fr;
        begin
            @(negedge clk);
            got_v    = 1'b0;
            in_push  = 1'b1;
            in_real  = real_f;
            in_frame = fr;
            @(negedge clk);
            in_push  = 1'b0;
            while (busy) @(negedge clk);
        end
    endtask

    initial begin
        $dumpfile("tb_dw_conv.vcd");
        $dumpvars(0, tb_dw_conv);

        $readmemh("rtl/gen/xl_g12/golden/conv1_out.hex",    in_mem);
        $readmemh("rtl/gen/xl_g12/golden/b1_s0_dw_out.hex", exp_mem);

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        for (clip = 0; clip < CLIPS; clip = clip + 1) begin
            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;

            // T real pushes then PAD flush pushes -- the drain that produces
            // the last PAD outputs (docs/diagrams/24_pipeline_drain.svg)
            for (i = 0; i < T + PAD; i = i + 1) begin
                frame = {C{1'b0}};
                if (i < T)
                    for (j = 0; j < NW; j = j + 1)
                        frame[j*WB +: WB] = in_mem[(clip*T + i)*NW + j];
                push(i < T, frame);

                t = i - PAD;                    // which output this push made
                if (t >= 0) begin
                    if (!got_v) begin
                        $display("FAIL clip%0d t=%0d: no output", clip, t);
                        errors = errors + 1;
                    end else begin
                        want = {C{1'b0}};
                        for (j = 0; j < NW; j = j + 1)
                            want[j*WB +: WB] = exp_mem[(clip*T + t)*NW + j];
                        checked = checked + 1;
                        if (got !== want) begin
                            errors = errors + 1;
                            if (errors <= 5)
                                $display("FAIL clip%0d t=%0d\n  got  %h\n  want %h",
                                         clip, t, got, want);
                        end
                    end
                end else if (got_v) begin
                    $display("FAIL clip%0d push %0d: output during fill",
                             clip, i);
                    errors = errors + 1;
                end
            end
            $display("ok   clip%0d: %0d frames", clip, T);
        end

        $display("\n%0d frames checked, %0d failures", checked, errors);
        $finish;
    end

    // a runaway FSM should not hang the run
    initial begin
        #20_000_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule

`default_nettype wire
