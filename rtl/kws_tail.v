// The whole tail: a binary frame in, a class index out.
//
//   conv2_pw  kws_pw_conv     binary MAC, raw accumulator (no threshold)
//     +       kws_affine      -> 8.6, relu
//   conv3     kws_dense_conv  int8 x 8.6
//     +       kws_affine      -> 5.6, relu
//   conv4     kws_dense_conv  fixed x 5.6
//     +       kws_affine      -> 8.6, signed logits
//   pool      sum over T frames, then argmax
//
// THERE IS NO PLANE IN HERE, and that is worth saying because every junction
// upstream needed one. kws_affine emits one channel per cycle and
// kws_dense_conv loads one per cycle, so they connect directly -- the dense
// conv's own act[] is the frame buffer, and it has to hold the frame anyway,
// since output channel o touches every input channel. The only state that
// spans frames is the pool.
//
// THE POOL NEVER DIVIDES. adaptive_avg_pool1d divides all twelve classes by the
// same T, and argmax does not care about a shared positive factor. So the sum
// is kept T times wider -- six bits at T=64 -- which is exact, where a divider
// would not be, and smaller than a divider besides.
//
// ONE FRAME AT A TIME, SEQUENTIALLY. 18,700 cycles a frame and 1.2M a clip,
// which at 10 Hz is 12 MHz. The three stages could overlap and it would buy
// nothing.
//
// WHERE THE BOUNDARY IS. conv2_dw hands over +-1 frames; everything from
// conv2_pw's accumulator onward lives here. That split is not arbitrary --
// conv2_pw is the first layer whose BN survives (docs/diagrams/30_bn_vanishes),
// so it is exactly where compares stop and arithmetic starts.

`timescale 1ns/1ps
`default_nettype none

module kws_tail #(
    // ---- conv2_pw: binary pointwise, raw accumulator ------------------- //
    parameter integer C2_IN     = 64,
    parameter integer C2_OUT    = 128,
    parameter integer C2_ACC    = 8,
    parameter integer WORD_BITS = 32,
    parameter         C2_W_FILE = "",
    // ---- its epilogue -------------------------------------------------- //
    parameter integer A2_GAIN   = 22,
    parameter integer A2_BIAS   = 26,
    parameter integer A2_SHIFT  = 18,
    parameter integer A2_OUT    = 14,
    parameter         A2_FILE   = "",
    // ---- conv3 --------------------------------------------------------- //
    parameter integer C3_OUT    = 128,
    parameter integer C3_W      = 8,
    parameter integer C3_ACC    = 28,
    parameter         C3_W_FILE = "",
    parameter integer A3_GAIN   = 18,
    parameter integer A3_BIAS   = 32,
    parameter integer A3_SHIFT  = 24,
    parameter integer A3_OUT    = 11,
    parameter         A3_FILE   = "",
    // ---- conv4 --------------------------------------------------------- //
    parameter integer C4_OUT    = 12,
    parameter integer C4_W      = 8,
    parameter integer C4_ACC    = 25,
    parameter         C4_W_FILE = "",
    parameter integer A4_GAIN   = 23,
    parameter integer A4_BIAS   = 31,
    parameter integer A4_SHIFT  = 27,
    parameter integer A4_OUT    = 14,
    parameter         A4_FILE   = "",
    // ---- the head ------------------------------------------------------ //
    parameter integer T_FRAMES  = 64,
    parameter integer POOL_BITS = 21,
    // widths the ports need, which a localparam cannot supply here
    parameter integer C2O_BITS  = 7,     // $clog2(C2_OUT)
    parameter integer C3O_BITS  = 7,     // $clog2(C3_OUT)
    parameter integer C4O_BITS  = 4      // $clog2(C4_OUT)
) (
    input  wire                clk,
    input  wire                rst_n,

    input  wire                start,      // new clip: clear the pool
    input  wire                in_valid,   // one +-1 frame from conv2_dw
    input  wire [C2_IN-1:0]    in_frame,
    output wire                busy,

    output reg                 class_valid,
    output reg [C4O_BITS-1:0]  class_idx
);

    // ---- conv2_pw ------------------------------------------------------- //
    // T_FILE is empty: this layer's epilogue is arithmetic, not a compare, so
    // only its raw accumulator is used and the thresholded output is dropped.
    /* verilator lint_off UNUSEDSIGNAL */
    wire               pw_ov_nc;
    wire [C2_OUT-1:0]  pw_of_nc;
    /* verilator lint_on UNUSEDSIGNAL */

    wire                     pw_busy, pw_av;
    wire [C2O_BITS-1:0]      pw_ach;
    wire signed [C2_ACC-1:0] pw_acc;

    kws_pw_conv #(.C_IN(C2_IN), .C_OUT(C2_OUT), .ACC_BITS(C2_ACC),
                  .WORD_BITS(WORD_BITS), .W_FILE(C2_W_FILE), .T_FILE(""),
                  .CO_BITS_P(C2O_BITS)) u_pw (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_frame(in_frame),
        .busy(pw_busy), .out_valid(pw_ov_nc), .out_frame(pw_of_nc),
        .acc_valid(pw_av), .acc_ch(pw_ach), .acc_out(pw_acc));

    wire                    a2_v;
    wire [C2O_BITS-1:0]     a2_ch;
    wire signed [A2_OUT-1:0] a2_val;

    kws_affine #(.C(C2_OUT), .ACC_BITS(C2_ACC), .GAIN_BITS(A2_GAIN),
                 .BIAS_BITS(A2_BIAS), .SHIFT(A2_SHIFT), .OUT_BITS(A2_OUT),
                 .RELU(1), .CH_BITS(C2O_BITS), .ROM_FILE(A2_FILE)) u_a2 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(pw_av), .in_ch(pw_ach), .in_acc(pw_acc),
        .out_valid(a2_v), .out_ch(a2_ch), .out_val(a2_val));

    // ---- conv3 ---------------------------------------------------------- //
    // a2 feeds u_d3's frame buffer directly, one channel per cycle. The only
    // sequencing is when to say "the frame is in": count the loads.
    reg  d3_go;
    wire                     d3_busy, d3_av;
    wire [C3O_BITS-1:0]      d3_ach;
    wire signed [C3_ACC-1:0] d3_acc;

    kws_dense_conv #(.C_IN(C2_OUT), .C_OUT(C3_OUT), .IN_BITS(A2_OUT),
                     .W_BITS(C3_W), .ACC_BITS(C3_ACC),
                     .CI_BITS(C2O_BITS), .CO_BITS(C3O_BITS),
                     .W_FILE(C3_W_FILE)) u_d3 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(a2_v), .in_ch(a2_ch), .in_val(a2_val),
        .start(d3_go), .busy(d3_busy),
        .acc_valid(d3_av), .acc_ch(d3_ach), .acc_out(d3_acc));

    wire                     a3_v;
    wire [C3O_BITS-1:0]      a3_ch;
    wire signed [A3_OUT-1:0] a3_val;

    kws_affine #(.C(C3_OUT), .ACC_BITS(C3_ACC), .GAIN_BITS(A3_GAIN),
                 .BIAS_BITS(A3_BIAS), .SHIFT(A3_SHIFT), .OUT_BITS(A3_OUT),
                 .RELU(1), .CH_BITS(C3O_BITS), .ROM_FILE(A3_FILE)) u_a3 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(d3_av), .in_ch(d3_ach), .in_acc(d3_acc),
        .out_valid(a3_v), .out_ch(a3_ch), .out_val(a3_val));

    // ---- conv4 ---------------------------------------------------------- //
    reg  d4_go;
    wire                     d4_busy, d4_av;
    wire [C4O_BITS-1:0]      d4_ach;
    wire signed [C4_ACC-1:0] d4_acc;

    kws_dense_conv #(.C_IN(C3_OUT), .C_OUT(C4_OUT), .IN_BITS(A3_OUT),
                     .W_BITS(C4_W), .ACC_BITS(C4_ACC),
                     .CI_BITS(C3O_BITS), .CO_BITS(C4O_BITS),
                     .W_FILE(C4_W_FILE)) u_d4 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(a3_v), .in_ch(a3_ch), .in_val(a3_val),
        .start(d4_go), .busy(d4_busy),
        .acc_valid(d4_av), .acc_ch(d4_ach), .acc_out(d4_acc));

    wire                     a4_v;
    wire [C4O_BITS-1:0]      a4_ch;
    wire signed [A4_OUT-1:0] a4_val;

    kws_affine #(.C(C4_OUT), .ACC_BITS(C4_ACC), .GAIN_BITS(A4_GAIN),
                 .BIAS_BITS(A4_BIAS), .SHIFT(A4_SHIFT), .OUT_BITS(A4_OUT),
                 .RELU(0), .CH_BITS(C4O_BITS), .ROM_FILE(A4_FILE)) u_a4 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(d4_av), .in_ch(d4_ach), .in_acc(d4_acc),
        .out_valid(a4_v), .out_ch(a4_ch), .out_val(a4_val));

    // ---- the two start pulses ------------------------------------------- //
    // Nothing else needs sequencing: each stage's output port set is the next
    // stage's input port set, so the data path is wire-to-wire. A dense conv
    // only needs telling when its frame is complete, and "complete" is just a
    // count of how many channels its producer has handed over.
    localparam integer L3_BITS = (C2_OUT <= 2) ? 1 : $clog2(C2_OUT) + 1;
    localparam integer L4_BITS = (C3_OUT <= 2) ? 1 : $clog2(C3_OUT) + 1;
    localparam integer L3_I = C2_OUT;
    localparam integer L4_I = C3_OUT;
    localparam [L3_BITS-1:0] L3_FULL = L3_I[L3_BITS-1:0];
    localparam [L4_BITS-1:0] L4_FULL = L4_I[L4_BITS-1:0];

    reg [L3_BITS-1:0] n3;
    reg [L4_BITS-1:0] n4;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            n3 <= {L3_BITS{1'b0}}; n4 <= {L4_BITS{1'b0}};
            d3_go <= 1'b0; d4_go <= 1'b0;
        end else begin
            d3_go <= 1'b0; d4_go <= 1'b0;
            if (in_valid) begin
                n3 <= {L3_BITS{1'b0}};
                n4 <= {L4_BITS{1'b0}};
            end else begin
                if (a2_v) begin
                    if (n3 + {{(L3_BITS-1){1'b0}}, 1'b1} == L3_FULL) begin
                        n3    <= {L3_BITS{1'b0}};
                        d3_go <= 1'b1;
                    end else begin
                        n3 <= n3 + {{(L3_BITS-1){1'b0}}, 1'b1};
                    end
                end
                if (a3_v) begin
                    if (n4 + {{(L4_BITS-1){1'b0}}, 1'b1} == L4_FULL) begin
                        n4    <= {L4_BITS{1'b0}};
                        d4_go <= 1'b1;
                    end else begin
                        n4 <= n4 + {{(L4_BITS-1){1'b0}}, 1'b1};
                    end
                end
            end
        end
    end

    // ---- pool: sum over frames, no divide -------------------------------- //
    reg signed [POOL_BITS-1:0] pool [0:C4_OUT-1];
    localparam integer TF_BITS = (T_FRAMES <= 2) ? 1 : $clog2(T_FRAMES) + 1;
    localparam integer TF_I = T_FRAMES;
    localparam [TF_BITS-1:0] TF_FULL = TF_I[TF_BITS-1:0];
    reg [TF_BITS-1:0] frames;

    wire signed [POOL_BITS-1:0] a4_x =
        {{(POOL_BITS-A4_OUT){a4_val[A4_OUT-1]}}, a4_val};

    localparam integer C4_LAST_I = C4_OUT - 1;
    localparam [C4O_BITS-1:0] C4_LAST = C4_LAST_I[C4O_BITS-1:0];
    wire frame_done = a4_v && (a4_ch == C4_LAST);

    integer j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (j = 0; j < C4_OUT; j = j + 1) pool[j] <= {POOL_BITS{1'b0}};
            frames <= {TF_BITS{1'b0}};
        end else if (start) begin
            for (j = 0; j < C4_OUT; j = j + 1) pool[j] <= {POOL_BITS{1'b0}};
            frames <= {TF_BITS{1'b0}};
        end else begin
            if (a4_v) pool[a4_ch] <= pool[a4_ch] + a4_x;
            if (frame_done)
                frames <= frames + {{(TF_BITS-1){1'b0}}, 1'b1};
        end
    end

    // ---- argmax ---------------------------------------------------------- //
    // Ties go to the lower index, matching torch.argmax and
    // export/tailfmt.pooled_argmax. It is one scan per clip, so it walks the
    // twelve rather than building a comparator tree.
    localparam [1:0] S_IDLE = 2'd0, S_SCAN = 2'd1, S_DONE = 2'd2;
    reg [1:0]                  ast;
    reg [C4O_BITS-1:0]         scan, best;
    reg signed [POOL_BITS-1:0] best_v;

    wire pool_ready = frame_done && (frames + {{(TF_BITS-1){1'b0}}, 1'b1}
                                     == TF_FULL);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ast <= S_IDLE; scan <= {C4O_BITS{1'b0}}; best <= {C4O_BITS{1'b0}};
            best_v <= {POOL_BITS{1'b0}};
            class_valid <= 1'b0; class_idx <= {C4O_BITS{1'b0}};
        end else begin
            class_valid <= 1'b0;
            case (ast)
            S_IDLE:
                // the last class of the last frame has just been added, so the
                // pool is complete on the NEXT cycle -- start the scan there
                if (pool_ready) begin
                    scan   <= {C4O_BITS{1'b0}};
                    best   <= {C4O_BITS{1'b0}};
                    best_v <= {POOL_BITS{1'b1}} ^ {1'b0, {(POOL_BITS-1){1'b1}}};
                    ast    <= S_SCAN;
                end
            S_SCAN: begin
                if ($signed(pool[scan]) > best_v) begin
                    best_v <= pool[scan];
                    best   <= scan;
                end
                if (scan == C4_LAST) ast <= S_DONE;
                else scan <= scan + {{(C4O_BITS-1){1'b0}}, 1'b1};
            end
            S_DONE: begin
                class_idx   <= best;
                class_valid <= 1'b1;
                ast         <= S_IDLE;
            end
            default: ast <= S_IDLE;
            endcase
        end
    end

    // ---- busy ------------------------------------------------------------ //
    // One flag set by the push and cleared by the frame's last output, NOT an
    // OR of the sub-modules' busies. That OR looks equivalent and is not: the
    // chain has gaps where nothing is busy but a frame is still in flight --
    // the three affine stages carry no busy at all, and d3_go is registered, so
    // between the last a2_v and d3_busy rising every term of the OR is low.
    // kws_tcs_sub already cost a debugging session to a one-cycle hole of
    // exactly this shape, and a hole here would let the next frame overwrite
    // the frame buffer mid-sweep.
    reg in_flight;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          in_flight <= 1'b0;
        else if (in_valid)   in_flight <= 1'b1;
        else if (frame_done) in_flight <= 1'b0;
    end

    assign busy = in_flight | (ast != S_IDLE);

`ifdef KWS_ASSERT
    // The sub-modules' busies are not wired into `busy` -- see above for why an
    // OR of them has holes. They are not dead, though: they are how the claim
    // gets checked. in_flight must cover every cycle any stage is working, and
    // if it ever does not, the flag is missing something and the next push
    // lands on a stage still using its operands.
    //
    // Only this direction. in_flight high while nothing is busy is the normal
    // case -- those are exactly the gaps the flag exists to bridge.
    //
    // The edges line up: kws_pw_conv's `busy` is (st != S_IDLE) with st
    // registered, so it rises the cycle after in_valid, which is the same cycle
    // in_flight does.
    // No `rst_n &&` guard. It would make rst_n both an async reset (every other
    // block here) and a synchronous term, which lint rightly flags -- and it is
    // not needed: under reset every busy is low by construction (pw is
    // st != S_IDLE with st reset to S_IDLE, the dense convs are run|vB with
    // both reset), so the condition is already false. Before the first clock
    // the registers are X, and `if (X)` does not fire either.
    always @(posedge clk)
        if ((pw_busy | d3_busy | d4_busy) && !in_flight) begin
            $display("ASSERT %m: busy stage while idle: pw=%b d3=%b d4=%b",
                     pw_busy, d3_busy, d4_busy);
            $finish;
        end

    always @(posedge clk) if (in_valid && busy) begin
        $display("ASSERT %m: frame pushed while the tail is still working");
        $finish;
    end
    // The scan reads `pool` while nothing writes it. If a4 were still emitting
    // the argmax would be taken over a half-updated pool -- and a wrong class
    // is indistinguishable from a wrong network.
    always @(posedge clk) if ((ast == S_SCAN) && a4_v) begin
        $display("ASSERT %m: pool written during the argmax scan");
        $finish;
    end
`endif

endmodule

`default_nettype wire
