// conv1: int8 weights against +-1 activations, stride 2. No multiplier.
//
//     acc[o][t] = sum over i,k of  q[o][i][k] * x[i][2t - PAD + k]
//
// NO MULTIPLIER, BECAUSE ONE SIDE IS +-1 (CLAUDE.md 3.3). The layer is called
// int8 and it is, but an int8 weight times +-1 is that weight added or
// subtracted. A signed accumulate per term, and the DSPs stay idle.
//
// THIS IS THE MOST EXPENSIVE LAYER IN THE NETWORK, which is the opposite of the
// intuition that the first one is small. Expanding 16 channels to 128 with
// k=11 is 176 terms per output channel across 128 channels: 22,528 cycles a
// frame and 1.44M an inference, more than the entire tail. The manifest had
// been saying so all along -- n_terms is 7592 here against 128 for the next
// largest layer.
//
// WHY NOT kws_bin_mac's SHIFT-AND-MASK. A zero-padded tap contributes nothing,
// and `2*popcount(XNOR) - N` cannot express a zero term -- it assumes every
// term is +-1, so kws_dw_conv has to slide the valid window down to bit 0 and
// shrink N (rtl/README.md 1). Accumulating one term per cycle has no such
// problem: an invalid tap simply adds zero. The alignment machinery disappears.
//
// STRIDE 2 IS WHERE THE FRAMES HALVE. 128 input frames become 64 outputs, and
// output t is due after input frame 2t + PAD has been pushed -- so the first
// output waits for push 5 and the last needs push 131, which is four pushes
// past the last real frame. The caller supplies those four as flushes, the same
// contract kws_dw_conv uses.

`timescale 1ns/1ps
`default_nettype none

module kws_conv1 #(
    parameter integer C_IN     = 16,
    parameter integer C_OUT    = 128,
    parameter integer K        = 11,
    parameter integer PAD      = 5,
    parameter integer STRIDE   = 2,
    parameter integer T_IN     = 128,   // real input frames in a clip
    parameter integer W_BITS   = 8,     // manifest: roms[conv1_w].weight_bits
    parameter integer ACC_BITS = 14,    // manifest: conv1 acc_bits
    parameter integer CO_BITS  = 7,     // $clog2(C_OUT); see kws_affine on why
    parameter         W_FILE   = "",    // C_OUT*C_IN*K int8, [out][in][k]
    parameter         T_FILE   = ""     // C_OUT thresholds, then C_OUT polarity
) (
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire                 start,     // new clip: clear the line buffer
    input  wire                 in_push,
    input  wire                 in_real,   // 0 = a flush, to drain the tail
    input  wire [C_IN-1:0]      in_frame,  // bit i = channel i, -1 -> 0

    output wire                 busy,
    output reg                  out_valid,
    output reg  [C_OUT-1:0]     out_frame
);

    localparam integer N_W     = C_OUT * C_IN * K;
    localparam integer WA_BITS = (N_W <= 2) ? 1 : $clog2(N_W);
    localparam integer TK_BITS = (K <= 2) ? 1 : $clog2(K);
    localparam integer TI_BITS = (C_IN <= 2) ? 1 : $clog2(C_IN);
    // the push counter must reach STRIDE*(T_OUT-1) + PAD, past the last frame
    localparam integer P_LAST  = STRIDE * (((T_IN + 2*PAD - K) / STRIDE)) + PAD;
    localparam integer PC_BITS = (P_LAST <= 2) ? 1 : $clog2(P_LAST + 1);

    // ---- the line buffer ------------------------------------------------- //
    // Slot k holds the frame pushed K-1-k pushes ago, so after push p slot k is
    // input frame p - (K-1) + k. `valid` shadows it with in_real, which is what
    // marks the conv's own zero padding at both ends.
    reg [C_IN-1:0]     fbuf [0:K-1];
    reg [K-1:0]        vld;
    reg [PC_BITS-1:0]  pcnt;

    integer d;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || start) begin
            for (d = 0; d < K; d = d + 1) fbuf[d] <= {C_IN{1'b0}};
            vld  <= {K{1'b0}};
            pcnt <= {PC_BITS{1'b0}};
        end else if (in_push && !busy) begin
            for (d = 0; d < K - 1; d = d + 1) fbuf[d] <= fbuf[d + 1];
            fbuf[K-1] <= in_frame;
            vld  <= {vld[K-2:0], in_real};
            pcnt <= pcnt + {{(PC_BITS-1){1'b0}}, 1'b1};
        end
    end

    // An output is due once PAD taps have arrived, then every STRIDE pushes.
    // Comparing the low bit against PAD's is the stride-2 form of that: with
    // PAD=5 the outputs land on odd pushes 5, 7, ... 131.
    localparam integer PAD_I = PAD;
    localparam [PC_BITS-1:0] PAD_C = PAD_I[PC_BITS-1:0];
    wire emit_due = in_push && !busy && (pcnt >= PAD_C) &&
                    (pcnt[0] == PAD_C[0]);

    // ---- weights and thresholds ------------------------------------------ //
    reg signed [W_BITS-1:0] w [0:N_W-1];
    reg [31:0] t_rom [0:2*C_OUT-1];
    integer m;
    initial begin
        for (m = 0; m < N_W; m = m + 1) w[m] = {W_BITS{1'b0}};
        for (m = 0; m < 2 * C_OUT; m = m + 1) t_rom[m] = 32'h0;
        if (W_FILE != "") $readmemh(W_FILE, w, 0, N_W - 1);
        if (T_FILE != "") $readmemh(T_FILE, t_rom, 0, 2 * C_OUT - 1);
    end

    // ---- stage A: sweep the terms ---------------------------------------- //
    // co outermost, then the input channel, then the tap -- which is exactly
    // the weight ROM's [out][in][k] order, so one monotonic address counter
    // serves the whole sweep and there is no multiply to form it.
    reg                 run;
    reg [CO_BITS-1:0]   co;
    reg [TI_BITS-1:0]   ti;
    reg [TK_BITS-1:0]   tk;
    reg [WA_BITS-1:0]   wa;

    localparam integer CO_LAST_I = C_OUT - 1;
    localparam integer TI_LAST_I = C_IN - 1;
    localparam integer TK_LAST_I = K - 1;
    localparam [CO_BITS-1:0] CO_LAST = CO_LAST_I[CO_BITS-1:0];
    localparam [TI_BITS-1:0] TI_LAST = TI_LAST_I[TI_BITS-1:0];
    localparam [TK_BITS-1:0] TK_LAST = TK_LAST_I[TK_BITS-1:0];

    wire term_last = (ti == TI_LAST) && (tk == TK_LAST);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run <= 1'b0; co <= {CO_BITS{1'b0}}; ti <= {TI_BITS{1'b0}};
            tk  <= {TK_BITS{1'b0}}; wa <= {WA_BITS{1'b0}};
        end else if (start) begin
            run <= 1'b0;
        end else if (emit_due) begin
            run <= 1'b1; co <= {CO_BITS{1'b0}}; ti <= {TI_BITS{1'b0}};
            tk  <= {TK_BITS{1'b0}}; wa <= {WA_BITS{1'b0}};
        end else if (run) begin
            wa <= wa + {{(WA_BITS-1){1'b0}}, 1'b1};
            if (tk == TK_LAST) begin
                tk <= {TK_BITS{1'b0}};
                if (ti == TI_LAST) begin
                    ti <= {TI_BITS{1'b0}};
                    if (co == CO_LAST) run <= 1'b0;
                    else co <= co + {{(CO_BITS-1){1'b0}}, 1'b1};
                end else begin
                    ti <= ti + {{(TI_BITS-1){1'b0}}, 1'b1};
                end
            end else begin
                tk <= tk + {{(TK_BITS-1){1'b0}}, 1'b1};
            end
        end
    end

    // ---- stage B: add or subtract ---------------------------------------- //
    // The weight ROM is 180 Kbit, so it reads synchronously and the counters
    // run one term ahead -- same shape as kws_dense_conv.
    reg                     vB, lastB, xq, okq;
    reg [CO_BITS-1:0]       coB;
    reg signed [W_BITS-1:0] wq;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vB <= 1'b0; lastB <= 1'b0; xq <= 1'b0; okq <= 1'b0;
            coB <= {CO_BITS{1'b0}}; wq <= {W_BITS{1'b0}};
        end else begin
            vB <= run;
            if (run) begin
                wq    <= w[wa];
                xq    <= fbuf[tk][ti];    // bit i of the frame in tap k
                okq   <= vld[tk];         // 0 where the conv pads
                lastB <= term_last;
                coB   <= co;
            end
        end
    end

    wire signed [ACC_BITS-1:0] wx =
        {{(ACC_BITS-W_BITS){wq[W_BITS-1]}}, wq};
    // +-1 times a weight is the weight, signed. A padded tap adds nothing --
    // and note the negation happens at ACC_BITS, not at W_BITS, where -(-128)
    // would not fit.
    wire signed [ACC_BITS-1:0] term =
        !okq ? {ACC_BITS{1'b0}} : (xq ? wx : -wx);

    // One bit wider than the accumulator, so an overflow is VISIBLE. Adding at
    // ACC_BITS and then testing the result cannot work: the sum has already
    // wrapped into range by the time it is compared, and the check is
    // vacuously true. Same shape as kws_bin_mac's CHK_BITS.
    localparam integer CHK_BITS = ACC_BITS + 1;
    reg signed [ACC_BITS-1:0] acc;
    wire signed [CHK_BITS-1:0] sum_w =
        {{1{acc[ACC_BITS-1]}}, acc} + {{1{term[ACC_BITS-1]}}, term};
    wire signed [ACC_BITS-1:0] sum = sum_w[ACC_BITS-1:0];

    // ---- threshold -------------------------------------------------------- //
    localparam integer TH_BITS = (ACC_BITS + 1 > 33) ? ACC_BITS + 1 : 33;
    localparam integer AW_T = (2 * C_OUT <= 2) ? 1 : $clog2(2 * C_OUT);
    localparam integer C_OUT_I = C_OUT;
    localparam [AW_T-1:0] C_OUT_A = C_OUT_I[AW_T-1:0];
    wire [AW_T-1:0] th_a = {{(AW_T-CO_BITS){1'b0}}, coB};

    wire signed [31:0]        thr = t_rom[th_a];
    wire                      ge  = t_rom[th_a + C_OUT_A][0];
    wire signed [TH_BITS-1:0] sum_x = {{(TH_BITS-ACC_BITS){sum[ACC_BITS-1]}},
                                       sum};
    wire signed [TH_BITS-1:0] thr_x = {{(TH_BITS-32){thr[31]}}, thr};
    wire                      fired = ((sum_x >= thr_x) == ge);

    reg  emitting;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= {ACC_BITS{1'b0}}; out_valid <= 1'b0;
            out_frame <= {C_OUT{1'b0}}; emitting <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            if (start) begin
                acc <= {ACC_BITS{1'b0}}; emitting <= 1'b0;
            end else if (emit_due) begin
                acc <= {ACC_BITS{1'b0}}; emitting <= 1'b1;
            end else if (vB) begin
                if (lastB) begin
                    out_frame[coB] <= fired;
                    acc <= {ACC_BITS{1'b0}};
                    if (coB == CO_LAST) begin
                        out_valid <= 1'b1;
                        emitting  <= 1'b0;
                    end
                end else begin
                    acc <= sum;
                end
            end
        end
    end

    assign busy = run | vB | emitting;

`ifdef KWS_ASSERT
    initial if (CO_BITS != ((C_OUT <= 2) ? 1 : $clog2(C_OUT))) begin
        $display("ASSERT %m: CO_BITS=%0d does not match C_OUT=%0d",
                 CO_BITS, C_OUT);
        $finish;
    end
    always @(posedge clk) if (in_push && busy) begin
        $display("ASSERT %m: pushed while busy");
        $finish;
    end
    // The bound is max_o sum|q|, and a partial sum is a subset of those terms,
    // so it holds all the way through -- not only at the end.
    always @(posedge clk) if (vB) begin
        if (sum_w > $signed({2'b00, {(ACC_BITS-1){1'b1}}}) ||
            sum_w < $signed({2'b11, {(ACC_BITS-1){1'b0}}})) begin
            $display("ASSERT %m: accumulator %0d escaped ACC_BITS=%0d",
                     sum_w, ACC_BITS);
            $finish;
        end
    end
`endif

endmodule

`default_nettype wire
