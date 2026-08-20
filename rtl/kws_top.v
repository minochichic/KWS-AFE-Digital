// The whole network: 16 comparator bits per frame in, a class index out.
//
//   AFE frames  -> conv1 -> [A] -> b1 -> [B] -> b2 -> [C] -> b3 -> [D]
//               -> conv2_dw -> kws_tail -> class
//
// FIVE PHASES, STRICTLY SEQUENTIAL. Layer L finishes the whole clip before
// L+1 starts reading, which is what lets each junction be one plane rather
// than a matched pair of arrival times (docs/diagrams/28_plane_buffer.svg).
// No ping-pong: that only buys overlapping one clip with the next, and at 10 Hz
// there is nothing to overlap. Four planes, 20 Kbit total.
//
// EACH PLANE'S FLUSH IS ITS CONSUMER'S, NOT ITS PRODUCER'S. A block with two
// depthwise stages needs 2*PAD flush pushes to drain; conv2_dw, with one, needs
// PAD. So the four are 12, 14, 16, 28 -- all different, all from the manifest.
//
// conv1's OWN FLUSHES ARE GENERATED HERE. Its last output needs input push
// 2*(T_OUT-1) + PAD = 131, four past the last real frame, and the caller has
// only T_IN frames to give. Asking the caller for four empty pushes would put a
// property of conv1's kernel into the AFE's interface, where it does not
// belong: what crosses that boundary is 16 comparator wires and frame timing,
// nothing about the network (docs/ICD.md).
//
// THE PARAMETER LIST IS LONG ON PURPOSE. Every width, kernel, padding and ROM
// path arrives from parameters.vh (CLAUDE.md 5), and there are twenty-one
// layers. Burying any of it here would mean a retrain needs an RTL edit, which
// is the one thing this structure exists to prevent.

`timescale 1ns/1ps
`default_nettype none

module kws_top #(
    parameter integer WORD_BITS = 32,
    parameter integer T_IN      = 128,   // AFE frames per clip
    parameter integer T_OUT     = 64,    // frames after conv1's stride

    // ---- conv1 ----------------------------------------------------------- //
    parameter integer N_CH      = 16,
    parameter integer C1_OUT    = 128,
    parameter integer C1_K      = 11,
    parameter integer C1_PAD    = 5,
    parameter integer C1_STRIDE = 2,
    parameter integer C1_ACC    = 14,
    parameter         C1_W      = "",
    parameter         C1_T      = "",

    // ---- b1 -------------------------------------------------------------- //
    parameter integer B1_MID    = 64,
    parameter integer B1_OUT    = 64,
    parameter integer B1_K      = 13,
    parameter integer B1_PAD    = 6,
    parameter integer B1_S0DW_A = 5, parameter integer B1_S0PW_A = 9,
    parameter integer B1_S1DW_A = 5, parameter integer B1_S1PW_A = 8,
    parameter integer B1_SKIP_A = 9, parameter integer B1_ADD_A  = 9,
    parameter B1_S0DW_W = "", parameter B1_S0DW_T = "",
    parameter B1_S0PW_W = "", parameter B1_S0PW_T = "",
    parameter B1_S1DW_W = "", parameter B1_S1DW_T = "",
    parameter B1_S1PW_W = "", parameter B1_SKIP_W = "", parameter B1_ADD_T = "",

    // ---- b2 -------------------------------------------------------------- //
    parameter integer B2_K      = 15,
    parameter integer B2_PAD    = 7,
    parameter integer B2_S0DW_A = 5, parameter integer B2_S0PW_A = 8,
    parameter integer B2_S1DW_A = 5, parameter integer B2_S1PW_A = 8,
    parameter integer B2_SKIP_A = 8, parameter integer B2_ADD_A  = 8,
    parameter B2_S0DW_W = "", parameter B2_S0DW_T = "",
    parameter B2_S0PW_W = "", parameter B2_S0PW_T = "",
    parameter B2_S1DW_W = "", parameter B2_S1DW_T = "",
    parameter B2_S1PW_W = "", parameter B2_ADD_T = "",

    // ---- b3 -------------------------------------------------------------- //
    parameter integer B3_K      = 17,
    parameter integer B3_PAD    = 8,
    parameter integer B3_S0DW_A = 6, parameter integer B3_S0PW_A = 8,
    parameter integer B3_S1DW_A = 6, parameter integer B3_S1PW_A = 8,
    parameter integer B3_SKIP_A = 8, parameter integer B3_ADD_A  = 8,
    parameter B3_S0DW_W = "", parameter B3_S0DW_T = "",
    parameter B3_S0PW_W = "", parameter B3_S0PW_T = "",
    parameter B3_S1DW_W = "", parameter B3_S1DW_T = "",
    parameter B3_S1PW_W = "", parameter B3_ADD_T = "",

    // ---- conv2_dw -------------------------------------------------------- //
    parameter integer C2_K      = 29,
    parameter integer C2_PAD    = 28,
    parameter integer C2_DIL    = 2,
    parameter integer C2_ACC    = 6,
    parameter         C2_W      = "",
    parameter         C2_T      = "",

    // ---- the tail (passed straight through to kws_tail) ------------------ //
    parameter integer TL_C2_OUT = 128,
    parameter integer TL_C2_ACC = 8,
    parameter         TL_C2_W   = "",
    parameter integer TL_A2_G   = 22, parameter integer TL_A2_B = 26,
    parameter integer TL_A2_S   = 18, parameter integer TL_A2_O = 14,
    parameter         TL_A2_F   = "",
    parameter integer TL_C3_OUT = 128, parameter integer TL_C3_W = 8,
    parameter integer TL_C3_ACC = 28,  parameter         TL_C3_WF = "",
    parameter integer TL_A3_G   = 18, parameter integer TL_A3_B = 32,
    parameter integer TL_A3_S   = 24, parameter integer TL_A3_O = 11,
    parameter         TL_A3_F   = "",
    parameter integer TL_C4_OUT = 12, parameter integer TL_C4_W = 8,
    parameter integer TL_C4_ACC = 25, parameter         TL_C4_WF = "",
    parameter integer TL_A4_G   = 23, parameter integer TL_A4_B = 31,
    parameter integer TL_A4_S   = 27, parameter integer TL_A4_O = 14,
    parameter         TL_A4_F   = "",
    parameter integer TL_POOL   = 21,
    parameter integer TL_C4O_B  = 4,
    // A stall is the one failure the other assertions cannot see: they all
    // check a value, and a stall has no wrong value to check. This bounds how
    // long any phase may take, so a hang names its phase instead of running out
    // the testbench clock in silence.
    parameter integer PHASE_LIMIT = 4000000
) (
    input  wire                clk,
    input  wire                rst_n,

    input  wire                start,     // new clip
    input  wire                in_valid,  // one AFE frame, +-1, N_CH wide
    input  wire [N_CH-1:0]     in_frame,
    // "a frame offered THIS cycle will be taken". Not derivable from `busy`,
    // which is high for the whole clip, and not from conv1's busy either: that
    // rises a cycle after the push reaches it, so a caller watching it pushes
    // again into the gap and the frame is dropped without a trace.
    output wire                in_ready,
    output wire                busy,

    output wire                class_valid,
    output wire [TL_C4O_B-1:0] class_idx
);

    localparam integer C1_FLUSH = C1_STRIDE * (T_OUT - 1) + C1_PAD - (T_IN - 1);
    localparam [2:0] S_IDLE = 3'd0, S_C1 = 3'd1, S_B1 = 3'd2, S_B2 = 3'd3,
                     S_B3 = 3'd4, S_TL = 3'd5;
    reg [2:0] st;

    // ---- phase 1: conv1 -> plane A --------------------------------------- //
    // The caller supplies T_IN frames; the four pushes conv1 still needs to
    // reach its last output are generated below, which is why in_valid and
    // c1_push are not the same wire.
    localparam integer PC_BITS = 9;
    // sized through an integer part-select, not written as an expression: the
    // sum is 32 bits and comparing it against a 9-bit counter is a widening
    // lint would rightly flag
    localparam integer PC_LIMIT_I = T_IN + C1_FLUSH + 2;
    localparam [PC_BITS-1:0] PC_LIMIT = PC_LIMIT_I[PC_BITS-1:0];
    reg [PC_BITS-1:0] pc;
    reg               c1_push, c1_real;
    reg [N_CH-1:0]    c1_frame;

    wire              c1_busy, c1_ov;
    wire [C1_OUT-1:0] c1_of;

    // ONE condition, used by both push paths and by in_ready. Writing it twice
    // is how the flush branch ended up without the !c1_push term: the caller's
    // path was fixed and the internal one was not, so every real frame passed
    // and the first flush issued a two-cycle in_push -- one intended push,
    // two shifts.
    //
    // Two different cycles have to be excluded. c1_push is registered, so the
    // cycle after it is set conv1 has not seen it yet and c1_busy is still low;
    // and once conv1 is sweeping, c1_busy covers the rest.
    wire can_push = (st == S_C1) && !c1_busy && !c1_push;

    kws_conv1 #(.C_IN(N_CH), .C_OUT(C1_OUT), .K(C1_K), .PAD(C1_PAD),
                .STRIDE(C1_STRIDE), .T_IN(T_IN), .W_BITS(8),
                .ACC_BITS(C1_ACC), .CO_BITS(7),
                .W_FILE(C1_W), .T_FILE(C1_T)) u_c1 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .in_push(c1_push), .in_real(c1_real), .in_frame(c1_frame),
        .busy(c1_busy), .out_valid(c1_ov), .out_frame(c1_of));

    // ---- the four planes -------------------------------------------------- //
    reg  pa_ws, pb_ws, pc_ws, pd_ws;   // write-side start
    reg  pa_rs, pb_rs, pc_rs, pd_rs;   // read-side start

    wire pa_full, pb_full, pc_full, pd_full;
    wire pa_push, pb_push, pc_push, pd_push;
    wire pa_real, pb_real, pc_real, pd_real;
    wire pa_done, pb_done, pc_done, pd_done;
    wire [C1_OUT-1:0] pa_frame;
    wire [B1_OUT-1:0] pb_frame, pc_frame, pd_frame;

    wire b1_busy, b1_ov;  wire [B1_OUT-1:0] b1_of;
    wire b2_busy, b2_ov;  wire [B1_OUT-1:0] b2_of;
    wire b3_busy, b3_ov;  wire [B1_OUT-1:0] b3_of;
    wire c2_busy, c2_ov;  wire [B1_OUT-1:0] c2_of;
    wire tl_busy;

    kws_plane #(.C(C1_OUT), .T(T_OUT), .FLUSH(2*B1_PAD)) u_pa (
        .clk(clk), .rst_n(rst_n),
        .wr_start(pa_ws), .wr_valid(c1_ov), .wr_frame(c1_of),
        .wr_full(pa_full),
        .rd_start(pa_rs), .rd_ready(!b1_busy),
        .rd_push(pa_push), .rd_real(pa_real), .rd_frame(pa_frame),
        .rd_done(pa_done));

    kws_plane #(.C(B1_OUT), .T(T_OUT), .FLUSH(2*B2_PAD)) u_pb (
        .clk(clk), .rst_n(rst_n),
        .wr_start(pb_ws), .wr_valid(b1_ov), .wr_frame(b1_of),
        .wr_full(pb_full),
        .rd_start(pb_rs), .rd_ready(!b2_busy),
        .rd_push(pb_push), .rd_real(pb_real), .rd_frame(pb_frame),
        .rd_done(pb_done));

    kws_plane #(.C(B1_OUT), .T(T_OUT), .FLUSH(2*B3_PAD)) u_pc (
        .clk(clk), .rst_n(rst_n),
        .wr_start(pc_ws), .wr_valid(b2_ov), .wr_frame(b2_of),
        .wr_full(pc_full),
        .rd_start(pc_rs), .rd_ready(!b3_busy),
        .rd_push(pc_push), .rd_real(pc_real), .rd_frame(pc_frame),
        .rd_done(pc_done));

    // conv2_dw has ONE depthwise, so its drain is PAD, not 2*PAD. And its
    // consumer downstream is kws_tail, which is busy far longer than conv2_dw
    // is -- so the plane has to wait on both.
    //
    // c2_ov as well as the two busies. kws_dw_conv drops its busy and raises
    // out_valid on the SAME edge, and kws_tail's in_flight is set the edge
    // after, so there is exactly one cycle where neither looks busy and a frame
    // is in flight between them. The plane samples rd_ready in S_WAIT and
    // pushes two cycles later, which is long enough to land in it.
    //
    // Fourth instance of the same shape in this design: two registered signals
    // that are supposed to hand over leave a cycle between them, and something
    // walks in. kws_tcs_sub's busy hole, kws_top's in_ready, its flush guard,
    // and now this one.
    wire tail_chain_busy = c2_busy | tl_busy | c2_ov;

    kws_plane #(.C(B1_OUT), .T(T_OUT), .FLUSH(C2_PAD)) u_pd (
        .clk(clk), .rst_n(rst_n),
        .wr_start(pd_ws), .wr_valid(b3_ov), .wr_frame(b3_of),
        .wr_full(pd_full),
        .rd_start(pd_rs), .rd_ready(!tail_chain_busy),
        .rd_push(pd_push), .rd_real(pd_real), .rd_frame(pd_frame),
        .rd_done(pd_done));

    // ---- the three residual blocks ---------------------------------------- //
    kws_block #(.C_IN(C1_OUT), .C_MID(B1_MID), .C_OUT(B1_OUT), .K(B1_K),
                .PAD(B1_PAD), .S0_DW_ACC(B1_S0DW_A), .S0_PW_ACC(B1_S0PW_A),
                .S1_DW_ACC(B1_S1DW_A), .S1_PW_ACC(B1_S1PW_A),
                .SKIP_ACC(B1_SKIP_A), .ADD_ACC(B1_ADD_A),
                .WORD_BITS(WORD_BITS),
                .S0_DW_W(B1_S0DW_W), .S0_DW_T(B1_S0DW_T),
                .S0_PW_W(B1_S0PW_W), .S0_PW_T(B1_S0PW_T),
                .S1_DW_W(B1_S1DW_W), .S1_DW_T(B1_S1DW_T),
                .S1_PW_W(B1_S1PW_W), .SKIP_W(B1_SKIP_W),
                .ADD_T(B1_ADD_T)) u_b1 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .in_push(pa_push), .in_real(pa_real), .in_frame(pa_frame),
        .busy(b1_busy), .out_valid(b1_ov), .out_frame(b1_of));

    // b2 and b3 keep the channel count, so their skip is the identity and
    // SKIP_W is empty -- kws_block adds the block input itself.
    kws_block #(.C_IN(B1_OUT), .C_MID(B1_OUT), .C_OUT(B1_OUT), .K(B2_K),
                .PAD(B2_PAD), .S0_DW_ACC(B2_S0DW_A), .S0_PW_ACC(B2_S0PW_A),
                .S1_DW_ACC(B2_S1DW_A), .S1_PW_ACC(B2_S1PW_A),
                .SKIP_ACC(B2_SKIP_A), .ADD_ACC(B2_ADD_A),
                .WORD_BITS(WORD_BITS),
                .S0_DW_W(B2_S0DW_W), .S0_DW_T(B2_S0DW_T),
                .S0_PW_W(B2_S0PW_W), .S0_PW_T(B2_S0PW_T),
                .S1_DW_W(B2_S1DW_W), .S1_DW_T(B2_S1DW_T),
                .S1_PW_W(B2_S1PW_W), .SKIP_W(""), .SKIP_ID(1),
                .ADD_T(B2_ADD_T)) u_b2 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .in_push(pb_push), .in_real(pb_real), .in_frame(pb_frame),
        .busy(b2_busy), .out_valid(b2_ov), .out_frame(b2_of));

    kws_block #(.C_IN(B1_OUT), .C_MID(B1_OUT), .C_OUT(B1_OUT), .K(B3_K),
                .PAD(B3_PAD), .S0_DW_ACC(B3_S0DW_A), .S0_PW_ACC(B3_S0PW_A),
                .S1_DW_ACC(B3_S1DW_A), .S1_PW_ACC(B3_S1PW_A),
                .SKIP_ACC(B3_SKIP_A), .ADD_ACC(B3_ADD_A),
                .WORD_BITS(WORD_BITS),
                .S0_DW_W(B3_S0DW_W), .S0_DW_T(B3_S0DW_T),
                .S0_PW_W(B3_S0PW_W), .S0_PW_T(B3_S0PW_T),
                .S1_DW_W(B3_S1DW_W), .S1_DW_T(B3_S1DW_T),
                .S1_PW_W(B3_S1PW_W), .SKIP_W(""), .SKIP_ID(1),
                .ADD_T(B3_ADD_T)) u_b3 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .in_push(pc_push), .in_real(pc_real), .in_frame(pc_frame),
        .busy(b3_busy), .out_valid(b3_ov), .out_frame(b3_of));

    // ---- conv2_dw, then the tail ------------------------------------------ //
    kws_dw_conv #(.C(B1_OUT), .K(C2_K), .PAD(C2_PAD), .DIL(C2_DIL),
                  .ACC_BITS(C2_ACC), .WORD_BITS(WORD_BITS),
                  .W_FILE(C2_W), .T_FILE(C2_T)) u_c2 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .in_push(pd_push), .in_real(pd_real), .in_frame(pd_frame),
        .busy(c2_busy), .out_valid(c2_ov), .out_frame(c2_of));

    kws_tail #(.C2_IN(B1_OUT), .C2_OUT(TL_C2_OUT), .C2_ACC(TL_C2_ACC),
               .WORD_BITS(WORD_BITS), .C2_W_FILE(TL_C2_W),
               .A2_GAIN(TL_A2_G), .A2_BIAS(TL_A2_B), .A2_SHIFT(TL_A2_S),
               .A2_OUT(TL_A2_O), .A2_FILE(TL_A2_F),
               .C3_OUT(TL_C3_OUT), .C3_W(TL_C3_W), .C3_ACC(TL_C3_ACC),
               .C3_W_FILE(TL_C3_WF),
               .A3_GAIN(TL_A3_G), .A3_BIAS(TL_A3_B), .A3_SHIFT(TL_A3_S),
               .A3_OUT(TL_A3_O), .A3_FILE(TL_A3_F),
               .C4_OUT(TL_C4_OUT), .C4_W(TL_C4_W), .C4_ACC(TL_C4_ACC),
               .C4_W_FILE(TL_C4_WF),
               .A4_GAIN(TL_A4_G), .A4_BIAS(TL_A4_B), .A4_SHIFT(TL_A4_S),
               .A4_OUT(TL_A4_O), .A4_FILE(TL_A4_F),
               .T_FRAMES(T_OUT), .POOL_BITS(TL_POOL),
               .C2O_BITS(7), .C3O_BITS(7), .C4O_BITS(TL_C4O_B)) u_tail (
        .clk(clk), .rst_n(rst_n),
        .start(start), .in_valid(c2_ov), .in_frame(c2_of), .busy(tl_busy),
        .class_valid(class_valid), .class_idx(class_idx));

    // ---- the sequencer ----------------------------------------------------- //
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; pc <= {PC_BITS{1'b0}};
            c1_push <= 1'b0; c1_real <= 1'b0; c1_frame <= {N_CH{1'b0}};
            pa_ws <= 1'b0; pb_ws <= 1'b0; pc_ws <= 1'b0; pd_ws <= 1'b0;
            pa_rs <= 1'b0; pb_rs <= 1'b0; pc_rs <= 1'b0; pd_rs <= 1'b0;
        end else begin
            c1_push <= 1'b0;
            pa_ws <= 1'b0; pb_ws <= 1'b0; pc_ws <= 1'b0; pd_ws <= 1'b0;
            pa_rs <= 1'b0; pb_rs <= 1'b0; pc_rs <= 1'b0; pd_rs <= 1'b0;

            if (start) begin
                st <= S_C1; pc <= {PC_BITS{1'b0}};
                pa_ws <= 1'b1; pb_ws <= 1'b1; pc_ws <= 1'b1; pd_ws <= 1'b1;
            end else case (st)
            S_C1: begin
                // real frames come from the caller; the tail flushes are ours
                if (in_valid && can_push && pc < T_IN[PC_BITS-1:0]) begin
                    c1_push  <= 1'b1; c1_real <= 1'b1; c1_frame <= in_frame;
                    pc       <= pc + {{(PC_BITS-1){1'b0}}, 1'b1};
                // `!c1_ov` as well as `!pa_full`: the plane's wr_full lags
                // conv1's out_valid by a cycle, so without it the FSM sees room
                // that the frame in flight has already claimed and issues one
                // flush too many. Harmless here -- conv1 ignores an off-phase
                // push -- but it makes pc land somewhere the arithmetic does
                // not predict, which is worse than the wasted cycle.
                end else if (can_push && pc >= T_IN[PC_BITS-1:0] &&
                             !pa_full && !c1_ov) begin
                    c1_push  <= 1'b1; c1_real <= 1'b0;
                    c1_frame <= {N_CH{1'b0}};
                    pc       <= pc + {{(PC_BITS-1){1'b0}}, 1'b1};
                // `pc >= T_IN` as well: on the second clip pa_full is still
                // set from the first one when S_C1 is re-entered, because the
                // plane's wr_start only lands a cycle later. Without the
                // counter the phase would end before a single frame arrived.
                end else if (pa_full && !c1_busy &&
                             pc >= T_IN[PC_BITS-1:0]) begin
                    st    <= S_B1;
                    pa_rs <= 1'b1;
                end
            end
            S_B1: if (pa_seen && pb_full) begin st <= S_B2; pb_rs <= 1'b1; end
            S_B2: if (pb_seen && pc_full) begin st <= S_B3; pc_rs <= 1'b1; end
            S_B3: if (pc_seen && pd_full) begin st <= S_TL; pd_rs <= 1'b1; end
            S_TL: if (class_valid) st <= S_IDLE;
            default: st <= S_IDLE;
            endcase
        end
    end

    assign in_ready = can_push && (pc < T_IN[PC_BITS-1:0]);

    assign busy = (st != S_IDLE);

    // ---- phase watchdog ---------------------------------------------------- //
    reg [23:0] phase_cyc;
    reg [2:0]  st_q;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase_cyc <= 24'd0; st_q <= S_IDLE;
        end else begin
            st_q <= st;
            if (st != st_q || st == S_IDLE) phase_cyc <= 24'd0;
            else phase_cyc <= phase_cyc + 24'd1;
        end
    end

    // Every rd_done is a ONE-CYCLE PULSE, and every phase ends on "the plane
    // finished pushing AND the next plane filled". Those are never true in the
    // same cycle: the done fires when the last frame is handed over, and the
    // consumer needs another ~1,600 cycles to turn it into the last write. So
    // the pulses are latched, and the phase tests the latch.
    //
    // Phase 5 had this from the start (pd_seen) and phases 2-4 did not, which
    // is how a condition that can never be satisfied got written three times.
    reg pa_seen, pb_seen, pc_seen, pd_seen;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || start) begin
            pa_seen <= 1'b0; pb_seen <= 1'b0;
            pc_seen <= 1'b0; pd_seen <= 1'b0;
        end else begin
            if (pa_done) pa_seen <= 1'b1;
            if (pb_done) pb_seen <= 1'b1;
            if (pc_done) pc_seen <= 1'b1;
            if (pd_done) pd_seen <= 1'b1;
        end
    end

`ifdef KWS_ASSERT
    // A plane must be full before its reader starts, and kws_plane asserts that
    // too -- but the useful place to catch it is here, where the phase that
    // failed to fill it has a name.
    always @(posedge clk) if (pa_rs && !pa_full) begin
        $display("ASSERT %m: b1 started before plane A filled"); $finish;
    end
    always @(posedge clk) if (pb_rs && !pb_full) begin
        $display("ASSERT %m: b2 started before plane B filled"); $finish;
    end
    always @(posedge clk) if (pc_rs && !pc_full) begin
        $display("ASSERT %m: b3 started before plane C filled"); $finish;
    end
    always @(posedge clk) if (pd_rs && !pd_full) begin
        $display("ASSERT %m: the tail started before plane D filled"); $finish;
    end
    // The drop that caused a hang rather than a failure. There is no else on
    // the S_C1 push branch, so a frame offered while conv1 is busy vanishes and
    // pc stops short -- plane A never fills and the phase waits forever.
    always @(posedge clk) if (in_valid && !in_ready) begin
        $display("ASSERT %m: frame offered while not ready -- it is dropped");
        $finish;
    end
    always @(posedge clk) if (phase_cyc > PHASE_LIMIT[23:0]) begin
        $display("ASSERT %m: phase %0d stalled for %0d cycles (pc=%0d)",
                 st, phase_cyc, pc);
        $finish;
    end
    always @(posedge clk) if (class_valid && !pd_seen) begin
        $display("ASSERT %m: a class came out before plane D finished");
        $finish;
    end
    // conv1's flush count is derived; if it is wrong the plane never fills and
    // the run hangs in S_C1 rather than failing, which is much harder to read.
    always @(posedge clk)
        if ((st == S_C1) && (pc > PC_LIMIT)) begin
            $display("ASSERT %m: %0d pushes and plane A still not full",
                     pc);
            $finish;
        end
`endif

endmodule

`default_nettype wire
