// The analog boundary: sixteen comparator wires in, the network's tensor out.
//
//   cmp[k] -> [2FF sync] -> [sticky OR] -> [sample at the frame edge] -> frame[k]
//
// This is the whole of docs/ICD.md section 5, and it is the only module that
// touches anything asynchronous. Everything downstream of it is synchronous and
// knows nothing about the analog side -- which is the point: an analog change
// costs a retrain and new .hex files, never an edit to a .v.
//
// THE STICKY OR *IS* THE DISCRETISATION RULE. CLAUDE.md 2.8 says a window is 1
// if the comparator fired at any point inside it -- `reduce = max`, and on
// {0,1} max is exactly OR. So there is nothing to approximate here: the latch
// is the rule. It is cleared only at a frame edge, never mid-window.
//
// PADDING IS -1, NOT NEUTRAL. 100 native frames become 128 with 14 on each
// side, and those pad frames are -1 for every channel, which packs as an
// all-zero word. -1 is a real observation -- "no energy in this band" -- and
// conv1 was trained against it. A neutral value would be a different input than
// the one the weights were fitted to.
//
// FRAME_CYCLES COMES FROM THE BOARD, NOT THE MANIFEST. The manifest carries
// 10 ms (`KWS_FRAME_MS`); turning that into clocks needs the clock frequency,
// which is a board fact. At 100 MHz it is 1,000,000.
//
// It is also one of only three things in docs/ICD.md section 6 that an analog
// change can force into RTL, alongside N_CH and CMP_INVERT. Everything else the
// colleague might change moves weights instead.

`timescale 1ns/1ps
`default_nettype none

module kws_frame_ctrl #(
    parameter integer N_CH         = 16,
    parameter integer FRAME_CYCLES = 1000000,  // clk_hz * frame_ms / 1000
    parameter integer NATIVE_T     = 100,      // clip_ms / frame_ms
    parameter integer T            = 128,      // what the network expects
    parameter integer PAD_LEFT     = 14,
    // The comparator's sense. ICD section 6 lists this as one of the three
    // analog changes that reach RTL: if the detector ends up inverting, this
    // flips and nothing else does.
    parameter integer CMP_INVERT   = 0
) (
    input  wire             clk,
    input  wire             rst_n,

    input  wire             start,      // begin a clip
    input  wire [N_CH-1:0]  cmp,        // ASYNCHRONOUS comparator outputs

    // one frame at a time into kws_top, which answers with in_ready
    input  wire             out_ready,
    output reg              out_valid,
    output reg  [N_CH-1:0]  out_frame,
    output wire             busy
);

    localparam integer PAD_RIGHT = T - PAD_LEFT - NATIVE_T;

    // ---- 1. two-flop synchroniser ---------------------------------------- //
    // cmp has no relationship to this clock, so a single flop can go
    // metastable. Vivado needs the attribute to keep the pair together and to
    // stop it reporting a false path as a timing failure.
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

    // ---- 2. frame timer --------------------------------------------------- //
    localparam integer FC_BITS = (FRAME_CYCLES <= 2) ? 1 : $clog2(FRAME_CYCLES);
    localparam integer FC_LAST_I = FRAME_CYCLES - 1;
    localparam [FC_BITS-1:0] FC_LAST = FC_LAST_I[FC_BITS-1:0];

    localparam [2:0] S_IDLE = 3'd0, S_PADL = 3'd1, S_RUN = 3'd2,
                     S_PADR = 3'd3;
    reg [2:0]         st;
    reg [FC_BITS-1:0] fc;

    wire frame_edge = (st == S_RUN) && (fc == FC_LAST);

    // ---- 3. the sticky OR ------------------------------------------------- //
    // The cycle at the boundary belongs to the window that is closing, which is
    // why the capture ORs in sync2 and the clear goes to zero rather than to
    // sync2. Every cycle lands in exactly one window: none counted twice, none
    // dropped.
    reg [N_CH-1:0] sticky;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || start)     sticky <= {N_CH{1'b0}};
        else if (frame_edge)     sticky <= {N_CH{1'b0}};
        else if (st == S_RUN)    sticky <= sticky | sync2;
    end

    // ---- 4. the sequence: 14 pads, 100 frames, 14 pads -------------------- //
    localparam integer TB_BITS = (T <= 2) ? 1 : $clog2(T) + 1;
    localparam integer PL_I = PAD_LEFT, NT_I = NATIVE_T, PR_I = PAD_RIGHT;
    localparam [TB_BITS-1:0] PL_C = PL_I[TB_BITS-1:0];
    localparam [TB_BITS-1:0] NT_C = NT_I[TB_BITS-1:0];
    localparam [TB_BITS-1:0] PR_C = PR_I[TB_BITS-1:0];
    localparam [TB_BITS-1:0] ONE  = {{(TB_BITS-1){1'b0}}, 1'b1};

    reg [TB_BITS-1:0] cnt;      // frames emitted within the current phase
    reg               pending;  // a frame is waiting for the consumer
    reg [N_CH-1:0]    held;

    wire take = pending && out_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; fc <= {FC_BITS{1'b0}}; cnt <= {TB_BITS{1'b0}};
            pending <= 1'b0; held <= {N_CH{1'b0}};
            out_valid <= 1'b0; out_frame <= {N_CH{1'b0}};
        end else begin
            out_valid <= 1'b0;

            if (start) begin
                // the left padding is emitted at once; there is no time
                // associated with it, only with the 100 real windows
                st <= S_PADL; fc <= {FC_BITS{1'b0}}; cnt <= {TB_BITS{1'b0}};
                pending <= 1'b1; held <= {N_CH{1'b0}};
            end else begin
                if (take) begin
                    out_valid <= 1'b1;
                    out_frame <= held;
                    pending   <= 1'b0;
                    cnt       <= cnt + ONE;
                end

                case (st)
                S_PADL:
                    // re-arm after every take, or the phase stops after one
                    // frame: `take` clears pending and only this puts it back
                    if (take) begin
                        if (cnt + ONE == PL_C) begin
                            st  <= S_RUN;
                            cnt <= {TB_BITS{1'b0}};
                            fc  <= {FC_BITS{1'b0}};
                            pending <= 1'b0;   // the next frame is time-driven
                        end else begin
                            pending <= 1'b1;
                            held    <= {N_CH{1'b0}};
                        end
                    end
                S_RUN: begin
                    fc <= (fc == FC_LAST) ? {FC_BITS{1'b0}}
                                          : fc + {{(FC_BITS-1){1'b0}}, 1'b1};
                    if (frame_edge) begin
                        held    <= sticky | sync2;
                        pending <= 1'b1;
                    end
                    if (take && (cnt + ONE == NT_C)) begin
                        st  <= S_PADR;
                        cnt <= {TB_BITS{1'b0}};
                        // the right padding, like the left, is time-free
                        pending <= 1'b1;
                        held    <= {N_CH{1'b0}};
                    end
                end
                S_PADR:
                    if (take && (cnt + ONE == PR_C)) begin
                        st      <= S_IDLE;
                        pending <= 1'b0;
                    end else if (take) begin
                        pending <= 1'b1;
                        held    <= {N_CH{1'b0}};
                    end
                default: st <= S_IDLE;
                endcase
            end
        end
    end

    assign busy = (st != S_IDLE);

`ifdef KWS_ASSERT
    initial if (PAD_LEFT + NATIVE_T + PAD_RIGHT != T) begin
        $display("ASSERT %m: %0d + %0d + %0d does not make T=%0d",
                 PAD_LEFT, NATIVE_T, PAD_RIGHT, T);
        $finish;
    end
    // A window closing while the previous frame is still waiting means the
    // network did not finish inside FRAME_CYCLES -- a real-time violation, and
    // the one failure here that is about the SYSTEM rather than this module.
    // It would otherwise show up as a frame quietly overwritten.
    always @(posedge clk) if (frame_edge && pending) begin
        $display("ASSERT %m: frame boundary while the previous frame is still "
                 "unread -- the network is slower than FRAME_CYCLES");
        $finish;
    end
    always @(posedge clk) if (start && busy) begin
        $display("ASSERT %m: restarted mid-clip");
        $finish;
    end
`endif

endmodule

`default_nettype wire
