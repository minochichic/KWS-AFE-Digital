// Binary MAC: the one primitive every binary layer is built from.
//
//   acc = 2 * popcount(act XNOR wgt) - N
//
// which equals sum(a_i * w_i) when both are +-1. No multiplier exists in this
// design; this is why.
//
// PACKING CONVENTION (export/pack.py, and tests/test_export.py pins it):
//   -1 -> bit 0, +1 -> bit 1, packed along the last axis, LSB FIRST.
//   Element i of the vector is bit (i % 32) of word (i / 32).
//
// THE TAIL. A vector whose length is not a multiple of WORD_BITS is
// zero-padded, and a zero bit is a real -1 -- it would contribute to the sum.
// The mask is applied HERE rather than left to the caller, because forgetting
// it produces a small, plausible, data-dependent error rather than an obvious
// failure. `n_valid` is the real term count and also the N in 2*P - N.
//
// Streaming, not combinational: the datapath is folded (CLAUDE.md 0), so one
// engine is time-shared across layers and the vector arrives a word at a time.
//
// ACC_BITS comes from parameters.vh, which export/emit.py generates from the
// analytic bound +-n_terms. Do not widen it by hand -- regenerate.

`default_nettype none

module kws_bin_mac #(
    parameter integer WORD_BITS = 32,
    parameter integer ACC_BITS  = 16,     // signed, holds [-n_valid, +n_valid]
    parameter integer CNT_BITS  = 16
) (
    input  wire                          clk,
    input  wire                          rst_n,

    // start pulses with the vector length; it is registered, so the caller may
    // change n_valid for the next layer while this one is still streaming
    input  wire                          start,
    input  wire [CNT_BITS-1:0]           n_valid,

    input  wire                          in_valid,
    input  wire [WORD_BITS-1:0]          act,
    input  wire [WORD_BITS-1:0]          wgt,

    output reg                           out_valid,
    output reg  signed [ACC_BITS-1:0]    acc
);

    // ---- popcount over one word -------------------------------------- //
    // A plain loop; synthesis infers the adder tree. $countones is
    // SystemVerilog and this stays Verilog-2001 so iverilog and Vivado agree.
    function [CNT_BITS-1:0] popcount;
        input [WORD_BITS-1:0] v;
        integer i;
        begin
            popcount = {CNT_BITS{1'b0}};
            for (i = 0; i < WORD_BITS; i = i + 1)
                popcount = popcount + {{(CNT_BITS-1){1'b0}}, v[i]};
        end
    endfunction

    reg [CNT_BITS-1:0] left;      // terms still to come
    reg [CNT_BITS-1:0] total;     // N, latched at start
    reg [CNT_BITS-1:0] ones;      // popcount so far

    // A sized net, not a part-select on the parameter: bit-selecting a
    // parameter is not portable across simulators.
    wire [CNT_BITS-1:0] wb = WORD_BITS[CNT_BITS-1:0];
    wire                full = (left >= wb);

    // The tail mask. Zero padding decodes as -1 and would contribute to the
    // sum, so the last partial word is masked here rather than by the caller.
    wire [WORD_BITS-1:0] mask = full ? {WORD_BITS{1'b1}}
                                     : ~({WORD_BITS{1'b1}} << left);

    wire [WORD_BITS-1:0] matched = ~(act ^ wgt) & mask;
    wire [CNT_BITS-1:0]  step    = full ? wb : left;
    wire [CNT_BITS-1:0]  ones_n  = ones + popcount(matched);
    wire                 last    = in_valid && (left <= wb);

    // 2*P - N at a width that provably holds it: P and N are CNT_BITS wide, so
    // 2*P needs CNT_BITS+1 and the signed difference CNT_BITS+2. ACC_BITS may
    // legitimately be narrower -- export/emit.py sizes it from the bound
    // +-n_terms -- so the narrowing is explicit above, not implicit here.
    localparam integer TMP_BITS = (CNT_BITS + 2 > ACC_BITS) ? CNT_BITS + 2
                                                            : ACC_BITS;
    wire signed [TMP_BITS-1:0] acc_full =
        $signed({{(TMP_BITS-CNT_BITS){1'b0}}, ones_n})
      + $signed({{(TMP_BITS-CNT_BITS){1'b0}}, ones_n})
      - $signed({{(TMP_BITS-CNT_BITS){1'b0}}, total});

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            left      <= {CNT_BITS{1'b0}};
            total     <= {CNT_BITS{1'b0}};
            ones      <= {CNT_BITS{1'b0}};
            acc       <= {ACC_BITS{1'b0}};
            out_valid <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            if (start) begin
                left  <= n_valid;
                total <= n_valid;
                ones  <= {CNT_BITS{1'b0}};
            end else if (in_valid) begin
                ones <= ones_n;
                left <= left - step;
                if (last) begin
                    // 2*P - N, computed at a width that holds it and then
                    // narrowed on purpose. The earlier version let a 17-bit
                    // subtraction fall into a 16-bit target and justified it
                    // with "our values are small" -- true today, and not a
                    // property of the code. verilator caught it.
                    acc       <= acc_full[ACC_BITS-1:0];
                    out_valid <= 1'b1;
                end
            end
        end
    end

`ifdef KWS_ASSERT
    // $finish, not $fatal: $fatal is a SystemVerilog task and a Verilog-2005
    // simulator may reject it. run_tb.sh decides pass/fail from the log.
    //
    // Every comparison happens at CHK_BITS, wider than every operand, so
    // nothing is implicitly extended. In an assertion an implicit extension is
    // exactly what makes a check quietly weaker than it reads.
    //
    // CHK_BITS = TMP_BITS+1 keeps every replication non-zero: TMP_BITS equals
    // ACC_BITS whenever ACC_BITS >= CNT_BITS+2, and a zero-width replication is
    // not legal Verilog.
    localparam integer CHK_BITS = TMP_BITS + 1;

    wire signed [CHK_BITS-1:0] acc_chk =
        {{(CHK_BITS-ACC_BITS){acc[ACC_BITS-1]}}, acc};
    wire signed [CHK_BITS-1:0] full_chk =
        {{(CHK_BITS-TMP_BITS){acc_full[TMP_BITS-1]}}, acc_full};
    wire signed [CHK_BITS-1:0] tot_chk =
        {{(CHK_BITS-CNT_BITS){1'b0}}, total};
    // acc_full after the part-select, sign-extended back. Comparing against
    // full_chk asks "did narrowing lose information?" without naming any
    // constant -- no shifted literals, so no width to get wrong.
    wire signed [CHK_BITS-1:0] narrowed_chk =
        {{(CHK_BITS-ACC_BITS){acc_full[ACC_BITS-1]}}, acc_full[ACC_BITS-1:0]};

    // 1. the P5-1 width argument: the value stays inside +-n_terms
    always @(posedge clk) if (out_valid) begin
        if (acc_chk > tot_chk || acc_chk < -tot_chk) begin
            $display("ASSERT %m: acc %0d outside +-%0d", acc_chk, tot_chk);
            $finish;
        end
    end
    // 2. ACC_BITS was wide enough. Check 1 alone would only sometimes catch a
    //    truncation, since a wrapped value can land back inside +-n_terms.
    always @(posedge clk) if (in_valid && last) begin
        if (full_chk !== narrowed_chk) begin
            $display("ASSERT %m: ACC_BITS=%0d too narrow for %0d",
                     ACC_BITS, full_chk);
            $finish;
        end
    end
`endif

endmodule

`default_nettype wire
