// kws_selftest_top: 하네스 자체가 맞는지 시뮬에서 먼저 본다.
//
// Clips  : `KWS_ST_CLIP_FILE   (export/slice_selftest.py 가 자른 것)
// Expect : `KWS_ST_EXP_FILE
//
// ---- 이 테스트벤치가 몇 개를 돌리는가 -------------------------------------- //
//
// **적게.** 클립 하나가 2.87 M 사이클이므로 600 개는 17 억 사이클이고 시뮬로는
// 며칠이다. 그게 애초에 칩을 쓰는 이유다(클립당 57.4 ms).
//
// 여기서 확인하는 것은 정확도가 아니라 **하네스가 프레임을 맞게 먹이는가**다.
// 그건 두세 클립에서 전부 드러난다 -- ready/valid 계약, start 펄스, 클래스
// 래치, 카운터, exp_rom 의 16 진수 해석.
//
// **이 단계를 건너뛰면 하네스 버그가 설계 버그로 보인다.** 하네스가 프레임을
// 하나 밀리게 먹이면 칩에서 "600 중 3 개 불일치" 로 나타나고, 그때 의심하는
// 것은 RTL 이지 하네스가 아니다. 가장 시간을 많이 먹는 오진이다.
//
//     ./slice_selftest --clips 2   -> 이 테스트벤치        (몇 분)
//     ./slice_selftest --clips 600 -> 합성 -> 칩           (34 초)
//
// 같은 하네스, 같은 벡터 파일, 클립 수만 다르다.

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module tb_selftest;

    localparam integer CLIPS = `KWS_ST_CLIPS;
    localparam integer T_IN  = `KWS_ST_T_IN;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg go = 1'b0;

    wire        done;
    wire [15:0] total;
    wire [15:0] match;
    wire        any_fail;
    wire [15:0] first_fail_idx;
    wire [3:0]  first_fail_got;
    wire [3:0]  first_fail_exp;

    always #10 clk = ~clk;            // 50 MHz, 보드와 같은 주기

    kws_selftest_top #(
        .CLIPS     (CLIPS),
        .CLIP_FILE (`KWS_ST_CLIP_FILE),
        .EXP_FILE  (`KWS_ST_EXP_FILE)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .go             (go),
        .done           (done),
        .total          (total),
        .match          (match),
        .any_fail       (any_fail),
        .first_fail_idx (first_fail_idx),
        .first_fail_got (first_fail_got),
        .first_fail_exp (first_fail_exp)
    );

    // 진행 상황을 클립마다 찍는다. 한 클립이 2.87 M 사이클이라 아무것도 안
    // 찍으면 "멈췄나 도는 중인가" 를 구별할 수 없다.
    reg [15:0] seen = 16'd0;
    always @(posedge clk) begin
        if (rst_n && total !== seen) begin
            seen <= total;
            $display("  clip %0d done at %0t: match=%0d", total - 1, $time,
                     match);
        end
    end

    // 워치독. 클립당 2.87 M 사이클 + 여유 4 배.
    localparam integer LIMIT = CLIPS * 2870000 * 4;
    integer cyc = 0;
    always @(posedge clk) begin
        cyc = cyc + 1;
        if (cyc > LIMIT) begin
            $display("FAIL watchdog at %0t: total=%0d of %0d, state stuck",
                     $time, total, CLIPS);
            $finish;
        end
    end

    initial begin
        // 파형 덤프는 **기본으로 끈다.** 클립 하나가 2.87 M 사이클이고 이 설계는
        // 신호가 2 만 개가 넘으므로, $dumpvars(0, ...) 를 켜면 시뮬이 기어가고
        // 파일이 GB 로 간다. 이 테스트벤치가 답하는 질문은 pass/fail 이다.
        //
        // 불일치를 파야 할 때만 켠다:  xvlog --define KWS_DUMP
`ifdef KWS_DUMP
        $dumpfile("tb_selftest.vcd");
        $dumpvars(0, tb_selftest);
`endif

        $display("== tb_selftest: %0d clips, T_IN %0d ==", CLIPS, T_IN);
        $display("   clips    %s", `KWS_ST_CLIP_FILE);
        $display("   expected %s", `KWS_ST_EXP_FILE);

        repeat (5) @(negedge clk);
        rst_n = 1'b1;
        repeat (5) @(negedge clk);

        @(negedge clk) go = 1'b1;
        @(negedge clk) go = 1'b0;

        wait (done === 1'b1);
        repeat (2) @(negedge clk);

        $display("");
        $display("total %0d / %0d", total, CLIPS);
        $display("match %0d / %0d", match, CLIPS);

        // 기대값은 **전부 일치**다. 정확도가 아니라 재현이다 -- 정답지는
        // export/golden.py 의 정수 경로가 낸 것이고, 우리도 같은 정수 연산을
        // 한다. 근사가 없으므로 하나라도 틀리면 버그다.
        if (total !== CLIPS[15:0]) begin
            $display("FAIL ran %0d clips, expected %0d", total, CLIPS);
            $finish;
        end
        if (any_fail === 1'b1) begin
            $display("FAIL first mismatch at clip %0d: got %0d, want %0d",
                     first_fail_idx, first_fail_got, first_fail_exp);
            $display("     re-run export.golden on that clip WITHOUT");
            $display("     --vectors-only to get the per-layer vectors, then");
            $display("     ./rtl/run_tb.sh top -- the first layer that differs");
            $display("     names the module.");
            $finish;
        end
        if (match !== CLIPS[15:0]) begin
            $display("FAIL match %0d != total %0d with any_fail low", match,
                     total);
            $finish;
        end

        $display("");
        $display("PASS all %0d clips reproduce predictions_fixed.txt", CLIPS);
        // run_xsim.ps1 은 로그에서 ", 0 failures" 를 찾는다 (tb_top 의
        // "%0d frames checked, %0d failures" 와 같은 관례). 그 줄이 없으면
        // 통과해도 스크립트가 실패로 본다.
        $display("%0d clips checked, %0d failures", total, total - match);
        $finish;
    end

endmodule

`default_nettype wire
