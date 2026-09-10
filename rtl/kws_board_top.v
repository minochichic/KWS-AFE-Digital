// 보드 최상위 -- 아날로그 경계부터 LED 까지 한 덩어리로 묶는다.
//
// 여기까지가 없어서 다음 셋이 전부 막혀 있었다:
//
//   1. cmp 16 핀이 한 번도 제약된 적이 없다. kws_top_synth 에는 그 포트가 없어서
//      합성 로그가 늘 `applied 9 pin constraints` 였다 (clk/rst_n/start/class_*/busy).
//   2. -impl 이 의미가 없었다. 핀이 다 제약돼야 write_bitstream 앞 DRC 가 검사한다.
//   3. **kws_frame_ctrl 과 kws_top 이 한 번도 함께 돈 적이 없다.** 각자 골든
//      벡터로 검증됐지만 사이가 안 이어져 있었다 -- grep 해 보면 frame_ctrl 을
//      인스턴스화하는 곳이 자기 테스트벤치뿐이다.
//
// 3 번이 가장 크다. docs/hanback_kit.md 3.3 의 "22 배 여유" 는 두 숫자를 나눠서
// 얻은 값이지 실행으로 확인한 것이 아니다. kws_frame_ctrl 의 실시간 어서션
// (`frame_edge && pending`)은 이 래퍼가 있어야 비로소 밟힌다.
//
// ---- 계층 ----------------------------------------------------------------- //
//
//   kws_board_top          <- 보드 핀, 리셋/start 정리, busy 합성, 결과 래치
//     +- kws_frame_ctrl    <- 아날로그 경계 (docs/ICD.md 5)
//     +- kws_top_synth     <- 매크로 바인딩 (기계 생성, 그대로 둔다)
//          +- kws_top
//
// **kws_top_synth 를 버리고 kws_top 을 직접 인스턴스화하지 않는 이유**: 저 파일은
// rtl/tb/tb_top.v 에서 기계적으로 추출된 것이고 tests/test_synth_top.py 가 두
// 파일이 같은 매크로 집합을 쓰는지 검사한다. 파라미터 100 줄을 여기 복사하면
// 그 검사를 우회하면서 조용히 어긋날 수 있다 -- 그리고 어긋난 채로도 합성은
// 성공한다. 저 파일의 머리말이 경고하는 사고가 정확히 그것이다.
//
// ---- 이 모듈은 연산을 하나도 하지 않는다 ----------------------------------- //
//
// 하는 일은 **보드의 지저분함을 흡수하는 것**뿐이다. 계산은 전부 아래에 있다.
// 보드가 또 바뀌면 여기만 고친다.

`timescale 1ns/1ps
`default_nettype none

`include "rtl/gen/active.vh"

module kws_board_top #(
    // ---- 보드 사실 -- RTL 에 들어오는 것은 이것뿐이다 --------------------- //
    // docs/ICD.md 6: 아날로그가 바뀌어도 RTL 에 닿는 것은 N_CH / CMP_INVERT /
    // FRAME_CYCLES 셋뿐이고, 나머지는 전부 가중치로 흡수된다. 그 셋이 여기 모인다
    // (N_CH 는 매크로에서 온다).
    //
    // 킷의 클럭 선택 스위치 F = 50 MHz (docs/hanback_kit.md 3.2).
    parameter integer CLK_HZ     = 50000000,
    // 비교기의 극성. 검출기가 반전이면 이것만 뒤집고 나머지는 그대로다.
    parameter integer CMP_INVERT = 0
) (
    input  wire                 clk,
    input  wire                 rst_n,        // 핀. 비동기. 내부 풀업
    input  wire                 start,        // 핀. 비동기. 내부 풀다운
    input  wire [`KWS_N_CH-1:0] cmp,          // ★ 아날로그 경계. 비동기
    output wire                 busy,
    output wire                 class_valid,
    output wire [3:0]           class_idx
);

    // 10 ms 를 클럭으로 바꾼다. 매니페스트는 ms 를 들고 있고(`KWS_FRAME_MS`),
    // 그것을 사이클로 바꾸려면 클럭 주파수가 필요한데 그건 보드 사실이다.
    // 50 MHz 에서 500,000. 나누기를 먼저 해서 곱셈 오버플로를 피한다.
    localparam integer FRAME_CYCLES = (CLK_HZ / 1000) * `KWS_FRAME_MS;

    // ======================================================================= //
    // 1. 리셋 -- 비동기 어서트, 동기 릴리스
    // ======================================================================= //
    //
    // rst_n 은 핀에서 온다. 언제 놓이는지 클럭과 아무 관계가 없다. 그대로 쓰면
    // 릴리스가 클럭 엣지 근처에 걸렸을 때 **어떤 FF 은 이번 사이클에, 어떤 FF 은
    // 다음 사이클에 풀린다.** 상태기계가 절반만 리셋된 채로 출발하고, 증상은
    // 간헐적이라 재현이 안 된다.
    //
    // 누를 때는 즉시(비동기), 놓을 때는 클럭에 맞춰서(동기) -- 표준 해법이다.
    //
    // kws_top.xdc 의 `set_false_path -from [get_ports rst_n]` 은 도구에게
    // "이 경로는 검사하지 마" 라고 말하는 것이지 이 문제를 푸는 것이 아니다.
    // 푸는 것은 여기다.
    (* ASYNC_REG = "TRUE" *) reg [1:0] rst_sync;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rst_sync <= 2'b00;
        else        rst_sync <= {rst_sync[0], 1'b1};
    end

    wire rst_n_s = rst_sync[1];

    // ======================================================================= //
    // 2. start -- 동기화 + 상승 엣지
    // ======================================================================= //
    //
    // start 는 DIP 스위치나 버튼이다. kws_frame_ctrl 은 이것을 **레벨**로 본다
    // (`if (start)` 가 매 사이클 상태를 S_PADL 로 되돌린다). 스위치를 올려둔
    // 동안 계속 1 이면 클립이 매 사이클 재시작되어 첫 프레임에서 못 벗어난다.
    //
    // 그래서 2FF 로 동기화하고 **올라가는 순간에만 1 사이클 펄스**를 만든다.
    // 채터링 디바운스는 안 넣었다 -- 필요해지면 여기에 카운터 하나를 더한다.
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

    wire start_pulse = start_sync[1] && !start_q;

    // ======================================================================= //
    // 3. 아날로그 경계 -> 네트워크
    // ======================================================================= //
    //
    // 핸드셰이크는 이름까지 맞는다. 다만 **한 사이클 차이가 있다**:
    // frame_ctrl 은 out_ready 를 본 사이클에 `take` 하고 out_valid 는 그
    // 다음 사이클에 뜬다. 그 사이에 kws_top 의 can_push 가 내려가면 프레임이
    // 조용히 사라진다 -- rtl/README.md 3-09 가 두 번 대가를 치른 종류다.
    //
    // 지금은 frame_ctrl 이 유일한 푸셔라 그 사이에 상태가 바뀔 이유가 없어서
    // 성립할 것으로 본다. **그러나 그것을 여기서 단정하지 않는다** --
    // kws_top.v:416 의 `in_valid && !in_ready` 어서션과 kws_frame_ctrl 의
    // `frame_edge && pending` 어서션이 tb_board_top 에서 그것을 판정한다.
    wire                 fc_valid;
    wire [`KWS_N_CH-1:0] fc_frame;
    wire                 fc_busy;
    wire                 top_ready;
    wire                 top_busy;
    wire                 cls_valid;
    wire [3:0]           cls_idx;

    kws_frame_ctrl #(
        .N_CH        (`KWS_N_CH),
        .FRAME_CYCLES(FRAME_CYCLES),
        .NATIVE_T    (`KWS_NATIVE_T),
        .T           (`KWS_T),
        .PAD_LEFT    (`KWS_PAD_LEFT),
        .CMP_INVERT  (CMP_INVERT)
    ) u_fc (
        .clk      (clk),
        .rst_n    (rst_n_s),
        .start    (start_pulse),
        .cmp      (cmp),
        .out_ready(top_ready),
        .out_valid(fc_valid),
        .out_frame(fc_frame),
        .busy     (fc_busy)
    );

    kws_top_synth u_net (
        .clk        (clk),
        .rst_n      (rst_n_s),
        .start      (start_pulse),
        .in_valid   (fc_valid),
        .in_frame   (fc_frame),
        .in_ready   (top_ready),
        .busy       (top_busy),
        .class_valid(cls_valid),
        .class_idx  (cls_idx)
    );

    // ======================================================================= //
    // 4. 밖으로 -- busy 합성, 결과 래치
    // ======================================================================= //
    //
    // busy 가 둘인 것은 안이 둘이기 때문이다. 밖에서 보는 것은 하나여야 하니
    // OR 로 낸다 -- "수집 중이거나 추론 중".
    assign busy = fc_busy || top_busy;

    // class_idx 를 래치하는 이유는 관측 때문이다. 추론이 10 Hz 라 100 ms 주기인데
    // class_valid 는 **1 사이클(50 MHz 에서 20 ns)** 이다. 500 만 분의 1 의
    // 시간만 유효하니 LED 도 사람 눈도 그것을 볼 수 없다.
    //
    // 잡아두면 다음 결과가 나올 때까지 100 ms 동안 표시된다. LED 든 로직
    // 애널라이저든 오실로스코프든, 관측자가 무엇이든 값이 붙들려 있어야 읽힌다.
    //
    // 킷의 LED/FND 핀 구성표가 아직 없어서(docs/hanback_kit.md 7) 지금 이 값은
    // 확장 포트 핀 22~25 로 나가고 동료 기판은 그 핀을 연결하지 않았다. 핀표가
    // 오면 배선만 옮기면 되고 이 래치는 그대로다.
    reg [3:0] class_idx_r;
    reg       class_valid_r;

    always @(posedge clk or negedge rst_n_s) begin
        if (!rst_n_s) begin
            class_idx_r   <= 4'd0;
            class_valid_r <= 1'b0;
        end else begin
            class_valid_r <= cls_valid;
            if (cls_valid) class_idx_r <= cls_idx;
        end
    end

    assign class_idx = class_idx_r;

    // class_valid is delayed with class_idx so the output pair has normal
    // synchronous-valid timing. An LED pulse stretcher can be added after the
    // board's LED/FND pin map is known.
    // Delay valid by the same cycle used to latch class_idx. A consumer that
    // samples class_idx when class_valid is high therefore sees the new class,
    // while class_idx remains stable between classifications for an LED/FND.
    assign class_valid = class_valid_r;

`ifdef KWS_ASSERT
    // 이 래퍼가 존재하는 이유 자체를 검사한다. FRAME_CYCLES 는 보드 사실에서
    // 유도되므로 CLK_HZ 를 잘못 주면 조용히 다른 회로가 된다 -- 10 ms 가 아닌
    // 프레임은 학습된 가중치와 다른 시간 척도를 뜻하고, 정확도로만 드러난다.
    initial begin
        if (FRAME_CYCLES <= 0) begin
            $display("ASSERT %m: FRAME_CYCLES %0d <= 0 (CLK_HZ %0d)",
                     FRAME_CYCLES, CLK_HZ);
            $finish;
        end
        $display("kws_board_top: CLK_HZ %0d, FRAME_MS %0d, FRAME_CYCLES %0d",
                 CLK_HZ, `KWS_FRAME_MS, FRAME_CYCLES);
    end
`endif

endmodule

`default_nettype wire
