// 자체 검사 하네스 -- 클립을 ROM 에서 먹이고, 답을 칩 안에서 채점한다.
//
// 이것이 존재하는 이유는 **속도**다. 클립 하나가 2.87 M 사이클이므로
//
//     RTL 시뮬   클립 1 개에 수십 초   ->  600 개면 며칠
//     칩        클립 1 개에 57.4 ms   ->  600 개면 34 초
//
// 하드웨어가 시뮬보다 1000 배 빠르다. 그게 FPGA 를 쓰는 이유이고, "몇 백~몇 천
// 개를 훑는다" 는 목표가 칩을 가리킨다.
//
// ---- 10 ms 타이머를 쓰지 않는 것이 핵심이다 -------------------------------- //
//
// kws_frame_ctrl 의 FRAME_CYCLES = 500,000 은 **실시간 오디오** 때문에 필요한
// 값이다. 저장된 벡터를 먹일 때는 그럴 이유가 없으므로 net_ready 가 허락하는
// 속도로 민다 -- 프레임당 약 0.45 ms, 클립당 57.4 ms.
//
// 그래서 이 하네스는 kws_frame_ctrl 을 **인스턴스화하지 않는다.** 그 모듈은
// 실시간 경로(kws_board_top)에 그대로 남는다. 여기서 검증하는 것은 네트워크가
// 실리콘에서 맞게 도는가이고, 아날로그 경계는 이 질문에 안 들어온다.
//
// ---- 정답지는 다른 도구가 만들었다 ----------------------------------------- //
//
// exp_rom 은 export/golden.py 가 낸 predictions_fixed.txt 다 -- 파이썬의 정수
// 경로가 낸 답. 칩은 Verilog 다. **두 독립 구현이 같은 수를 내는지**를 보는
// 것이고, 자기가 자기를 채점하는 순환이 아니다 (rtl/RUNNING.md 4).
//
// 기대값은 "정확도 82.5%" 가 아니라 **CLIPS 개 전부 일치**다. 이진 층은
// 2*popcount(XNOR) - N 이고 꼬리는 고정소수점이라 근사가 없다. 하나라도
// 틀리면 버그다.
//
// ---- 카운터 하나로는 부족하다 ---------------------------------------------- //
//
// "597/600" 은 어느 3 개가 틀렸는지 말해주지 않는다. first_fail_* 가 그것을
// 답하고, 그 인덱스 하나만 있으면 그 클립을 **시뮬로 가져가 층별 골든 벡터로**
// 어느 모듈인지 파낼 수 있다.
//
//     칩    600 개를 34 초에 훑는다  ->  "문제가 있나 / 어느 입력인가"
//     시뮬  그 한 입력을 파고든다    ->  "어느 층인가"

`timescale 1ns/1ps
`default_nettype none

module kws_selftest #(
    parameter integer N_CH      = 16,
    parameter integer T_IN      = 128,
    parameter integer WORD_BITS = 32,
    parameter integer CLS_BITS  = 4,
    // 몇 개를 돌릴 것인가. ROM 이 이 수만큼만 선언되므로 BRAM 예산이 여기서
    // 정해진다 -- 32 비트 폭에서 약 680 개가 상한이다(docs/fpga_primitives.md 4).
    parameter integer CLIPS     = 600,
    // ROM 파일. rtl/build_selftest.tcl 이 1000 클립 벡터에서 CLIPS 개를 잘라
    // 만든다. $readmemh 는 주소로만 자를 수 있고 **파일 오프셋으로는 못 자르므로**
    // (인자가 메모리 주소다) 오프셋은 파일 쪽에서 해결한다.
    parameter         CLIP_FILE = "",
    parameter         EXP_FILE  = ""
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 go,          // 1 사이클 펄스: 전체 스윕 시작

    // ---- 네트워크 쪽 (kws_top 의 계약 그대로) ---------------------------- //
    output reg                  net_start,
    // 조합이다 -- 위의 "ROM 읽기는 동기, valid 는 조합" 참조.
    output wire                 net_valid,
    output wire [N_CH-1:0]      net_frame,
    input  wire                 net_ready,
    input  wire                 net_busy,
    input  wire                 net_cls_valid,
    input  wire [CLS_BITS-1:0]  net_cls,

    // ---- 결과 -- 이 여섯 개가 밖으로 나가는 전부다 ----------------------- //
    output reg                  done,
    output reg  [15:0]          total,       // 몇 개 돌았나 (멈춤 감지)
    output reg  [15:0]          match,       // 몇 개 맞았나
    output reg                  any_fail,
    output reg  [15:0]          first_fail_idx,
    output reg  [CLS_BITS-1:0]  first_fail_got,
    output reg  [CLS_BITS-1:0]  first_fail_exp
);

    // NWI 는 tb_top.v 와 같은 계산이다. 16 채널 / 32 비트 워드에서 1 이다.
    localparam integer NWI = (N_CH + WORD_BITS - 1) / WORD_BITS;

    // 폭을 **파일 폭 그대로** 선언한다. export/pack.py 의 to_hex_words 가
    // f"{v:08x}" 로 항상 8 자리를 쓰므로 파일은 32 비트/프레임이고, 상위
    // 16 비트는 구조적으로 0 이다(채울 채널이 없다). 그래도 16 비트로 선언해
    // $readmemh 의 절단에 기대지는 않는다 -- rtl/README.md 3-07 이 "선언된 폭
    // 그대로 읽어야 한다" 로 이미 한 번 대가를 치른 교훈이다.
    reg [WORD_BITS-1:0] clip_rom [0:CLIPS*T_IN*NWI-1];

    // 정답지. **16 진수 파일이어야 한다.** predictions_fixed.txt 는 10 진수이고
    // 클래스 11 이 실제로 나오므로 $readmemh 가 "11" 을 0x11 = 17 로 읽는다.
    // export/predictions_to_hex.py 가 그 변환을 하고 왕복 검사까지 한다.
    reg [CLS_BITS-1:0]  exp_rom  [0:CLIPS-1];

    initial begin
        $readmemh(CLIP_FILE, clip_rom);
        $readmemh(EXP_FILE,  exp_rom);
    end

    localparam [2:0] S_IDLE  = 3'd0,  // go 를 기다린다
                     S_START = 3'd1,  // net_start 1 사이클
                     S_FETCH = 3'd2,  // ROM 이 현재 fi 를 따라잡는 한 사이클
                     S_HOLD  = 3'd3,  // 데이터를 들고 net_ready 를 기다린다
                     S_WAIT  = 3'd4,  // class_valid 를 기다린다
                     S_NEXT  = 3'd5,  // 채점하고 busy 가 내려가길 기다린다
                     S_DONE  = 3'd6;

    reg [2:0]  st;
    reg [15:0] ci;                    // clip index
    reg [15:0] fi;                    // frame index within the clip
    reg [CLS_BITS-1:0] got;
    reg                got_v;

    // ---- ROM 읽기는 동기, valid 는 조합 -- 둘 다 이유가 있다 -------------- //
    //
    // **valid 를 조합으로 두는 이유**: kws_top 의 계약은 `in_valid && in_ready`
    // 가 같은 사이클에 성립하는 것이고, kws_top.v:416 이 `in_valid && !in_ready`
    // 를 어서션으로 잡는다. net_ready 를 보고 **다음** 사이클에 net_valid 를
    // 세우면 그 사이에 ready 가 내려갈 수 있다 -- 처음에 그렇게 썼고 시뮬 330 ns
    // 에서 바로 걸렸다. 같은 사이클에 묶으면 구조적으로 못 어긴다.
    //
    // 조합 루프가 아닌 이유: kws_top 의 in_ready 는 `can_push && (pc < T_IN)` 이고
    // can_push 도 pc 도 전부 레지스터다. in_valid 에 의존하지 않는다.
    //
    // **ROM 읽기를 동기로 두는 이유**: 비동기 읽기(`assign d = rom[a]`)로 쓰면
    // Vivado 가 BRAM 을 못 쓰고 LUT 으로 푼다(rtl/README.md 3-08, 그리고 가중치
    // ROM 35 개가 실제로 그렇게 됐다). 2,400 Kbit 를 LUT 으로 풀면 칩에 안 들어간다.
    //
    // 그래서 프레임당 2 사이클(S_FETCH + S_HOLD)을 쓴다. 클립당 256 사이클이고
    // 네트워크가 클립당 2.87 M 을 쓰므로 0.009% 다.
    localparam integer AW = (CLIPS * T_IN * NWI <= 2) ? 1
                          : $clog2(CLIPS * T_IN * NWI);
    wire [AW-1:0] rom_addr = (ci * T_IN + fi);
    reg [WORD_BITS-1:0] rom_q;

    always @(posedge clk) rom_q <= clip_rom[rom_addr];

    // NWI == 1 이므로 워드 하나가 곧 프레임이다. 채널 c 가 비트 c, LSB first
    // (golden.json 의 input layout). 상위 16 비트는 구조적으로 0 이다.
    assign net_frame = rom_q[N_CH-1:0];
    assign net_valid = (st == S_HOLD) && net_ready;

    // 결과를 잡아두는 것은 class_valid 가 1 사이클이기 때문이다. S_WAIT 에서
    // 놓치면 영원히 못 본다.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            got   <= {CLS_BITS{1'b0}};
            got_v <= 1'b0;
        end else if (st == S_START) begin
            got_v <= 1'b0;            // 클립마다 새로
        end else if (net_cls_valid) begin
            got   <= net_cls;
            got_v <= 1'b1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st             <= S_IDLE;
            ci             <= 16'd0;
            fi             <= 16'd0;
            net_start      <= 1'b0;
            done           <= 1'b0;
            total          <= 16'd0;
            match          <= 16'd0;
            any_fail       <= 1'b0;
            first_fail_idx <= 16'd0;
            first_fail_got <= {CLS_BITS{1'b0}};
            first_fail_exp <= {CLS_BITS{1'b0}};
        end else begin
            net_start <= 1'b0;

            case (st)
            S_IDLE:
                if (go) begin
                    ci    <= 16'd0;
                    total <= 16'd0;
                    match <= 16'd0;
                    done  <= 1'b0;
                    st    <= S_START;
                end

            S_START: begin
                net_start <= 1'b1;
                fi        <= 16'd0;
                st        <= S_FETCH;
            end

            // rom_q 가 현재 fi 를 따라잡는 한 사이클. rom_addr 가 조합이고
            // rom_q 가 레지스터이므로, fi 를 바꾼 다음 사이클에 데이터가 맞는다.
            S_FETCH:
                st <= S_HOLD;

            S_HOLD:
                // net_valid 는 위에서 `(st == S_HOLD) && net_ready` 로 조합
                // 생성된다. 그러니 여기서는 **전송이 일어났는가**만 본다 --
                // 그 조건이 곧 net_valid 다.
                if (net_ready) begin
                    fi <= fi + 16'd1;
                    if (fi + 16'd1 == T_IN[15:0]) st <= S_WAIT;
                    else                          st <= S_FETCH;
                end

            S_WAIT:
                if (got_v)
                    st <= S_NEXT;

            S_NEXT:
                // busy 가 내려간 뒤에 다음 클립을 시작한다. tb_top 의
                // `while (busy)` 와 같다 -- 라인버퍼가 비워질 시간을 준다.
                if (!net_busy) begin
                    total <= total + 16'd1;
                    if (got == exp_rom[ci]) begin
                        match <= match + 16'd1;
                    end else if (!any_fail) begin
                        // **첫 실패만** 잡는다. 그 인덱스 하나로 시뮬에서
                        // 층별로 파낼 수 있고, 두 번째 이후는 같은 원인일
                        // 가능성이 높아 핀/레지스터를 더 쓸 값이 없다.
                        any_fail       <= 1'b1;
                        first_fail_idx <= ci;
                        first_fail_got <= got;
                        first_fail_exp <= exp_rom[ci];
                    end

                    if (ci + 16'd1 == CLIPS[15:0]) begin
                        st   <= S_DONE;
                        done <= 1'b1;
                    end else begin
                        ci <= ci + 16'd1;
                        st <= S_START;
                    end
                end

            S_DONE:
                // 끝나면 머문다. go 를 다시 주면 처음부터 -- 두 번 돌려 같은
                // 수가 나오는지가 **시뮬이 절대 못 잡는** 검사다(메타스테이빌
                // 리티, 타이밍 마진). 시뮬은 0 지연 논리로 돈다.
                if (go) st <= S_IDLE;

            default: st <= S_IDLE;
            endcase
        end
    end

`ifdef KWS_ASSERT
    // 계약 위반을 그 사이클에 잡는다. kws_top 쪽에도 같은 어서션이 있지만
    // 이쪽이 푸셔이므로 여기서 잡히는 것이 원인에 가깝다.
    always @(posedge clk) if (rst_n && net_valid && !net_ready) begin
        $display("ASSERT %m: pushed a frame while net_ready was low");
        $finish;
    end
    // 멈춤은 값을 안 남기므로 어서션이 못 잡는 유일한 실패다(rtl/README.md 3-09).
    // 클립 하나가 2.87 M 사이클이니 그 10 배를 넘으면 멈춘 것이다.
    localparam integer WATCHDOG = 30000000;
    integer stuck;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) stuck <= 0;
        else if (st == S_IDLE || st == S_DONE) stuck <= 0;
        else begin
            stuck <= stuck + 1;
            if (stuck > WATCHDOG) begin
                $display("ASSERT %m: stuck in state %0d on clip %0d frame %0d",
                         st, ci, fi);
                $finish;
            end
        end
    end
`endif

endmodule

`default_nettype wire
