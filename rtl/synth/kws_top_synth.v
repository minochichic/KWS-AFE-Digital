// 합성용 최상위 래퍼 -- 손으로 쓰지 않았다.
//
// kws_top 의 ROM 경로 파라미터는 기본값이 "" 이고, RTL 은 `if (ROM_FILE != "")`
// 로 걸러 아무것도 안 싣는다. 그래서 `synth_design -top kws_top` 을 그냥 돌리면
// **가중치가 하나도 없는 회로가 에러 없이 합성된다.** 테스트벤치는 매크로로 경로를
// 꽂아주지만 테스트벤치는 합성 대상이 아니므로, 같은 일을 하는 얇은 래퍼가 필요하다.
//
// 아래 파라미터 블록은 rtl/tb/tb_top.v 에서 **기계적으로 추출**했다. 베껴 쓰면
// 언젠가 어긋나고, 어긋난 채로도 합성은 성공한다 -- 조용히 다른 회로가 된다.
// tests/test_synth_top.py 가 두 파일이 같은 매크로 집합을 쓰는지 검사한다.
//
// 갱신하려면: python -m rtl.synth.regen  (또는 tests/test_synth_top.py 실패 메시지 참조)

`include "rtl/gen/active.vh"

module kws_top_synth (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire        in_valid,
    input  wire [`KWS_N_CH-1:0] in_frame,
    output wire        in_ready,
    output wire        busy,
    output wire        class_valid,
    output wire [3:0]  class_idx
);

    // tb_top 의 localparam 과 같은 정의. 여기서도 필요하다.
    localparam integer T_IN  = `KWS_T;
    localparam integer T_OUT = T_IN / `KWS_L0_CONV1_STRIDE;

    kws_top #(.WORD_BITS(`KWS_WORD_BITS), .T_IN(T_IN), .T_OUT(T_OUT),
              .N_CH(`KWS_N_CH), .C1_OUT(`KWS_L0_CONV1_OUT_CH),
              .C1_K(`KWS_L0_CONV1_KERNEL), .C1_PAD(`KWS_L0_CONV1_PADDING),
              .C1_STRIDE(`KWS_L0_CONV1_STRIDE),
              .C1_ACC(`KWS_L0_CONV1_ACC_BITS),
              .C1_W(`KWS_ROM_CONV1_W),
              .C1_T(`KWS_ROM_CONV1_T),

              .B1_MID(`KWS_L3_B1_S0_PW_OUT_CH), .B1_OUT(`KWS_L5_B1_S1_PW_OUT_CH),
              .B1_K(`KWS_L2_B1_S0_DW_KERNEL), .B1_PAD(`KWS_L2_B1_S0_DW_PADDING),
              .B1_S0DW_A(`KWS_L2_B1_S0_DW_ACC_BITS),
              .B1_S0PW_A(`KWS_L3_B1_S0_PW_ACC_BITS),
              .B1_S1DW_A(`KWS_L4_B1_S1_DW_ACC_BITS),
              .B1_S1PW_A(`KWS_L5_B1_S1_PW_ACC_BITS),
              .B1_SKIP_A(`KWS_L1_B1_SKIP_ACC_BITS),
              .B1_ADD_A(`KWS_L6_B1_ADD_ACC_BITS),
              .B1_S0DW_W(`KWS_ROM_B1_S0_DW_W),
              .B1_S0DW_T(`KWS_ROM_B1_S0_DW_T),
              .B1_S0PW_W(`KWS_ROM_B1_S0_PW_W),
              .B1_S0PW_T(`KWS_ROM_B1_S0_PW_T),
              .B1_S1DW_W(`KWS_ROM_B1_S1_DW_W),
              .B1_S1DW_T(`KWS_ROM_B1_S1_DW_T),
              .B1_S1PW_W(`KWS_ROM_B1_S1_PW_W),
              .B1_SKIP_W(`KWS_ROM_B1_SKIP_W),
              .B1_ADD_T(`KWS_ROM_B1_ADD_T),

              .B2_K(`KWS_L7_B2_S0_DW_KERNEL), .B2_PAD(`KWS_L7_B2_S0_DW_PADDING),
              .B2_S0DW_A(`KWS_L7_B2_S0_DW_ACC_BITS),
              .B2_S0PW_A(`KWS_L8_B2_S0_PW_ACC_BITS),
              .B2_S1DW_A(`KWS_L9_B2_S1_DW_ACC_BITS),
              .B2_S1PW_A(`KWS_L10_B2_S1_PW_ACC_BITS),
              .B2_SKIP_A(`KWS_L11_B2_ADD_ACC_BITS),
              .B2_ADD_A(`KWS_L11_B2_ADD_ACC_BITS),
              .B2_S0DW_W(`KWS_ROM_B2_S0_DW_W),
              .B2_S0DW_T(`KWS_ROM_B2_S0_DW_T),
              .B2_S0PW_W(`KWS_ROM_B2_S0_PW_W),
              .B2_S0PW_T(`KWS_ROM_B2_S0_PW_T),
              .B2_S1DW_W(`KWS_ROM_B2_S1_DW_W),
              .B2_S1DW_T(`KWS_ROM_B2_S1_DW_T),
              .B2_S1PW_W(`KWS_ROM_B2_S1_PW_W),
              .B2_ADD_T(`KWS_ROM_B2_ADD_T),

              .B3_K(`KWS_L12_B3_S0_DW_KERNEL),
              .B3_PAD(`KWS_L12_B3_S0_DW_PADDING),
              .B3_S0DW_A(`KWS_L12_B3_S0_DW_ACC_BITS),
              .B3_S0PW_A(`KWS_L13_B3_S0_PW_ACC_BITS),
              .B3_S1DW_A(`KWS_L14_B3_S1_DW_ACC_BITS),
              .B3_S1PW_A(`KWS_L15_B3_S1_PW_ACC_BITS),
              .B3_SKIP_A(`KWS_L16_B3_ADD_ACC_BITS),
              .B3_ADD_A(`KWS_L16_B3_ADD_ACC_BITS),
              .B3_S0DW_W(`KWS_ROM_B3_S0_DW_W),
              .B3_S0DW_T(`KWS_ROM_B3_S0_DW_T),
              .B3_S0PW_W(`KWS_ROM_B3_S0_PW_W),
              .B3_S0PW_T(`KWS_ROM_B3_S0_PW_T),
              .B3_S1DW_W(`KWS_ROM_B3_S1_DW_W),
              .B3_S1DW_T(`KWS_ROM_B3_S1_DW_T),
              .B3_S1PW_W(`KWS_ROM_B3_S1_PW_W),
              .B3_ADD_T(`KWS_ROM_B3_ADD_T),

              .C2_K(`KWS_L17_CONV2_DW_KERNEL),
              .C2_PAD(`KWS_L17_CONV2_DW_PADDING),
              .C2_DIL(`KWS_L17_CONV2_DW_DILATION),
              .C2_ACC(`KWS_L17_CONV2_DW_ACC_BITS),
              .C2_W(`KWS_ROM_CONV2_DW_W),
              .C2_T(`KWS_ROM_CONV2_DW_T),

              .TL_C2_OUT(`KWS_CONV2_PW_N_OUT), .TL_C2_ACC(`KWS_CONV2_PW_ACC_BITS),
              .TL_C2_W(`KWS_ROM_CONV2_PW_W),
              .TL_A2_G(`KWS_CONV2_PW_GAIN_BITS),
              .TL_A2_B(`KWS_CONV2_PW_BIAS_BITS),
              .TL_A2_S(`KWS_CONV2_PW_SHIFT), .TL_A2_O(`KWS_CONV2_PW_OUT_BITS),
              .TL_A2_F(`KWS_ROM_CONV2_PW_BN),
              .TL_C3_OUT(`KWS_CONV3_N_OUT), .TL_C3_W(`KWS_CONV3_W_BITS),
              .TL_C3_ACC(`KWS_CONV3_ACC_BITS),
              .TL_C3_WF(`KWS_ROM_CONV3_W),
              .TL_A3_G(`KWS_CONV3_GAIN_BITS), .TL_A3_B(`KWS_CONV3_BIAS_BITS),
              .TL_A3_S(`KWS_CONV3_SHIFT), .TL_A3_O(`KWS_CONV3_OUT_BITS),
              .TL_A3_F(`KWS_ROM_CONV3_BN),
              .TL_C4_OUT(`KWS_CONV4_N_OUT), .TL_C4_W(`KWS_CONV4_W_BITS),
              .TL_C4_ACC(`KWS_CONV4_ACC_BITS),
              .TL_C4_WF(`KWS_ROM_CONV4_W),
              .TL_A4_G(`KWS_CONV4_GAIN_BITS), .TL_A4_B(`KWS_CONV4_BIAS_BITS),
              .TL_A4_S(`KWS_CONV4_SHIFT), .TL_A4_O(`KWS_CONV4_OUT_BITS),
              .TL_A4_F(`KWS_ROM_CONV4_BN),
              .TL_POOL(`KWS_CONV4_POOL_BITS), .TL_C4O_B(4)
    ) u_top (
        .clk(clk), .rst_n(rst_n),
        .start(start), .in_valid(in_valid), .in_frame(in_frame),
        .in_ready(in_ready), .busy(busy),
        .class_valid(class_valid), .class_idx(class_idx)
    );

endmodule
