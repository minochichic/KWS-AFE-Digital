// 브링업 1 — VCCO 를 재기 위해 EXT0~EXT15 를 High 로 고정한다.
//
// 이 파일에 로직이 없는 것이 의도다. **클럭도 리셋도 쓰지 않는다.**
// 그래서 이게 실패하면 원인이 딱 하나로 좁혀진다 -- 툴체인이나 JTAG 이지
// 클럭도, 타이밍도, 우리 설계도 아니다. 클럭이 필요한 확인은 walk_probe.v 가 한다.
//
// 이 비트스트림이 답하는 것:
//   1. Vivado 합성-배치배선-비트스트림이 끝까지 도는가
//   2. JTAG 로 보드에 실제로 올라가는가
//   3. **Exp. Port 뱅크의 VCCO 가 몇 V 인가**  <- 이것 때문에 만든다
//   4. 16 가닥이 전부 살아 있는가 (하나라도 0 V 면 그 핀이 죽은 것)
//
// 재는 법: J6 핀 49 나 50 을 접지 기준으로 잡고 핀 3(ch00)에 프로브를 댄다.
// 출력이 High 이므로 그 전압이 곧 VCCO 다. 부하가 DMM(10 MΩ)뿐이라 드라이브
// 강하가 없고, 킷의 직렬 33 Ω 도 DC 에서는 아무 영향이 없다.
//
// ⚠️ AFE 를 물린 채로 올리지 말 것. 이 설계는 그 16핀을 **출력으로 구동**하고
//    비교기도 푸시풀이라 출력끼리 싸운다. 킷의 33 Ω 만으로는 100 mA 가까이 흐르고
//    LPV7215 는 580 nA 마이크로파워 부품이라 못 견딘다. docs/hanback_kit.md 4.4.

`timescale 1ns/1ps
`default_nettype none

module vcco_probe #(
    parameter integer N_CH = 16,
    // 전부 1. 하나라도 0 V 로 읽히면 배선이나 핀 배정이 틀린 것이고, 16 개가
    // 전부 같은 값이면 그 값이 VCCO 다. 패턴을 섞으면 "왜 이 핀만 0 이지" 를
    // 먼저 의심하게 되는데, 지금은 그런 질문을 만들 때가 아니다.
    parameter [15:0] PATTERN = 16'hFFFF
) (
    output wire [N_CH-1:0] ext
);

    assign ext = PATTERN[N_CH-1:0];

endmodule

`default_nettype wire
