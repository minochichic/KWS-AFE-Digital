/*
 * board.h -- STM32 웨이크 보드부 (firmware/stm32_wake/README.md "board.c 가 할 일")
 *
 * 판단은 core/wake_core 가, 바이트는 core/link 가 한다. 여기는 레지스터만 만진다:
 *   LPTIM1 (LSE, ARR 327) 10 ms 박자 → EXTI pending | 이전 GPIOB->IDR = 칸 → wake_tick
 *   WAKE_CLIP_READY → PA8(WAKE_FPGA) High → 패킷 → 결과 8 B (200 ms) → PA8 Low → wake_sent
 *
 * -DWAKE_REPLAY 로 빌드하면 AFE 대신 gen/replay_clips.h 의 골든 클립을 보낸다 (단계 3).
 */
#ifndef BOARD_H
#define BOARD_H

#include <stdint.h>

#define BOARD_LPTIM_ARR      327u   /* 328 틱 / 32768 Hz = 10.0098 ms */
#define BOARD_RESULT_MS      200u   /* FPGA 결과 대기 (전송 17.6 + 추론 57.4 ms 실측) */
#define BOARD_TX_MS          100u

/* 보드에 LED 가 없다 -- 디버거나 STM32_Programmer_CLI 로 이 구조체를 읽는다. */
typedef struct {
  uint32_t ticks;        /* 10 ms 박자 수 */
  uint32_t clips;        /* 보낸 클립 수 */
  uint32_t ok;           /* CRC 맞는 결과를 받은 수 */
  uint32_t timeouts;     /* 결과 없음 */
  uint32_t bad;          /* 결과는 왔는데 sync/CRC 오류 */
  uint32_t match;        /* 재생 모드: class_idx == replay_expect */
  uint16_t last_frame;   /* 마지막 칸 (bit k = ch k) */
  uint8_t  last_class;
  uint8_t  last_keyword;
  int32_t  last_margin;
} board_stats_t;

extern volatile board_stats_t board_stats;

void board_init(void);
void board_poll(void);

#endif /* BOARD_H */
