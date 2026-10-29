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

/*
 * 브링업 테스트 (AFE 없이). 1 이면:
 *   - PB0~PB15 내부 풀다운 -- 떠 있는 입력이 1 로 읽히지 않게. 손으로 1.8 V 를 대서 펄스를 낸다
 *   - DBGMCU Stop 디버그 유지 -- Stop 2 중에도 SWD 로 board_stats 를 읽을 수 있다 (전력은 는다)
 *   - 채널별 에지 수 board_stats.edges[]
 * 전력을 잴 때와 AFE 를 붙인 뒤에는 0.
 */
#ifndef BOARD_TEST
#define BOARD_TEST           1
#endif
#define BOARD_TEST_AWAKE_MS  3u     /* 테스트 빌드: 깨어날 때마다 이만큼 깨어 있는다 (SWD 읽기용) */

/* 보드에 LED 가 없다 -- 디버거나 STM32_Programmer_CLI 로 이 구조체를 읽는다. */
typedef struct {
  uint32_t ticks;        /* 10 ms 박자 수 */
  uint32_t stops;        /* Stop 2 에 들어간 횟수 */
  uint32_t clips;        /* 보낸 클립 수 */
  uint32_t ok;           /* CRC 맞는 결과를 받은 수 */
  uint32_t timeouts;     /* 결과 없음 */
  uint32_t bad;          /* 결과는 왔는데 sync/CRC 오류 */
  uint32_t match;        /* 재생 모드: class_idx == replay_expect */
  uint16_t last_frame;   /* 마지막 칸 (bit k = ch k) */
  uint8_t  last_class;
  uint8_t  last_keyword;
  int32_t  last_margin;
  uint16_t edges_or;     /* 지금까지 에지가 한 번이라도 온 채널 (EXTI pending 누적 OR) */
  uint8_t  max_on;       /* 칸 하나에서 켜진 채널 수의 최댓값 */
  uint8_t  state;        /* wake_core 상태 사본: 0 WARMUP, 1 IDLE, 2 CAPTURE, 3 HOLD */
  int32_t  bg_q;         /* wake_core 배경 사본 (= 배경 채널 수 x 32) */
  uint32_t injected;     /* board_inject 를 칸에 넣은 횟수 */
  uint32_t replay_first_bad;  /* 재생: 처음 틀린 클립 번호, 없으면 0xFFFFFFFF */
  uint32_t replay_bad_info;   /* 재생: 그 클립의 got | exp << 8 | rc << 16 (rc 1 = 타임아웃) */
  uint32_t replay_state;      /* 재생: 0 대기, 1 진행 중, 2 끝 */
  uint32_t wake_test_n;       /* WAKE 테스트: 시행 수 (각 시행 = WAKE 있음 1 + 없음 1) */
  uint32_t wake_with_ok;      /* 반쪽 패킷 뒤 WAKE 를 올리고 보낸 클립이 맞은 수 (기대: 전부) */
  uint32_t wake_without_ok;   /* 반쪽 패킷 뒤 WAKE 없이 보낸 클립이 맞은 수 (기대: 0 -- 대조군) */
  uint32_t uart_err;          /* 마지막 실패 수신의 huart2.ErrorCode | HAL 반환값 << 16 (0x08 = 오버런) */
  uint32_t rx_fail_ms;        /* 마지막 실패 수신이 전송 끝에서 몇 ms 만에 끝났나 (200 근처 = 진짜 타임아웃) */
#if BOARD_TEST
  uint16_t edges[16];    /* 채널별 에지가 잡힌 칸 수 */
#endif
  uint32_t magic;        /* BOARD_STATS_MAGIC. SWD 읽기가 중간에 Stop 2 에 걸려 0 으로 잘렸는지 판별 */
} board_stats_t;

#define BOARD_STATS_MAGIC    0xB0A2D57Au

extern volatile board_stats_t board_stats;

/* 테스트 빌드: SWD 로 0 아닌 마스크를 쓰면 다음 칸 하나에 OR 된다 (tools/read_stats.ps1 -Inject) */
extern volatile uint32_t board_inject;

/* 재생 빌드: SWD 로 0 아닌 값을 쓰면 골든 클립을 처음부터 끝까지 한 번 보낸다
 * (tools/read_stats.ps1 -Replay). 부팅하자마자 보내지 않는 이유: FPGA 가 아직 안 올라왔으면
 * 600 개가 전부 타임아웃으로 끝난다. */
extern volatile uint32_t board_replay_go;   /* 1 = 골든 재생, 2 = WAKE 테스트 */
#define BOARD_REPLAY_GAP_MS  5u     /* 클립 사이 간격 -- FPGA 가 다음 패킷 받을 준비 */
#define BOARD_WAKE_TEST_N    20u    /* WAKE 테스트 시행 수 */

/* 재생: 클립 i 에 FPGA 가 낸 클래스. 0xFF = 타임아웃, 0xFE = sync/CRC 오류 */
#define BOARD_GOT_TIMEOUT    0xFFu
#define BOARD_GOT_BAD        0xFEu

void board_init(void);
void board_poll(void);

#endif /* BOARD_H */
