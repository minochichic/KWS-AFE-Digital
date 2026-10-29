/*
 * board.c -- STM32 웨이크 보드부 (설명은 board.h, 설계는 docs/wake_stm32_plan.md)
 *
 * EXTI 는 NVIC 를 끈 채 pending 비트만 에지 래치로 쓴다 (CubeMX NVIC 탭의 EXTI 는 끔).
 * 칸 = 이전 박자의 GPIOB->IDR | 이번 10 ms 의 EXTI pending -- "칸 시작 때 켜져 있었거나
 * 칸 동안 켜지는 순간이 있었으면 1", FPGA kws_frame_ctrl 의 sticky OR 과 같은 규칙.
 */
#include "board.h"
#include "main.h"

#include <string.h>

#include "../../../core/link.h"
#include "../../../core/wake_core.h"

#ifdef WAKE_REPLAY
#include "../../../gen/replay_clips.h"
#endif

extern LPTIM_HandleTypeDef hlptim1;
extern UART_HandleTypeDef huart2;

volatile board_stats_t board_stats;
volatile uint32_t board_inject;
volatile uint32_t board_replay_go;

static wake_t w;
static volatile uint8_t clip_pending;   /* ISR 가 세우고 board_poll 이 내린다 */
static uint16_t prev_idr;

/* WAKE 올림 → 패킷 → 결과 8 B → WAKE 내림. 반환 = LINK_OK / LINK_ERR_* / 1(타임아웃) */
static int send_clip(const uint16_t *frames, link_result_t *r)
{
  static uint8_t pkt[LINK_PACKET_BYTES(WAKE_NATIVE)];
  uint8_t res[LINK_RESULT_BYTES];
  size_t n = link_frame_packet(frames, WAKE_NATIVE, pkt);
  int rc;

  /* 지난 클립의 늦게 온 바이트를 버린다 */
  __HAL_UART_SEND_REQ(&huart2, UART_RXDATA_FLUSH_REQUEST);
  __HAL_UART_CLEAR_FLAG(&huart2, UART_CLEAR_OREF | UART_CLEAR_FEF | UART_CLEAR_NEF);

  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_SET);      /* FPGA 수신기 초기화 */
  HAL_UART_Transmit(&huart2, pkt, (uint16_t)n, BOARD_TX_MS);

  if (HAL_UART_Receive(&huart2, res, sizeof(res), BOARD_RESULT_MS) != HAL_OK)
  {
    rc = 1;
  }
  else
  {
    rc = link_parse_result(res, r);
  }
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_RESET);

  board_stats.clips++;
  if (rc == LINK_OK)
  {
    board_stats.ok++;
    board_stats.last_class = r->class_idx;
    board_stats.last_keyword = r->keyword_idx;
    board_stats.last_margin = r->margin;
  }
  else if (rc == 1)
  {
    board_stats.timeouts++;
  }
  else
  {
    board_stats.bad++;
  }
  return rc;
}

/*
 * 다음 박자까지 Stop 2. LPTIM1 은 LSE 로 계속 세고 그 인터럽트가 깨운다.
 * EXTI 는 NVIC 가 꺼져 있어 깨우지 않는다 -- 에지는 pending 비트에 남아 다음 박자가 읽는다.
 * 인터럽트를 막은 채 clip_pending 을 보고 들어가야 그 사이 박자를 놓치지 않는다
 * (PRIMASK 가 서 있어도 대기 중인 인터럽트는 WFI 를 깨운다).
 */
#ifndef WAKE_REPLAY
static void enter_stop2(void)
{
  __disable_irq();
  if (!clip_pending)
  {
    HAL_SuspendTick();
    board_stats.stops++;
    HAL_PWREx_EnterSTOP2Mode(PWR_STOPENTRY_WFI);
    HAL_ResumeTick();
  }
  __enable_irq();                                /* 여기서 LPTIM ISR 이 돈다 */
}
#endif

void board_init(void)
{
  memset((void *)&board_stats, 0, sizeof(board_stats));
  board_stats.magic = BOARD_STATS_MAGIC;
  board_inject = 0;
  board_replay_go = 0;
  board_stats.replay_first_bad = 0xFFFFFFFFu;
  wake_init(&w);

#if BOARD_TEST
  GPIOB->PUPDR = 0xAAAAAAAAu;                    /* 16 핀 모두 풀다운 (10b) */
  HAL_DBGMCU_EnableDBGStopMode();
#endif

  /* Stop 2 에서 깨면 MSI (4 MHz, 들어가기 전 범위 그대로) -- SystemClock_Config 재호출 불필요 */
  __HAL_RCC_WAKEUPSTOP_CLK_CONFIG(RCC_STOP_WAKEUPCLOCK_MSI);

  /* 리셋 전에 래치된 에지는 버린다 */
  EXTI->PR1 = 0xFFFFu;
  prev_idr = (uint16_t)GPIOB->IDR;

  /* 재생 빌드도 박자는 돌린다 -- ticks 가 올라가야 read_stats.ps1 이 살아 있다고 본다 */
  if (HAL_LPTIM_Counter_Start_IT(&hlptim1, BOARD_LPTIM_ARR) != HAL_OK)
  {
    Error_Handler();
  }
}

/* 10 ms 박자 (LPTIM1 자동 리로드 일치) */
void HAL_LPTIM_AutoReloadMatchCallback(LPTIM_HandleTypeDef *hlptim)
{
  (void)hlptim;
#ifdef WAKE_REPLAY
  board_stats.ticks++;                           /* 재생은 AFE 를 안 본다 */
  return;
#endif
  uint16_t pr = (uint16_t)(EXTI->PR1 & 0xFFFFu);
  EXTI->PR1 = pr;                                /* 1 을 써서 지운다 */
  uint16_t idr = (uint16_t)GPIOB->IDR;
  uint16_t frame = (uint16_t)(prev_idr | pr);
  prev_idr = idr;

#if BOARD_TEST
  if (board_inject != 0)                         /* 디버거가 넣은 가짜 칸 (한 번만) */
  {
    frame |= (uint16_t)board_inject;
    board_inject = 0;
    board_stats.injected++;
  }
#endif

  board_stats.ticks++;
  board_stats.last_frame = frame;
  board_stats.edges_or |= pr;
  uint8_t on = wake_popcount16(frame);
  if (on > board_stats.max_on)
  {
    board_stats.max_on = on;
  }
#if BOARD_TEST
  for (uint32_t k = 0; pr != 0; k++, pr >>= 1)
  {
    if ((pr & 1u) && board_stats.edges[k] != 0xFFFF)
    {
      board_stats.edges[k]++;
    }
  }
#endif

  if (wake_tick(&w, frame) == WAKE_CLIP_READY)
  {
    clip_pending = 1;
  }
  board_stats.state = w.state;
  board_stats.bg_q = w.bg_q;
}

void board_poll(void)
{
#ifdef WAKE_REPLAY
  static uint32_t i;
  link_result_t r = {0};

  if (board_replay_go == 0)
  {
    __WFI();
    return;
  }
  if (i == 0)                                    /* 한 바퀴 시작: 결과 카운터만 비운다 */
  {
    board_stats.clips = 0;
    board_stats.ok = 0;
    board_stats.timeouts = 0;
    board_stats.bad = 0;
    board_stats.match = 0;
    board_stats.replay_first_bad = 0xFFFFFFFFu;
    board_stats.replay_bad_info = 0;
    board_stats.replay_state = 1;
  }

  int rc = send_clip(replay_frames[i], &r);
  if (rc == LINK_OK && r.class_idx == replay_expect[i])
  {
    board_stats.match++;
  }
  else if (board_stats.replay_first_bad == 0xFFFFFFFFu)
  {
    board_stats.replay_first_bad = i;
    board_stats.replay_bad_info = (uint32_t)r.class_idx | ((uint32_t)replay_expect[i] << 8)
                                | ((uint32_t)(rc & 0xFF) << 16);
  }
  HAL_Delay(BOARD_REPLAY_GAP_MS);

  if (++i == REPLAY_N)
  {
    i = 0;
    board_replay_go = 0;
    board_stats.replay_state = 2;
  }
#else
  link_result_t r;

  if (clip_pending)
  {
    /* HOLD 동안 core 는 clip 을 건드리지 않는다 -- ISR 이 돌아도 그대로 읽어도 된다 */
    send_clip(wake_clip(&w), &r);
    __disable_irq();
    wake_sent(&w);
    clip_pending = 0;
    __enable_irq();
    return;
  }
  enter_stop2();
#if BOARD_TEST
  /* Stop 2 중에는 DBG_STOP 을 켜도 SWD 로 RAM 이 0 으로 읽힌다 (실측). 깨어 있는 창을
   * 넓혀 tools/read_stats.ps1 이 걸리게 한다. 박자는 LPTIM 이 정하므로 판단엔 영향 없음. */
  HAL_Delay(BOARD_TEST_AWAKE_MS);
#endif
#endif
}
