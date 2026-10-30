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
#ifdef WAKE_REPLAY
volatile uint8_t board_replay_got[REPLAY_N];     /* tools/read_stats.ps1 -Replay 가 읽어 보고서를 만든다 */
#endif

volatile uint32_t board_stream_go;
#if defined(WAKE_STREAM) && defined(WAKE_AFELOG)
#error "WAKE_STREAM and WAKE_AFELOG both use board_caps -- pick one"
#endif
#if defined(WAKE_STREAM) || defined(WAKE_AFELOG)
volatile uint32_t board_cap_n;
volatile board_cap_t board_caps[BOARD_MAX_CAPS];
static volatile uint32_t s_ready_end;            /* 클립이 찬 칸 (ISR -> board_poll) */
#endif
#ifdef WAKE_STREAM
#include "../../../gen/stream_clips.h"
static volatile uint8_t  s_on;                   /* 스트림을 먹이는 중 */
static volatile uint32_t s_idx, s_k;             /* 스트림 번호, 다음에 먹일 칸 */
#endif
#ifdef WAKE_AFELOG
volatile board_quiet_t board_quiet;
volatile uint32_t board_log_reset;
#endif

static wake_t w;
static volatile uint8_t clip_pending;   /* ISR 가 세우고 board_poll 이 내린다 */
static uint16_t prev_idr;

static uint8_t pkt[LINK_PACKET_BYTES(WAKE_NATIVE)];

/* WAKE 올림 → 패킷 → 결과 8 B → WAKE 내림. 반환 = LINK_OK / LINK_ERR_* / 1(타임아웃)
 * use_wake = 0 은 WAKE 테스트의 대조군 전용: WAKE 를 안 올리고 보낸다. */
static int send_clip_ex(const uint16_t *frames, link_result_t *r, int use_wake)
{
  static uint8_t res[LINK_RESULT_BYTES];
  size_t n = link_frame_packet(frames, WAKE_NATIVE, pkt);
  int rc;

  /* 지난 클립의 늦게 온 바이트를 버린다 */
  __HAL_UART_SEND_REQ(&huart2, UART_RXDATA_FLUSH_REQUEST);
  __HAL_UART_CLEAR_FLAG(&huart2, UART_CLEAR_OREF | UART_CLEAR_FEF | UART_CLEAR_NEF);

  /* 결과는 인터럽트로 받는다. 폴링(HAL_UART_Receive)이면 그 사이 10 ms 박자 ISR 이 끼어
   * 87 us 마다 오는 바이트를 놓쳐 오버런이 났다 -- 일반 빌드에서 매번 (2026-10-29 실측,
   * ErrorCode 0x08, 전송 끝 57 ms = FPGA 답 도착 시점). USART2 는 LPTIM 보다 우선순위가 높다. */
  HAL_UART_Receive_IT(&huart2, res, sizeof(res));

  if (use_wake)
  {
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_SET);    /* 상승 에지 = FPGA 수신기 초기화 */
  }
  HAL_UART_Transmit(&huart2, pkt, (uint16_t)n, BOARD_TX_MS);

  uint32_t t_rx = HAL_GetTick();
  while (huart2.RxState != HAL_UART_STATE_READY && HAL_GetTick() - t_rx < BOARD_RESULT_MS)
  {
  }
  if (huart2.RxState != HAL_UART_STATE_READY || huart2.ErrorCode != HAL_UART_ERROR_NONE)
  {
    rc = 1;
    board_stats.uart_err = huart2.ErrorCode | ((uint32_t)huart2.RxState << 16);
    board_stats.rx_fail_ms = HAL_GetTick() - t_rx;
    HAL_UART_AbortReceive(&huart2);
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

static int send_clip(const uint16_t *frames, link_result_t *r)
{
  return send_clip_ex(frames, r, 1);
}

#ifdef WAKE_REPLAY
/* 패킷 앞 절반만 보내고 끊는다 -- 바이트가 빠진 링크를 흉내 낸다. FPGA 파서는 그 패킷을
 * 받는 중인 채로 남는다. 다음 패킷 앞에서 WAKE 가 올라가야 그 반쪽이 버려진다. */
static void send_half_packet(const uint16_t *frames)
{
  size_t n = link_frame_packet(frames, WAKE_NATIVE, pkt);
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_SET);
  HAL_UART_Transmit(&huart2, pkt, (uint16_t)(n / 2), BOARD_TX_MS);
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_RESET);
  HAL_Delay(20);
}

/* 시행마다: [반쪽 → WAKE 있이 온전한 클립] 그리고 [반쪽 → WAKE 없이 온전한 클립].
 * WAKE 선이 이어져 있으면 앞은 전부 맞고, 뒤(대조군)는 반쪽에 이어 붙어 CRC 로 버려진다.
 * 선이 끊겼으면 둘 다 틀린다. 대조군이 맞으면 이 테스트가 WAKE 를 재지 못한다는 뜻이다. */
static void wake_test(void)
{
  link_result_t r;
  board_stats.wake_test_n = 0;
  board_stats.wake_with_ok = 0;
  board_stats.wake_without_ok = 0;

  for (uint32_t j = 0; j < BOARD_WAKE_TEST_N; j++)
  {
    send_half_packet(replay_frames[j]);
    memset(&r, 0, sizeof(r));
    if (send_clip_ex(replay_frames[j], &r, 1) == LINK_OK && r.class_idx == replay_expect[j])
    {
      board_stats.wake_with_ok++;
    }
    HAL_Delay(BOARD_REPLAY_GAP_MS);

    send_half_packet(replay_frames[j]);
    memset(&r, 0, sizeof(r));
    if (send_clip_ex(replay_frames[j], &r, 0) == LINK_OK && r.class_idx == replay_expect[j])
    {
      board_stats.wake_without_ok++;
    }
    HAL_Delay(BOARD_REPLAY_GAP_MS);
    board_stats.wake_test_n++;
  }
}
#endif

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
  board_stats.dbgmcu_boot = DBGMCU->CR;
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

  /* UART 수신이 박자 ISR 에 밀리지 않게: USART2 0 (CubeMX), LPTIM1 1 */
  HAL_NVIC_SetPriority(LPTIM1_IRQn, 1, 0);

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

#ifdef WAKE_STREAM
#ifdef BOARD_STREAM_AUTOSTART
  if (board_stats.ticks == (uint32_t)(BOARD_STREAM_AUTOSTART) * 100u && board_stats.replay_state == 0)
  {
    board_stream_go = 1;
  }
#endif
  /* 시작: 보내는 중인 클립이 없을 때만 (wake_init 이 그 클립을 지우므로) */
  if (board_stream_go != 0 && !s_on && board_stats.replay_state != 1 && !clip_pending)
  {
    if (board_stream_go <= STREAM_N)
    {
      s_idx = board_stream_go - 1;
      s_k = 0;
      s_on = 1;
      board_cap_n = 0;
      wake_init(&w);
      board_stats.replay_state = 1;
    }
    board_stream_go = 0;
  }
  if (s_on)                                      /* 핀 대신 스트림 칸 */
  {
    frame = stream_frames[stream_off[s_idx] + s_k];
    if (++s_k == stream_len[s_idx])
    {
      s_on = 0;
    }
  }
#endif

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
#ifdef WAKE_STREAM
    s_ready_end = s_k;                           /* 방금 먹인 칸 번호 + 1 = Capture.end */
#elif defined(WAKE_AFELOG)
    s_ready_end = board_stats.ticks;
#endif
  }
#ifdef WAKE_AFELOG
  if (w.state == WAKE_S_IDLE)                    /* 조용한 박자 (트리거·캡처·HOLD 가 아님) */
  {
    board_quiet.n++;
    for (uint32_t k = 0; k < 16u; k++)
    {
      board_quiet.on[k] += (frame >> k) & 1u;
    }
  }
#endif
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
  if (board_replay_go == 2)
  {
    board_stats.replay_state = 1;
    wake_test();
    board_stats.replay_state = 2;
    board_replay_go = 0;
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
  board_replay_got[i] = rc == LINK_OK ? r.class_idx
                      : rc == 1       ? BOARD_GOT_TIMEOUT : BOARD_GOT_BAD;
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

#ifdef WAKE_AFELOG
  if (board_log_reset != 0)
  {
    board_log_reset = 0;
    __disable_irq();
    board_cap_n = 0;
    memset((void *)&board_quiet, 0, sizeof(board_quiet));
    __enable_irq();
  }
#endif
  if (clip_pending)
  {
    /* HOLD 동안 core 는 clip 을 건드리지 않는다 -- ISR 이 돌아도 그대로 읽어도 된다 */
#if defined(WAKE_STREAM) || defined(WAKE_AFELOG)
    uint32_t c = board_cap_n;
    memset(&r, 0, sizeof(r));
    int rc = send_clip(wake_clip(&w), &r);
#ifdef WAKE_STREAM
    int keep = board_stats.replay_state == 1;
#else
    int keep = 1;                                /* AFE 기록: 처음 BOARD_MAX_CAPS 개, 덮어쓰지 않는다 */
#endif
    if (keep && c < BOARD_MAX_CAPS)
    {
      board_caps[c].end = s_ready_end;
      board_caps[c].got = rc == LINK_OK ? r.class_idx : rc == 1 ? BOARD_GOT_TIMEOUT : BOARD_GOT_BAD;
      board_caps[c].bg_q = (uint16_t)w.bg_q;     /* 캡처·HOLD 동안 배경은 그대로 -- 트리거 때 값 */
      memcpy((void *)board_caps[c].clip, wake_clip(&w), sizeof(board_caps[c].clip));
      board_cap_n = c + 1;
    }
#else
    send_clip(wake_clip(&w), &r);
#endif
    __disable_irq();
    wake_sent(&w);
    clip_pending = 0;
    __enable_irq();
    return;
  }
#ifdef WAKE_STREAM
  if (board_stats.replay_state == 1 && !s_on)    /* 스트림 끝, 보내는 중인 클립도 없음 */
  {
    board_stats.replay_state = 2;
  }
  if (board_stats.replay_state == 2)             /* 끝난 뒤엔 Stop 2 대신 Sleep: board_caps(3 KB)를 */
  {                                              /* SWD 로 읽는 동안 RAM 이 0 으로 읽히지 않게 */
    __WFI();
    return;
  }
#endif
#ifdef WAKE_AFELOG
  __WFI();                                       /* Sleep: board_caps 를 언제든 SWD 로 읽게 */
  return;
#endif
  enter_stop2();
#if BOARD_TEST
  /* Stop 2 중에는 DBG_STOP 을 켜도 SWD 로 RAM 이 0 으로 읽힌다 (실측). 깨어 있는 창을
   * 넓혀 tools/read_stats.ps1 이 걸리게 한다. 박자는 LPTIM 이 정하므로 판단엔 영향 없음. */
  HAL_Delay(BOARD_TEST_AWAKE_MS);
#endif
#endif
}
