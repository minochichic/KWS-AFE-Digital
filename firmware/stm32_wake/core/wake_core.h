/* wake_core -- the STM32's decision logic, with no hardware in it.
 *
 * One call per 10 ms frame. A frame is 16 bits, bit k = channel k = "comparator
 * k was on at some point in these 10 ms" (board.c builds it from the EXTI
 * pending bits and GPIOB->IDR). The core decides, from frames alone, when a
 * word has started and hands back the 100-frame clip to send to the FPGA.
 *
 *   wake_init(&w);
 *   every 10 ms:   if (wake_tick(&w, frame) == WAKE_CLIP_READY) {
 *                      send(wake_clip(&w), WAKE_NATIVE);   -- board.c
 *                      ...result received or timed out...
 *                      wake_sent(&w);
 *                  }
 *
 * No registers, no interrupts, no clock: the same file builds for the STM32
 * and for the Mac, where tests/test_firmware_core.py checks it frame for
 * frame against experiments/eval_trigger.py::simulate_triggers.
 *
 * States
 *   WARMUP   the first WAKE_WARMUP frames: only learn the background
 *   IDLE     every frame: is it sound? (on >= K and on >= background + DELTA)
 *            quiet frames update the background, sound frames do not
 *   CAPTURE  the clip is the ring (PRE + RUN - 1 frames) + the trigger frame
 *            + frames as they come, until WAKE_NATIVE
 *   HOLD     WAKE_BUSY frames, and until wake_sent(): no trigger, no learning
 * The ring takes every frame in every state, so a clip may reach back into the
 * previous one -- exactly what the simulator does.
 */
#ifndef WAKE_CORE_H
#define WAKE_CORE_H

#include <stdint.h>
#include "wake_cfg.h"

typedef enum {
    WAKE_NONE = 0,
    WAKE_CLIP_READY = 1          /* wake_clip() holds WAKE_NATIVE frames */
} wake_event_t;

enum { WAKE_S_WARMUP = 0, WAKE_S_IDLE, WAKE_S_CAPTURE, WAKE_S_HOLD };

#define WAKE_RING_ALLOC (WAKE_RING > 0 ? WAKE_RING : 1)

typedef struct {
    uint16_t ring[WAKE_RING_ALLOC]; /* last WAKE_RING frames, oldest at head */
    uint16_t clip[WAKE_NATIVE];
    int32_t  bg_q;               /* background on-count x 32                  */
    uint8_t  head;               /* ring write position = oldest frame        */
    uint8_t  n;                  /* frames in clip while capturing            */
    uint8_t  state;
    uint8_t  count;              /* warm-up frames seen / hold frames left    */
    uint8_t  run;                /* consecutive sound frames                  */
    uint8_t  sent;               /* wake_sent() since the last clip           */
} wake_t;

void            wake_init(wake_t *w);
wake_event_t    wake_tick(wake_t *w, uint16_t frame);
const uint16_t *wake_clip(const wake_t *w);
void            wake_sent(wake_t *w);
uint8_t         wake_popcount16(uint16_t x);

#endif /* WAKE_CORE_H */
