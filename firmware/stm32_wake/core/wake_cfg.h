/* The wake trigger rule -- docs/wake_stm32_plan.md 2.3, decided 2026-10-26
 * (picked on val from 120 policies, confirmed once on test).
 *
 * experiments/eval_trigger.py FIRMWARE_POLICY carries the same numbers, and
 * tests/test_firmware_core.py fails if the two drift apart. Change both or
 * neither: the recall / duty / false-alarm figures in the plan were measured
 * for exactly these values.
 *
 * Every value can be overridden with -D, which is how the host tests compile
 * other policies to check the core against the simulator beyond the default.
 */
#ifndef WAKE_CFG_H
#define WAKE_CFG_H

#ifndef WAKE_K
#define WAKE_K           1      /* channels on for a frame to count as sound   */
#endif
#ifndef WAKE_DELTA
#define WAKE_DELTA       4      /* ... AND on >= background + DELTA (0 = off)  */
#endif
#ifndef WAKE_RUN
#define WAKE_RUN         1      /* consecutive sound frames to trigger         */
#endif
#ifndef WAKE_PRE
#define WAKE_PRE         30     /* frames kept from before the onset           */
#endif
#ifndef WAKE_BUSY
#define WAKE_BUSY        8      /* frames ignored after a clip (UART + FPGA)    */
#endif
#ifndef WAKE_COUNT_MASK
#define WAKE_COUNT_MASK  0xFFFFu /* channels that are counted (all 16)          */
#endif

/* Fixed by the model and by the arithmetic, not policy knobs. */
#define WAKE_NATIVE      100    /* frames per clip: 1 s of 10 ms (KWS_NATIVE_T) */
#define WAKE_BG_SHIFT    5      /* background EMA weight 1/32                   */
#define WAKE_WARMUP      (1 << WAKE_BG_SHIFT)  /* listen this long before arming */

/* History the clip needs: the PRE frames before the onset plus the RUN-1
 * frames of the run before the trigger frame itself. */
#define WAKE_RING        (WAKE_PRE + WAKE_RUN - 1)

#if WAKE_K < 1 || WAKE_RUN < 1 || WAKE_DELTA < 0 || WAKE_BUSY < 0
#error "wake_cfg.h: K and RUN must be >= 1, DELTA and BUSY >= 0"
#endif
#if WAKE_RING + 1 > WAKE_NATIVE
#error "wake_cfg.h: PRE + RUN must fit inside one clip"
#endif
/* The earliest trigger is frame WARMUP + RUN - 1; its clip starts PRE + RUN - 1
 * frames earlier. PRE <= WARMUP keeps that inside what has been heard, so the
 * ring is always full when the core arms and the simulator never clamps. */
#if WAKE_PRE > WAKE_WARMUP
#error "wake_cfg.h: PRE must not exceed WARMUP"
#endif

#endif /* WAKE_CFG_H */
