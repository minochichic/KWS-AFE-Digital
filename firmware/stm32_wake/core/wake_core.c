/* wake_core -- see wake_core.h for what and why. */
#include "wake_core.h"

#include <string.h>

uint8_t wake_popcount16(uint16_t x)
{
    /* SWAR: Cortex-M4 has no popcount instruction */
    x = (uint16_t)(x - ((x >> 1) & 0x5555u));
    x = (uint16_t)((x & 0x3333u) + ((x >> 2) & 0x3333u));
    x = (uint16_t)((x + (x >> 4)) & 0x0F0Fu);
    return (uint8_t)((x + (x >> 8)) & 0x1Fu);
}

void wake_init(wake_t *w)
{
    memset(w, 0, sizeof *w);
    w->state = WAKE_S_WARMUP;
}

const uint16_t *wake_clip(const wake_t *w)
{
    return w->clip;
}

void wake_sent(wake_t *w)
{
    w->sent = 1;
}

/* IDLE's decision for one frame. Returns 1 when this frame completes a run. */
static int idle_frame(wake_t *w, uint8_t on)
{
    int sound = on >= WAKE_K;
#if WAKE_DELTA > 0
    /* on >= background + DELTA, both sides x32 so it stays integer */
    sound = sound && ((int32_t)on << WAKE_BG_SHIFT)
                         >= w->bg_q + ((int32_t)WAKE_DELTA << WAKE_BG_SHIFT);
#endif
    if (sound) {
        w->run++;
    } else {
        w->run = 0;
        /* learn the floor only from quiet frames: bg += (on - bg) / 32 */
        w->bg_q += (int32_t)on - (w->bg_q >> WAKE_BG_SHIFT);
    }
    return w->run >= WAKE_RUN;
}

static void ring_push(wake_t *w, uint16_t frame)
{
#if WAKE_RING > 0
    w->ring[w->head] = frame;
    w->head = (uint8_t)(w->head + 1 == WAKE_RING ? 0 : w->head + 1);
#else
    (void)w; (void)frame;
#endif
}

static wake_event_t clip_add(wake_t *w, uint16_t frame)
{
    w->clip[w->n++] = frame;
    if (w->n < WAKE_NATIVE)
        return WAKE_NONE;
    w->state = WAKE_S_HOLD;
    w->count = WAKE_BUSY;
    w->sent  = 0;
    return WAKE_CLIP_READY;
}

wake_event_t wake_tick(wake_t *w, uint16_t frame)
{
    wake_event_t ev = WAKE_NONE;
    uint8_t on = wake_popcount16((uint16_t)(frame & WAKE_COUNT_MASK));

    switch (w->state) {
    case WAKE_S_WARMUP:
        /* WARMUP == 32 frames, so the sum IS 32 x mean -- bg_q's scale */
        w->bg_q += on;
        if (++w->count == WAKE_WARMUP) {
            w->state = WAKE_S_IDLE;
            w->run = 0;
        }
        break;

    case WAKE_S_HOLD:
        if (w->count > 0) {
            w->count--;
            break;
        }
        if (!w->sent)
            break;
        /* busy is over and the FPGA answered: THIS frame is judged again,
         * like the simulator's first frame after `end + busy` */
        w->state = WAKE_S_IDLE;
        w->run = 0;
        /* fall through */

    case WAKE_S_IDLE:
        if (idle_frame(w, on)) {
            uint8_t i;
            /* the ring holds the WAKE_RING frames before this one, oldest at
             * head: they open the clip, this frame follows */
            for (i = 0; i < WAKE_RING; i++) {
                unsigned j = (unsigned)w->head + i;
                w->clip[i] = w->ring[j >= WAKE_RING ? j - WAKE_RING : j];
            }
            w->n = WAKE_RING;
            w->run = 0;
            w->state = WAKE_S_CAPTURE;
            ev = clip_add(w, frame);
        }
        break;

    case WAKE_S_CAPTURE:
        ev = clip_add(w, frame);
        break;

    default:
        wake_init(w);
        break;
    }

    ring_push(w, frame);
    return ev;
}
