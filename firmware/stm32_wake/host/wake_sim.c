/* Host driver for wake_core: streams in, captures out (tests only).
 *
 * stdin:   S <n>  then n frames in hex          (repeated)
 * stdout:  per capture   C <start> <end>
 *                        D <100 frames in hex>
 *          per stream    E
 * wake_sent() is called right after each clip, so HOLD lasts exactly
 * WAKE_BUSY frames -- the simulator's `busy`.
 */
#include <stdio.h>
#include "wake_core.h"

int main(void)
{
    unsigned n, t, f, i;
    wake_t w;
    while (scanf(" S %u", &n) == 1) {
        wake_init(&w);
        for (t = 0; t < n; t++) {
            if (scanf("%x", &f) != 1)
                return 2;
            if (wake_tick(&w, (uint16_t)f) == WAKE_CLIP_READY) {
                const uint16_t *c = wake_clip(&w);
                printf("C %u %u\nD", t + 1 - WAKE_NATIVE, t + 1);
                for (i = 0; i < WAKE_NATIVE; i++)
                    printf(" %x", c[i]);
                printf("\n");
                wake_sent(&w);
            }
        }
        printf("E\n");
    }
    return 0;
}
