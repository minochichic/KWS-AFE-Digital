/* Host driver for link.c (tests only).
 *
 *   link_cli crc  <bytes in hex...>     -> CRC-8 in hex
 *   link_cli pkt  (stdin: n, then n frames in hex)  -> packet bytes in hex
 *   link_cli res  <8 bytes in hex>      -> "<rc> <class> <keyword> <margin>"
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "link.h"

int main(int argc, char **argv)
{
    if (argc >= 2 && !strcmp(argv[1], "crc")) {
        uint8_t b[256];
        int i, n = argc - 2;
        for (i = 0; i < n; i++)
            b[i] = (uint8_t)strtoul(argv[i + 2], NULL, 16);
        printf("%02x\n", link_crc8(b, (size_t)n, 0));
        return 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "pkt")) {
        uint16_t fr[255];
        uint8_t out[LINK_PACKET_BYTES(255)];
        unsigned n, i, f;
        size_t k;
        if (scanf("%u", &n) != 1 || n == 0 || n > 255)
            return 2;
        for (i = 0; i < n; i++) {
            if (scanf("%x", &f) != 1)
                return 2;
            fr[i] = (uint16_t)f;
        }
        k = link_frame_packet(fr, (uint8_t)n, out);
        for (i = 0; i < k; i++)
            printf("%s%02x", i ? " " : "", out[i]);
        printf("\n");
        return 0;
    }
    if (argc == 10 && !strcmp(argv[1], "res")) {
        uint8_t p[LINK_RESULT_BYTES];
        link_result_t r = {0, 0, 0};
        int i, rc;
        for (i = 0; i < 8; i++)
            p[i] = (uint8_t)strtoul(argv[i + 2], NULL, 16);
        rc = link_parse_result(p, &r);
        printf("%d %u %u %ld\n", rc, r.class_idx, r.keyword_idx, (long)r.margin);
        return 0;
    }
    fprintf(stderr, "usage: link_cli crc|pkt|res ...\n");
    return 2;
}
