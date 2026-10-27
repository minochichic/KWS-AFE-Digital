/* link -- the STM32 <-> FPGA wake link as bytes (docs/ICD.md 7.2).
 *
 *   STM32 -> FPGA   0xA5 | N | frame[0..N-1] (2 bytes, LE, bit k = ch k) | CRC-8
 *   FPGA  -> STM32  0x5A | class_idx | keyword_idx | margin (int32 LE) | CRC-8
 *
 * CRC-8/SMBUS: polynomial 0x07, init 0, no reflection, over every byte before
 * it. Check value crc8("123456789") = 0xF4. The STM32's CRC peripheral can do
 * the same (POLYSIZE 8, POL 0x07, INIT 0, no REV) -- this software version is
 * what the Mac tests check against export/wake_link.py and rtl/kws_crc8.v.
 */
#ifndef LINK_H
#define LINK_H

#include <stddef.h>
#include <stdint.h>

#define LINK_SYNC_IN            0xA5u
#define LINK_SYNC_OUT           0x5Au
#define LINK_PACKET_BYTES(n)    (2u * (unsigned)(n) + 3u)
#define LINK_RESULT_BYTES       8u

typedef struct {
    uint8_t class_idx;
    uint8_t keyword_idx;
    int32_t margin;              /* kws_tail keyword_margin, pool-sum units */
} link_result_t;

enum { LINK_OK = 0, LINK_ERR_SYNC = -1, LINK_ERR_CRC = -2 };

uint8_t link_crc8(const uint8_t *p, size_t n, uint8_t crc);

/* n frames -> out[LINK_PACKET_BYTES(n)]. Returns the byte count. */
size_t  link_frame_packet(const uint16_t *frames, uint8_t n, uint8_t *out);

/* pkt[LINK_RESULT_BYTES] -> *r. Returns LINK_OK or a LINK_ERR_*. */
int     link_parse_result(const uint8_t *pkt, link_result_t *r);

#endif /* LINK_H */
