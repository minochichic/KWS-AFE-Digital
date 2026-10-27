/* link -- see link.h. */
#include "link.h"

uint8_t link_crc8(const uint8_t *p, size_t n, uint8_t crc)
{
    unsigned c = crc;            /* unsigned: no int promotion in the shifts */
    while (n--) {
        uint8_t b;
        c ^= *p++;
        for (b = 0; b < 8; b++)
            c = ((c & 0x80u) ? (c << 1) ^ 0x07u : c << 1) & 0xFFu;
    }
    return (uint8_t)c;
}

size_t link_frame_packet(const uint16_t *frames, uint8_t n, uint8_t *out)
{
    size_t k = 0;
    uint8_t i;
    out[k++] = LINK_SYNC_IN;
    out[k++] = n;
    for (i = 0; i < n; i++) {
        out[k++] = (uint8_t)(frames[i] & 0xFFu);
        out[k++] = (uint8_t)(frames[i] >> 8);
    }
    out[k] = link_crc8(out, k, 0);
    return k + 1;
}

int link_parse_result(const uint8_t *pkt, link_result_t *r)
{
    uint32_t m;
    if (pkt[0] != LINK_SYNC_OUT)
        return LINK_ERR_SYNC;
    if (link_crc8(pkt, LINK_RESULT_BYTES - 1, 0) != pkt[LINK_RESULT_BYTES - 1])
        return LINK_ERR_CRC;
    m = (uint32_t)pkt[3] | (uint32_t)pkt[4] << 8 | (uint32_t)pkt[5] << 16
      | (uint32_t)pkt[6] << 24;
    r->class_idx   = pkt[1];
    r->keyword_idx = pkt[2];
    /* two's complement without relying on the implementation-defined
     * unsigned -> signed conversion */
    r->margin      = (m & 0x80000000u) ? -(int32_t)(~m) - 1 : (int32_t)m;
    return LINK_OK;
}
