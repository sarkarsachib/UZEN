/* Upstream-faithful reference dequantizers for Q4_K and Q6_K.
 *
 * The structs and dequantization loops mirror ggml's ggml-quants.c /
 * ggml-common.h. Compiled by tests only; this is not part of the engine.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct __attribute__((packed)) {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

typedef struct __attribute__((packed)) {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
} block_q6_K;

static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static void dequantize_q4_k(const block_q4_K * x, float * y) {
    const uint8_t * q = x->qs;
    const float d = half_to_float(x->d);
    const float min = half_to_float(x->dmin);
    int is = 0;
    uint8_t sc, m;
    for (int j = 0; j < 256; j += 64) {
        get_scale_min_k4(is + 0, x->scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, x->scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

static void dequantize_q6_k(const block_q6_K * x, float * y) {
    const float d = half_to_float(x->d);
    const uint8_t * ql = x->ql;
    const uint8_t * qh = x->qh;
    const int8_t * sc = x->scales;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    FILE * file = fopen(argv[2], "rb");
    if (!file) return 3;
    float out[256];
    if (strcmp(argv[1], "q4k") == 0) {
        block_q4_K block;
        if (fread(&block, sizeof(block), 1, file) != 1) return 4;
        dequantize_q4_k(&block, out);
    } else if (strcmp(argv[1], "q6k") == 0) {
        block_q6_K block;
        if (fread(&block, sizeof(block), 1, file) != 1) return 4;
        dequantize_q6_k(&block, out);
    } else {
        return 5;
    }
    fclose(file);
    for (int i = 0; i < 256; ++i) printf("%.9g\n", out[i]);
    return 0;
}
