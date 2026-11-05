// src/kernels/cpu/tq_avx2.cpp - the trellis expert rows (TQ2_T / TQK6 / TQK7) on AVX2 + FMA (tq_cpu.hpp).
// Compiled for AVX2 only and reached behind cpu_avx2_ok(), so nothing here can fault on an older CPU.
//
// Per 128-weight block: the 32 states in scalar code (tq_states.hpp: one 32-bit read, a shift and a mask each, used
// as table indices straight away), four 8-byte state-table rows per YMM register as scalar loads, then per token
// VPMADDWD against the unpacked int16 activations and a VPHADDD tree that leaves the four exact sub-block sums
// S[k] in one XMM, so the float part is the scalar kernel's, lane for lane.  The weights are decoded once per block
// for all the tokens of the group.
//
// Measured on Zen 5 (cycles per block, one token, AVX2): ~48 with this scheme; computing the states with byte
// shuffles instead (VPSHUFB / VPSRLVD per 8 states) and moving them to GPRs or through memory was slower (60-80),
// and ggml-cpu's AVX2 vec_dot for these types (agentionai llama.cpp, the same state table) takes about 2.3x as long.
#include "strata/kernels/cpu/tq_cpu.hpp"

#include "tq_states.hpp"

#include <immintrin.h>

#include <algorithm>

namespace strata::kernels::cpu::tq_detail {
namespace {

// four state-table rows (8 bytes each) into one YMM: a load and three broadcast loads blended in.  Broadcast loads
// and VPBLENDD avoid the shuffle port that VPINSRQ / VINSERTI128 need (the two measure the same on Zen 5).
inline __m256i rows4(const long long* tab, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
    __m256i v = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) (tab + s0)));
    v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s1]), 0x0C);
    v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s2]), 0x30);
    return _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s3]), 0xC0);
}

template <int KB>
void rows_t(const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt, float* const* out, int r0,
            int r1) {
    const int nb = n / 128;
    constexpr size_t bsz = 2 + 4 * KB;
    const long long* tab = (const long long*) table;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        __m128 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            const uint8_t* qs = blk + 2;
            const uint64_t wr = tq_wrap<KB>(qs);
            __m256i wv[8];   // wv[m] = weights 16m .. 16m+15 (states 4m .. 4m+3)
#if defined(__GNUC__)
#pragma GCC unroll 8
#endif
            for (int m = 0; m < 8; ++m)
                wv[m] = rows4(tab, tq_state<KB>(qs, wr, 4 * m), tq_state<KB>(qs, wr, 4 * m + 1),
                              tq_state<KB>(qs, wr, 4 * m + 2), tq_state<KB>(qs, wr, 4 * m + 3));
            const __m128 dx = _mm_set1_ps(tq_half_to_float(tq_ld16(blk)));
            for (int t = 0; t < nt; ++t) {
                const __m256i* q = (const __m256i*) (a[t]->q + 128 * i);
                const __m256i p0 = _mm256_add_epi32(_mm256_madd_epi16(wv[0], _mm256_load_si256(q + 0)),
                                                    _mm256_madd_epi16(wv[1], _mm256_load_si256(q + 1)));
                const __m256i p1 = _mm256_add_epi32(_mm256_madd_epi16(wv[2], _mm256_load_si256(q + 2)),
                                                    _mm256_madd_epi16(wv[3], _mm256_load_si256(q + 3)));
                const __m256i p2 = _mm256_add_epi32(_mm256_madd_epi16(wv[4], _mm256_load_si256(q + 4)),
                                                    _mm256_madd_epi16(wv[5], _mm256_load_si256(q + 5)));
                const __m256i p3 = _mm256_add_epi32(_mm256_madd_epi16(wv[6], _mm256_load_si256(q + 6)),
                                                    _mm256_madd_epi16(wv[7], _mm256_load_si256(q + 7)));
                // per 128-bit lane [p0, p1, p2, p3] partial sums, then the two lanes added: S[0..3], exact
                const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(p0, p1), _mm256_hadd_epi32(p2, p3));
                const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
                const __m128 sc = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + 4 * i));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s), sc, acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = ((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f);
        }
    }
}

// TQP (147-149): 64 pair-steps per 128-value block.  A step's two weights are the FIRST pair of its state
// table row, i.e. codebook point tq_point(s): the 2048 points as int16 pairs (tq_pair_table, 8 KiB) stay in L1,
// where the 256 KiB state table's rows did not.
//
// The points come 8 states at a time in SIMD (TqpPlan): two 16-byte windows of the block's 8*KB stream bytes (one
// per 128-bit lane, never past the block), VPSHUFB to each state's 4 bytes, VPSRLVD by its bit offset, the mask, then
// the hash (VPMULLD, VPSRLD).  The 2-3 states whose 15 bits run past the end of the stream are redone in scalar code.
// Unlike the 4-weight types above (no hash), the SIMD states pay here because the hash goes with them.  Each YMM of
// 16 int16 weights is then built from 8 table loads (VMOVD / VPINSRD), so a 32-value group (16 steps) is two YMM of
// weights against two YMM of activations: VPMADDWD, then per token rows_t's VPHADDD tree leaves the four exact group
// sums in one XMM, so the float part is the scalar kernel's.
//
// Measured on Zen 3 (5800X, TQP5-7 rows of 2048, 96 MiB per thread), ns per block at one / four tokens:
//   1 thread    21 / 28-29   this kernel
//               26-28 / 33-34  scalar states and hash, the pairs through a stack buffer
//               39-41 / 56-62  scalar states, 8-byte state-table loads per pair, a horizontal sum per group
//   8 threads   22 / 31  against 28 / 35 and 41 / 62 (15 threads on the 8 cores: no more than 8)
// VPGATHERDD for the pairs (from this table or a state-indexed one) was slower than the scalar loads there.
struct TqpPlan {
    alignas(32) uint8_t shuf[8][32];   // per 8 states: each state's 4 bytes from its lane's window
    alignas(32) uint32_t shift[8][8];  // and its bit offset in them
    int base[8][2];                    // the window of each 128-bit lane, bytes from the stream's start
    int nwrap;
    int wrap_t[8];                     // the states that run past the end of the stream
};

// constexpr, so each kernel sees its plan as constants (the windows, the shifts and which states wrap): built at run
// time behind a guard instead, the same kernel measured 27 ns per block, not 21
constexpr TqpPlan make_tqp_plan(int kb) {
    TqpPlan p{};
    const int nb = 8 * kb;
    for (int k = 0; k < 8; ++k)
        for (int l = 0; l < 2; ++l) {
            const int base = std::min(((63 - (8 * k + 4 * l + 3)) * kb) >> 3, nb - 16);   // the lane's lowest state
            p.base[k][l] = base;
            for (int j = 0; j < 4; ++j) {
                const int t = 8 * k + 4 * l + j, off = (63 - t) * kb, b = off >> 3;
                const bool wraps = ((off + 14) >> 3) >= nb;
                if (wraps) p.wrap_t[p.nwrap++] = t;
                for (int q = 0; q < 4; ++q) {
                    const int ix = b - base + q;
                    p.shuf[k][16 * l + 4 * j + q] = (uint8_t) (wraps || ix < 0 || ix > 15 ? 0x80 : ix);
                }
                p.shift[k][4 * l + j] = (uint32_t) (off & 7);
            }
        }
    return p;
}

template <int KB>
inline constexpr TqpPlan kTqpPlan = make_tqp_plan(KB);

// the codebook points of a block's 64 states (tq_point(tq_state_tqp(qs, t)))
template <int KB>
inline void tqp_points(const uint8_t* qs, const TqpPlan& p, uint32_t* pt) {
    const __m256i mask = _mm256_set1_epi32((int) kTqStateMask), mul = _mm256_set1_epi32((int) 0x9e3779b1u);
#pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
        const __m256i v0 = _mm256_set_m128i(_mm_loadu_si128((const __m128i*) (qs + p.base[k][1])),
                                            _mm_loadu_si128((const __m128i*) (qs + p.base[k][0])));
        __m256i v = _mm256_shuffle_epi8(v0, _mm256_load_si256((const __m256i*) p.shuf[k]));
        v = _mm256_and_si256(_mm256_srlv_epi32(v, _mm256_load_si256((const __m256i*) p.shift[k])), mask);
        _mm256_store_si256((__m256i*) (pt + 8 * k), _mm256_srli_epi32(_mm256_mullo_epi32(v, mul), 21));
    }
    for (int i = 0; i < p.nwrap; ++i) pt[p.wrap_t[i]] = tq_point(tq_state_tqp<KB>(qs, p.wrap_t[i]));
}

// 8 steps' pairs into one YMM
inline __m256i pairs8(const int* tab, const uint32_t* pt) {
    __m128i lo = _mm_cvtsi32_si128(tab[pt[0]]), hi = _mm_cvtsi32_si128(tab[pt[4]]);
    lo = _mm_insert_epi32(lo, tab[pt[1]], 1);
    hi = _mm_insert_epi32(hi, tab[pt[5]], 1);
    lo = _mm_insert_epi32(lo, tab[pt[2]], 2);
    hi = _mm_insert_epi32(hi, tab[pt[6]], 2);
    lo = _mm_insert_epi32(lo, tab[pt[3]], 3);
    hi = _mm_insert_epi32(hi, tab[pt[7]], 3);
    return _mm256_set_m128i(hi, lo);
}

template <int KB>
void rows_tqp_t(const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt, float* const* out, int r0,
                int r1) {
    const int nb = n / 128;
    constexpr size_t bsz = 2 + 8 * KB;
    const int* pairs = (const int*) tq_pair_table();
    constexpr const TqpPlan& plan = kTqpPlan<KB>;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        __m128 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            alignas(32) uint32_t pt[64];
            tqp_points<KB>(blk + 2, plan, pt);
            __m256i wv[8];   // wv[2g], wv[2g+1]: the weights of the 32-value group g
#pragma GCC unroll 8
            for (int k = 0; k < 8; ++k) wv[k] = pairs8(pairs, pt + 8 * k);
            const __m128 dx = _mm_set1_ps(tq_half_to_float(tq_ld16(blk)));
            for (int t = 0; t < nt; ++t) {
                const __m256i* q = (const __m256i*) (a[t]->q + 128 * i);
                const __m256i p0 = _mm256_add_epi32(_mm256_madd_epi16(wv[0], _mm256_load_si256(q + 0)),
                                                    _mm256_madd_epi16(wv[1], _mm256_load_si256(q + 1)));
                const __m256i p1 = _mm256_add_epi32(_mm256_madd_epi16(wv[2], _mm256_load_si256(q + 2)),
                                                    _mm256_madd_epi16(wv[3], _mm256_load_si256(q + 3)));
                const __m256i p2 = _mm256_add_epi32(_mm256_madd_epi16(wv[4], _mm256_load_si256(q + 4)),
                                                    _mm256_madd_epi16(wv[5], _mm256_load_si256(q + 5)));
                const __m256i p3 = _mm256_add_epi32(_mm256_madd_epi16(wv[6], _mm256_load_si256(q + 6)),
                                                    _mm256_madd_epi16(wv[7], _mm256_load_si256(q + 7)));
                const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(p0, p1), _mm256_hadd_epi32(p2, p3));
                const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
                const __m128 sc = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + 4 * i));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s), sc, acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = ((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f);
        }
    }
}

}  // namespace

void tq_rows_avx2_tqp(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                      float* const* out, int r0, int r1) {
    if (kb == 5) rows_tqp_t<5>(w, row_bytes, n, a, nt, out, r0, r1);
    else if (kb == 6) rows_tqp_t<6>(w, row_bytes, n, a, nt, out, r0, r1);
    else rows_tqp_t<7>(w, row_bytes, n, a, nt, out, r0, r1);
}

void tq_rows_avx2(int kb, const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                  float* const* out, int r0, int r1) {
    if (kb == 8) rows_t<8>(table, w, row_bytes, n, a, nt, out, r0, r1);
    else if (kb == 6) rows_t<6>(table, w, row_bytes, n, a, nt, out, r0, r1);
    else if (kb == 5) rows_t<5>(table, w, row_bytes, n, a, nt, out, r0, r1);   // experimental TQK5X
    else if (kb == 4) rows_t<4>(table, w, row_bytes, n, a, nt, out, r0, r1);   // experimental TQK4X
    else rows_t<7>(table, w, row_bytes, n, a, nt, out, r0, r1);
}

}  // namespace strata::kernels::cpu::tq_detail
