// threefish1024_avx512_cryptopp.cpp
// Optional AVX-512F Threefish-1024 block routes for Crypto++.
//
// Integration contract:
//   - Compile this file only on x86-64 targets.
//   - Define CRYPTOPP_THREEFISH1024_AVX512_AVAILABLE when building Crypto++
//     and when compiling threefish.cpp/threefish.h if you want the dispatch path.
//   - Runtime dispatch is guarded by Cryptopp_Threefish1024_AVX512_Available().
//   - The scalar Threefish implementation remains the fallback.
//
// GCC/Clang can compile this file without global -mavx512f because of the
// target pragma below. MSVC still needs /arch:AVX512 for this translation unit.

#if defined(__x86_64__) || defined(_M_X64)

#include <stdint.h>
#include <string.h>

#if defined(_MSC_VER)
# include <intrin.h>
#else
# include <cpuid.h>
#endif

// Runtime detection must NOT be compiled under target("avx512f") to avoid
// the compiler emitting AVX-512 instructions in the detection path itself.
namespace cryptopp_threefish1024_avx512_detect {

static uint64_t xgetbv0() noexcept
{
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}

static void cpuidex(int out[4], int leaf, int subleaf) noexcept
{
#if defined(_MSC_VER)
    __cpuidex(out, leaf, subleaf);
#else
    uint32_t a, b, c, d;
    __cpuid_count(static_cast<uint32_t>(leaf), static_cast<uint32_t>(subleaf), a, b, c, d);
    out[0] = static_cast<int>(a);
    out[1] = static_cast<int>(b);
    out[2] = static_cast<int>(c);
    out[3] = static_cast<int>(d);
#endif
}

static bool has_avx512f_runtime() noexcept
{
    int regs[4] = {0,0,0,0};
    cpuidex(regs, 0, 0);
    if (regs[0] < 7)
        return false;

    cpuidex(regs, 1, 0);
    const bool osxsave = (static_cast<uint32_t>(regs[2]) & (1U << 27)) != 0;
    const bool avx = (static_cast<uint32_t>(regs[2]) & (1U << 28)) != 0;
    if (!osxsave || !avx)
        return false;

    const uint64_t xcr0 = xgetbv0();
    if ((xcr0 & 0xE6U) != 0xE6U)
        return false;

    cpuidex(regs, 7, 0);
    const bool avx512f = (static_cast<uint32_t>(regs[1]) & (1U << 16)) != 0;
    return avx512f;
}

} // namespace cryptopp_threefish1024_avx512_detect

// AVX-512 computation code below this point.
#if defined(__GNUC__) || defined(__clang__)
# pragma GCC push_options
# pragma GCC target("avx512f")
#endif

#include <immintrin.h>

namespace cryptopp_threefish1024_avx512 {

static inline uint64_t load_le64(const uint8_t* p) noexcept
{
    uint64_t v;
    memcpy(&v, p, sizeof(v));
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    v = __builtin_bswap64(v);
#endif
    return v;
}

static inline void store_le64(uint8_t* p, uint64_t v) noexcept
{
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    v = __builtin_bswap64(v);
#endif
    memcpy(p, &v, sizeof(v));
}

// Rotation constants for each sub-round (indexed by round mod 8).
// Lane order: pair(0,1), pair(2,3), pair(4,5), pair(6,7), pair(8,9), pair(10,11), pair(12,13), pair(14,15)
static inline __m512i get_rotation(unsigned r) noexcept
{
    // Stored as aligned arrays; lane order [0..7] = pair(0,1)..pair(14,15).
    alignas(64) static const int64_t ROTS[8][8] = {
        {24, 13,  8, 47,  8, 17, 22, 37},
        {38, 19, 10, 55, 49, 18, 23, 52},
        {33,  4, 51, 13, 34, 41, 59, 17},
        { 5, 20, 48, 41, 47, 28, 16, 25},
        {41,  9, 37, 31, 12, 47, 44, 30},
        {16, 34, 56, 51,  4, 53, 42, 41},
        {31, 44, 47, 46, 19, 42, 44, 25},
        { 9, 48, 35, 52, 23, 31, 37, 20},
    };
    return _mm512_load_si512(&ROTS[r & 7U]);
}

// Forward permutation on evens: new_evens[i] = old_evens[PERM_E[i]]
// Derived from Threefish-1024 permutation {0,9,2,13,6,11,4,15,10,7,12,3,14,5,8,1}
static inline __m512i get_perm_e() noexcept {
    alignas(64) static const int64_t v[8] = {0, 1, 3, 2, 5, 6, 7, 4};
    return _mm512_load_si512(v);
}
// Forward permutation on odds: new_odds[i] = old_odds[PERM_O[i]]
static inline __m512i get_perm_o() noexcept {
    alignas(64) static const int64_t v[8] = {4, 6, 5, 7, 3, 1, 2, 0};
    return _mm512_load_si512(v);
}
// Inverse permutation on evens
static inline __m512i get_inv_perm_e() noexcept {
    alignas(64) static const int64_t v[8] = {0, 1, 3, 2, 7, 4, 5, 6};
    return _mm512_load_si512(v);
}
// Inverse permutation on odds
static inline __m512i get_inv_perm_o() noexcept {
    alignas(64) static const int64_t v[8] = {7, 5, 6, 4, 0, 2, 1, 3};
    return _mm512_load_si512(v);
}

// Indices to deinterleave 16 contiguous words into even/odd halves
// using _mm512_permutex2var_epi64(lo8, idx, hi8)
static inline __m512i get_deinterleave_even() noexcept {
    alignas(64) static const int64_t v[8] = {0, 2, 4, 6, 8, 10, 12, 14};
    return _mm512_load_si512(v);
}
static inline __m512i get_deinterleave_odd() noexcept {
    alignas(64) static const int64_t v[8] = {1, 3, 5, 7, 9, 11, 13, 15};
    return _mm512_load_si512(v);
}

// Indices to reinterleave even/odd halves back to sequential word order
static inline __m512i get_interleave_lo() noexcept {
    alignas(64) static const int64_t v[8] = {0, 8, 1, 9, 2, 10, 3, 11};
    return _mm512_load_si512(v);
}
static inline __m512i get_interleave_hi() noexcept {
    alignas(64) static const int64_t v[8] = {4, 12, 5, 13, 6, 14, 7, 15};
    return _mm512_load_si512(v);
}

static void encrypt_block_avx512(const uint64_t in[16], uint64_t out[16],
                                  const uint64_t* sk) noexcept
{
    const __m512i perm_e = get_perm_e();
    const __m512i perm_o = get_perm_o();
    const __m512i di_even = get_deinterleave_even();
    const __m512i di_odd  = get_deinterleave_odd();

    // Load state and deinterleave into even/odd lanes
    __m512i lo = _mm512_loadu_si512(in);
    __m512i hi = _mm512_loadu_si512(in + 8);
    __m512i evens = _mm512_permutex2var_epi64(lo, di_even, hi);
    __m512i odds  = _mm512_permutex2var_epi64(lo, di_odd,  hi);

    for (unsigned d = 0; d < 20; ++d) {
        // Subkey injection
        const uint64_t* subkey = sk + d * 16;
        __m512i sk_lo = _mm512_loadu_si512(subkey);
        __m512i sk_hi = _mm512_loadu_si512(subkey + 8);
        evens = _mm512_add_epi64(evens, _mm512_permutex2var_epi64(sk_lo, di_even, sk_hi));
        odds  = _mm512_add_epi64(odds,  _mm512_permutex2var_epi64(sk_lo, di_odd,  sk_hi));

        for (unsigned r = 0; r < 4; ++r) {
            // Mix
            __m512i rot = get_rotation((4 * d + r) & 7U);
            evens = _mm512_add_epi64(evens, odds);
            odds  = _mm512_xor_si512(_mm512_rolv_epi64(odds, rot), evens);
            // Permute
            evens = _mm512_permutexvar_epi64(perm_e, evens);
            odds  = _mm512_permutexvar_epi64(perm_o, odds);
        }
    }

    // Final subkey injection (subkey 20)
    const uint64_t* subkey = sk + 20 * 16;
    __m512i sk_lo = _mm512_loadu_si512(subkey);
    __m512i sk_hi = _mm512_loadu_si512(subkey + 8);
    evens = _mm512_add_epi64(evens, _mm512_permutex2var_epi64(sk_lo, di_even, sk_hi));
    odds  = _mm512_add_epi64(odds,  _mm512_permutex2var_epi64(sk_lo, di_odd,  sk_hi));

    // Reinterleave and store
    const __m512i il_lo = get_interleave_lo();
    const __m512i il_hi = get_interleave_hi();
    _mm512_storeu_si512(out, _mm512_permutex2var_epi64(evens, il_lo, odds));
    _mm512_storeu_si512(out + 8, _mm512_permutex2var_epi64(evens, il_hi, odds));
}

static void decrypt_block_avx512(const uint64_t in[16], uint64_t out[16],
                                  const uint64_t* sk) noexcept
{
    const __m512i inv_perm_e = get_inv_perm_e();
    const __m512i inv_perm_o = get_inv_perm_o();
    const __m512i di_even = get_deinterleave_even();
    const __m512i di_odd  = get_deinterleave_odd();

    // Load state and deinterleave into even/odd lanes
    __m512i lo = _mm512_loadu_si512(in);
    __m512i hi = _mm512_loadu_si512(in + 8);
    __m512i evens = _mm512_permutex2var_epi64(lo, di_even, hi);
    __m512i odds  = _mm512_permutex2var_epi64(lo, di_odd,  hi);

    // Subtract final subkey (subkey 20)
    const uint64_t* subkey = sk + 20 * 16;
    __m512i sk_lo = _mm512_loadu_si512(subkey);
    __m512i sk_hi = _mm512_loadu_si512(subkey + 8);
    evens = _mm512_sub_epi64(evens, _mm512_permutex2var_epi64(sk_lo, di_even, sk_hi));
    odds  = _mm512_sub_epi64(odds,  _mm512_permutex2var_epi64(sk_lo, di_odd,  sk_hi));

    for (int d = 19; d >= 0; --d) {
        for (int r = 3; r >= 0; --r) {
            // Inverse permute
            evens = _mm512_permutexvar_epi64(inv_perm_e, evens);
            odds  = _mm512_permutexvar_epi64(inv_perm_o, odds);
            // Inverse mix
            __m512i rot = get_rotation((4 * d + r) & 7U);
            odds  = _mm512_rorv_epi64(_mm512_xor_si512(odds, evens), rot);
            evens = _mm512_sub_epi64(evens, odds);
        }
        // Subtract subkey
        subkey = sk + static_cast<unsigned>(d) * 16;
        sk_lo = _mm512_loadu_si512(subkey);
        sk_hi = _mm512_loadu_si512(subkey + 8);
        evens = _mm512_sub_epi64(evens, _mm512_permutex2var_epi64(sk_lo, di_even, sk_hi));
        odds  = _mm512_sub_epi64(odds,  _mm512_permutex2var_epi64(sk_lo, di_odd,  sk_hi));
    }

    // Reinterleave and store
    const __m512i il_lo = get_interleave_lo();
    const __m512i il_hi = get_interleave_hi();
    _mm512_storeu_si512(out, _mm512_permutex2var_epi64(evens, il_lo, odds));
    _mm512_storeu_si512(out + 8, _mm512_permutex2var_epi64(evens, il_hi, odds));
}

static inline void load_block_words(const uint8_t in[128], uint64_t w[16]) noexcept
{
    for (unsigned i = 0; i < 16; ++i)
        w[i] = load_le64(in + 8 * i);
}

static inline void store_block_words_xor(uint8_t out[128], const uint64_t w[16], const uint8_t* xorBlock) noexcept
{
    if (xorBlock) {
        for (unsigned i = 0; i < 128; ++i)
            out[i] = static_cast<uint8_t>(reinterpret_cast<const uint8_t*>(w)[i] ^ xorBlock[i]);
    } else {
        memcpy(out, w, 128);
    }
}

// =====================================================================
// Multi-block parallel encryption (word-parallel approach).
// Processes 8 blocks simultaneously: each __m512i holds word[i] from 8 blocks.
// Permutation is FREE (register renaming). Rotation is broadcast scalar.
// =====================================================================

// Threefish-1024 rotation constants: R[round_mod_8][pair]
static const uint64_t TF1024_ROTATIONS[8][8] = {
    {24, 13,  8, 47,  8, 17, 22, 37},
    {38, 19, 10, 55, 49, 18, 23, 52},
    {33,  4, 51, 13, 34, 41, 59, 17},
    { 5, 20, 48, 41, 47, 28, 16, 25},
    {41,  9, 37, 31, 12, 47, 44, 30},
    {16, 34, 56, 51,  4, 53, 42, 41},
    {31, 44, 47, 46, 19, 42, 44, 25},
    { 9, 48, 35, 52, 23, 31, 37, 20},
};

// Threefish-1024 word permutation
static const unsigned TF1024_PERM[16] = {0,9,2,13,6,11,4,15,10,7,12,3,14,5,8,1};
// Inverse permutation
static const unsigned TF1024_INV_PERM[16] = {0,15,2,11,6,13,4,9,14,1,8,5,10,3,12,7};

static void encrypt_8blocks_parallel(
    __m512i state[16],      // in/out: state[i] = word i of 8 blocks
    const uint64_t* sk      // pre-expanded subkeys (21 × 16 words)
) noexcept
{
    for (unsigned d = 0; d < 20; ++d) {
        // Subkey injection
        const uint64_t* subkey = sk + d * 16;
        for (unsigned i = 0; i < 16; ++i)
            state[i] = _mm512_add_epi64(state[i], _mm512_set1_epi64(static_cast<long long>(subkey[i])));

        for (unsigned r = 0; r < 4; ++r) {
            const uint64_t* rc = TF1024_ROTATIONS[(4 * d + r) & 7U];
            // Mix: 8 pairs (0,1),(2,3),...,(14,15)
            for (unsigned p = 0; p < 8; ++p) {
                __m512i rot = _mm512_set1_epi64(static_cast<long long>(rc[p]));
                state[2*p]   = _mm512_add_epi64(state[2*p], state[2*p+1]);
                state[2*p+1] = _mm512_xor_si512(_mm512_rolv_epi64(state[2*p+1], rot), state[2*p]);
            }
            // Permute (register renaming)
            __m512i tmp[16];
            for (unsigned i = 0; i < 16; ++i)
                tmp[i] = state[TF1024_PERM[i]];
            for (unsigned i = 0; i < 16; ++i)
                state[i] = tmp[i];
        }
    }

    // Final subkey injection (subkey 20)
    const uint64_t* subkey = sk + 20 * 16;
    for (unsigned i = 0; i < 16; ++i)
        state[i] = _mm512_add_epi64(state[i], _mm512_set1_epi64(static_cast<long long>(subkey[i])));
}

static void decrypt_8blocks_parallel(
    __m512i state[16],
    const uint64_t* sk
) noexcept
{
    // Subtract final subkey (subkey 20)
    const uint64_t* subkey = sk + 20 * 16;
    for (unsigned i = 0; i < 16; ++i)
        state[i] = _mm512_sub_epi64(state[i], _mm512_set1_epi64(static_cast<long long>(subkey[i])));

    for (int d = 19; d >= 0; --d) {
        for (int r = 3; r >= 0; --r) {
            // Inverse permute
            __m512i tmp[16];
            for (unsigned i = 0; i < 16; ++i)
                tmp[i] = state[TF1024_INV_PERM[i]];
            for (unsigned i = 0; i < 16; ++i)
                state[i] = tmp[i];
            // Inverse mix
            const uint64_t* rc = TF1024_ROTATIONS[(4 * d + r) & 7U];
            for (unsigned p = 0; p < 8; ++p) {
                __m512i rot = _mm512_set1_epi64(static_cast<long long>(rc[p]));
                state[2*p+1] = _mm512_rorv_epi64(_mm512_xor_si512(state[2*p+1], state[2*p]), rot);
                state[2*p]   = _mm512_sub_epi64(state[2*p], state[2*p+1]);
            }
        }
        // Subtract subkey
        subkey = sk + static_cast<unsigned>(d) * 16;
        for (unsigned i = 0; i < 16; ++i)
            state[i] = _mm512_sub_epi64(state[i], _mm512_set1_epi64(static_cast<long long>(subkey[i])));
    }
}

} // namespace cryptopp_threefish1024_avx512

// =====================================================================
// Extern C interface
// =====================================================================

extern "C" int Cryptopp_Threefish1024_AVX512_Available() noexcept
{
    return cryptopp_threefish1024_avx512_detect::has_avx512f_runtime() ? 1 : 0;
}

extern "C" void Cryptopp_Threefish1024_AVX512_ExpandKeyFromRKeyTweak(
    const uint64_t rkey[17], const uint64_t tweak[3], uint64_t subkeys[21 * 16]) noexcept
{
    for (unsigned s = 0; s <= 20; ++s) {
        uint64_t* dst = subkeys + s * 16;
        for (unsigned i = 0; i < 16; ++i)
            dst[i] = rkey[(s + i) % 17];
        dst[13] += tweak[s % 3];
        dst[14] += tweak[(s + 1) % 3];
        dst[15] += s;
    }
}

extern "C" void Cryptopp_Threefish1024_AVX512_EncryptBlock(
    const uint8_t inBlock[128], const uint8_t* xorBlock, uint8_t outBlock[128],
    const uint64_t subkeys[21 * 16]) noexcept
{
    uint64_t in[16], out[16];
    cryptopp_threefish1024_avx512::load_block_words(inBlock, in);
    cryptopp_threefish1024_avx512::encrypt_block_avx512(in, out, subkeys);
    cryptopp_threefish1024_avx512::store_block_words_xor(outBlock, out, xorBlock);
}

extern "C" void Cryptopp_Threefish1024_AVX512_DecryptBlock(
    const uint8_t inBlock[128], const uint8_t* xorBlock, uint8_t outBlock[128],
    const uint64_t subkeys[21 * 16]) noexcept
{
    uint64_t in[16], out[16];
    cryptopp_threefish1024_avx512::load_block_words(inBlock, in);
    cryptopp_threefish1024_avx512::decrypt_block_avx512(in, out, subkeys);
    cryptopp_threefish1024_avx512::store_block_words_xor(outBlock, out, xorBlock);
}

extern "C" size_t Cryptopp_Threefish1024_AVX512_AdvancedProcessBlocks(
    const uint8_t* inBlocks, const uint8_t* xorBlocks, uint8_t* outBlocks,
    size_t length, unsigned int flags, const uint64_t subkeys[21 * 16],
    int encrypt) noexcept
{
    using namespace cryptopp_threefish1024_avx512;

    static const size_t BLOCK_SIZE = 128;
    static const unsigned int BT_InBlockIsCounter = 1;
    static const unsigned int BT_DontIncrementInOutPointers = 2;
    static const unsigned int BT_XorInput = 4;
    static const unsigned int BT_ReverseDirection = 8;
    static const unsigned int BT_AllowParallel = 16;

    size_t inIncrement = (flags & (BT_InBlockIsCounter | BT_DontIncrementInOutPointers)) ? 0 : BLOCK_SIZE;
    size_t xorIncrement = xorBlocks ? BLOCK_SIZE : 0;
    size_t outIncrement = (flags & BT_DontIncrementInOutPointers) ? 0 : BLOCK_SIZE;

    if (flags & BT_ReverseDirection) {
        inBlocks += length - BLOCK_SIZE;
        if (xorBlocks)
            xorBlocks += length - BLOCK_SIZE;
        outBlocks += length - BLOCK_SIZE;
        inIncrement = 0 - inIncrement;
        xorIncrement = 0 - xorIncrement;
        outIncrement = 0 - outIncrement;
    }

    const bool xorInput = xorBlocks && (flags & BT_XorInput);
    const bool xorOutput = xorBlocks && !(flags & BT_XorInput);

    // Process 8 blocks in parallel (skip reverse direction & dont-increment cases)
    if ((flags & BT_AllowParallel) && !(flags & BT_ReverseDirection) &&
        !(flags & BT_DontIncrementInOutPointers)) {
        while (length >= 8 * BLOCK_SIZE) {
            __m512i state[16];

            if (flags & BT_InBlockIsCounter) {
                // CTR mode: base counter + offsets for word 15
                uint64_t base[16];
                load_block_words(inBlocks, base);
                for (unsigned w = 0; w < 15; ++w)
                    state[w] = _mm512_set1_epi64(static_cast<long long>(base[w]));

                // Byte 127 (MSB of word 15 on LE x86) is the counter byte
                alignas(64) static const uint64_t CTR_OFFSETS[8] = {
                    0ULL, 1ULL << 56, 2ULL << 56, 3ULL << 56,
                    4ULL << 56, 5ULL << 56, 6ULL << 56, 7ULL << 56
                };
                state[15] = _mm512_add_epi64(
                    _mm512_set1_epi64(static_cast<long long>(base[15])),
                    _mm512_load_si512(CTR_OFFSETS));

                const_cast<uint8_t*>(inBlocks)[BLOCK_SIZE - 1] += 8;
            } else {
                // Load 8 independent blocks → word-parallel layout
                for (unsigned w = 0; w < 16; ++w) {
                    alignas(64) uint64_t words[8];
                    for (unsigned b = 0; b < 8; ++b)
                        words[b] = load_le64(inBlocks + b * BLOCK_SIZE + w * 8);
                    state[w] = _mm512_load_si512(words);
                }
                inBlocks += 8 * BLOCK_SIZE;
            }

            if (xorInput) {
                for (unsigned w = 0; w < 16; ++w) {
                    alignas(64) uint64_t xw[8];
                    for (unsigned b = 0; b < 8; ++b)
                        xw[b] = load_le64(xorBlocks + b * xorIncrement + w * 8);
                    state[w] = _mm512_xor_si512(state[w], _mm512_load_si512(xw));
                }
                xorBlocks += 8 * xorIncrement;
            }

            // Core: encrypt or decrypt 8 blocks in parallel
            if (encrypt)
                encrypt_8blocks_parallel(state, subkeys);
            else
                decrypt_8blocks_parallel(state, subkeys);

            // Transpose back and store: dump state[16] to memory once
            alignas(64) uint64_t flat[16][8]; // flat[word][block]
            for (unsigned w = 0; w < 16; ++w)
                _mm512_store_si512(flat[w], state[w]);

            // Write out each block
            for (unsigned b = 0; b < 8; ++b) {
                uint8_t* dst = outBlocks;
                if (xorOutput) {
                    for (unsigned w = 0; w < 16; ++w) {
                        uint64_t val = flat[w][b];
                        for (unsigned j = 0; j < 8; ++j)
                            dst[w * 8 + j] = static_cast<uint8_t>(val >> (j * 8)) ^ xorBlocks[w * 8 + j];
                    }
                    xorBlocks += xorIncrement;
                } else {
                    for (unsigned w = 0; w < 16; ++w)
                        memcpy(dst + w * 8, &flat[w][b], 8);
                }
                outBlocks += outIncrement;
            }

            length -= 8 * BLOCK_SIZE;
        }
    }

    // Process remaining blocks one at a time
    while (length >= BLOCK_SIZE) {
        uint64_t in[16], out[16];
        load_block_words(inBlocks, in);

        if (xorInput) {
            uint64_t xw[16];
            load_block_words(xorBlocks, xw);
            for (unsigned i = 0; i < 16; ++i) in[i] ^= xw[i];
        }

        if (encrypt)
            encrypt_block_avx512(in, out, subkeys);
        else
            decrypt_block_avx512(in, out, subkeys);

        if (xorOutput)
            store_block_words_xor(outBlocks, out, xorBlocks);
        else
            memcpy(outBlocks, out, 128);

        if (flags & BT_InBlockIsCounter)
            const_cast<uint8_t*>(inBlocks)[BLOCK_SIZE - 1]++;

        inBlocks += inIncrement;
        if (xorBlocks)
            xorBlocks += xorIncrement;
        outBlocks += outIncrement;
        length -= BLOCK_SIZE;
    }

    return length;
}

#if defined(__GNUC__) || defined(__clang__)
# pragma GCC pop_options
#endif

#else  // non-x86-64 stub, normally not compiled

#include <stdint.h>
#include <stddef.h>
extern "C" int Cryptopp_Threefish1024_AVX512_Available() noexcept { return 0; }
extern "C" void Cryptopp_Threefish1024_AVX512_ExpandKeyFromRKeyTweak(const uint64_t*, const uint64_t*, uint64_t*) noexcept {}
extern "C" void Cryptopp_Threefish1024_AVX512_EncryptBlock(const uint8_t*, const uint8_t*, uint8_t*, const uint64_t*) noexcept {}
extern "C" void Cryptopp_Threefish1024_AVX512_DecryptBlock(const uint8_t*, const uint8_t*, uint8_t*, const uint64_t*) noexcept {}
extern "C" size_t Cryptopp_Threefish1024_AVX512_AdvancedProcessBlocks(const uint8_t*, const uint8_t*, uint8_t*, size_t length, unsigned int, const uint64_t*, int) noexcept { return length; }

#endif
