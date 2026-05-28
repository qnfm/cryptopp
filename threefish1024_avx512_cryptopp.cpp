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

static inline __m512i loadu512(const uint64_t* p) noexcept
{
    return _mm512_loadu_si512(reinterpret_cast<const void*>(p));
}

static inline void storeu512(uint64_t* p, __m512i v) noexcept
{
    _mm512_storeu_si512(reinterpret_cast<void*>(p), v);
}

static inline __m512i set8(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3,
                           uint64_t i4, uint64_t i5, uint64_t i6, uint64_t i7) noexcept
{
    return _mm512_set_epi64(static_cast<long long>(i7), static_cast<long long>(i6),
                            static_cast<long long>(i5), static_cast<long long>(i4),
                            static_cast<long long>(i3), static_cast<long long>(i2),
                            static_cast<long long>(i1), static_cast<long long>(i0));
}

static inline __m512i rotation(unsigned r) noexcept
{
    switch (r & 7U) {
    case 0: return set8(24,13, 8,47, 8,17,22,37);
    case 1: return set8(38,19,10,55,49,18,23,52);
    case 2: return set8(33, 4,51,13,34,41,59,17);
    case 3: return set8( 5,20,48,41,47,28,16,25);
    case 4: return set8(41, 9,37,31,12,47,44,30);
    case 5: return set8(16,34,56,51, 4,53,42,41);
    case 6: return set8(31,44,47,46,19,42,44,25);
    default:return set8( 9,48,35,52,23,31,37,20);
    }
}

static inline void add_subkey_words(uint64_t w[16], const uint64_t* sk) noexcept
{
    for (unsigned i = 0; i < 16; ++i)
        w[i] += sk[i];
}

static inline void sub_subkey_words(uint64_t w[16], const uint64_t* sk) noexcept
{
    for (unsigned i = 0; i < 16; ++i)
        w[i] -= sk[i];
}

static inline void mix_words_avx512(uint64_t w[16], __m512i r) noexcept
{
    __m512i a = _mm512_set_epi64(static_cast<long long>(w[14]), static_cast<long long>(w[12]),
                                 static_cast<long long>(w[10]), static_cast<long long>(w[8]),
                                 static_cast<long long>(w[6]),  static_cast<long long>(w[4]),
                                 static_cast<long long>(w[2]),  static_cast<long long>(w[0]));
    __m512i b = _mm512_set_epi64(static_cast<long long>(w[15]), static_cast<long long>(w[13]),
                                 static_cast<long long>(w[11]), static_cast<long long>(w[9]),
                                 static_cast<long long>(w[7]),  static_cast<long long>(w[5]),
                                 static_cast<long long>(w[3]),  static_cast<long long>(w[1]));

    a = _mm512_add_epi64(a, b);
    b = _mm512_xor_si512(_mm512_rolv_epi64(b, r), a);

    alignas(64) uint64_t aa[8], bb[8];
    storeu512(aa, a);
    storeu512(bb, b);
    w[0]  = aa[0]; w[1]  = bb[0];
    w[2]  = aa[1]; w[3]  = bb[1];
    w[4]  = aa[2]; w[5]  = bb[2];
    w[6]  = aa[3]; w[7]  = bb[3];
    w[8]  = aa[4]; w[9]  = bb[4];
    w[10] = aa[5]; w[11] = bb[5];
    w[12] = aa[6]; w[13] = bb[6];
    w[14] = aa[7]; w[15] = bb[7];
}

static inline void inv_mix_words_avx512(uint64_t w[16], __m512i r) noexcept
{
    __m512i a = _mm512_set_epi64(static_cast<long long>(w[14]), static_cast<long long>(w[12]),
                                 static_cast<long long>(w[10]), static_cast<long long>(w[8]),
                                 static_cast<long long>(w[6]),  static_cast<long long>(w[4]),
                                 static_cast<long long>(w[2]),  static_cast<long long>(w[0]));
    __m512i b = _mm512_set_epi64(static_cast<long long>(w[15]), static_cast<long long>(w[13]),
                                 static_cast<long long>(w[11]), static_cast<long long>(w[9]),
                                 static_cast<long long>(w[7]),  static_cast<long long>(w[5]),
                                 static_cast<long long>(w[3]),  static_cast<long long>(w[1]));

    b = _mm512_rorv_epi64(_mm512_xor_si512(b, a), r);
    a = _mm512_sub_epi64(a, b);

    alignas(64) uint64_t aa[8], bb[8];
    storeu512(aa, a);
    storeu512(bb, b);
    w[0]  = aa[0]; w[1]  = bb[0];
    w[2]  = aa[1]; w[3]  = bb[1];
    w[4]  = aa[2]; w[5]  = bb[2];
    w[6]  = aa[3]; w[7]  = bb[3];
    w[8]  = aa[4]; w[9]  = bb[4];
    w[10] = aa[5]; w[11] = bb[5];
    w[12] = aa[6]; w[13] = bb[6];
    w[14] = aa[7]; w[15] = bb[7];
}

static inline void permute_words(uint64_t w[16]) noexcept
{
    static const unsigned P[16] = {0,9,2,13,6,11,4,15,10,7,12,3,14,5,8,1};
    uint64_t t[16];
    for (unsigned i = 0; i < 16; ++i)
        t[i] = w[P[i]];
    memcpy(w, t, sizeof(t));
}

static inline void inv_permute_words(uint64_t w[16]) noexcept
{
    static const unsigned IP[16] = {0,15,2,11,6,13,4,9,14,1,8,5,10,3,12,7};
    uint64_t t[16];
    for (unsigned i = 0; i < 16; ++i)
        t[i] = w[IP[i]];
    memcpy(w, t, sizeof(t));
}

static void encrypt_words(const uint64_t in[16], uint64_t out[16], const uint64_t* sk) noexcept
{
    uint64_t w[16];
    memcpy(w, in, sizeof(w));

    for (unsigned d = 0; d < 20; ++d) {
        add_subkey_words(w, sk + d * 16);
        for (unsigned r = 0; r < 4; ++r) {
            mix_words_avx512(w, rotation((4 * d + r) & 7U));
            permute_words(w);
        }
    }

    add_subkey_words(w, sk + 20 * 16);
    memcpy(out, w, sizeof(w));
}

static void decrypt_words(const uint64_t in[16], uint64_t out[16], const uint64_t* sk) noexcept
{
    uint64_t w[16];
    memcpy(w, in, sizeof(w));

    sub_subkey_words(w, sk + 20 * 16);
    for (int d = 19; d >= 0; --d) {
        for (int r = 3; r >= 0; --r) {
            inv_permute_words(w);
            inv_mix_words_avx512(w, rotation((static_cast<unsigned>(4 * d + r)) & 7U));
        }
        sub_subkey_words(w, sk + static_cast<unsigned>(d) * 16);
    }

    memcpy(out, w, sizeof(w));
}

static inline void load_block_words(const uint8_t in[128], uint64_t w[16]) noexcept
{
    for (unsigned i = 0; i < 16; ++i)
        w[i] = load_le64(in + 8 * i);
}

static inline void store_block_words_xor(uint8_t out[128], const uint64_t w[16], const uint8_t* xorBlock) noexcept
{
    uint8_t tmp[128];
    for (unsigned i = 0; i < 16; ++i)
        store_le64(tmp + 8 * i, w[i]);

    if (xorBlock) {
        for (unsigned i = 0; i < 128; ++i)
            out[i] = static_cast<uint8_t>(tmp[i] ^ xorBlock[i]);
    } else {
        memcpy(out, tmp, 128);
    }
}

} // namespace cryptopp_threefish1024_avx512

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
    cryptopp_threefish1024_avx512::encrypt_words(in, out, subkeys);
    cryptopp_threefish1024_avx512::store_block_words_xor(outBlock, out, xorBlock);
}

extern "C" void Cryptopp_Threefish1024_AVX512_DecryptBlock(
    const uint8_t inBlock[128], const uint8_t* xorBlock, uint8_t outBlock[128],
    const uint64_t subkeys[21 * 16]) noexcept
{
    uint64_t in[16], out[16];
    cryptopp_threefish1024_avx512::load_block_words(inBlock, in);
    cryptopp_threefish1024_avx512::decrypt_words(in, out, subkeys);
    cryptopp_threefish1024_avx512::store_block_words_xor(outBlock, out, xorBlock);
}

#if defined(__GNUC__) || defined(__clang__)
# pragma GCC pop_options
#endif

#else  // non-x86-64 stub, normally not compiled

#include <stdint.h>
extern "C" int Cryptopp_Threefish1024_AVX512_Available() noexcept { return 0; }
extern "C" void Cryptopp_Threefish1024_AVX512_ExpandKeyFromRKeyTweak(const uint64_t*, const uint64_t*, uint64_t*) noexcept {}
extern "C" void Cryptopp_Threefish1024_AVX512_EncryptBlock(const uint8_t*, const uint8_t*, uint8_t*, const uint64_t*) noexcept {}
extern "C" void Cryptopp_Threefish1024_AVX512_DecryptBlock(const uint8_t*, const uint8_t*, uint8_t*, const uint64_t*) noexcept {}

#endif
