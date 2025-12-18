#include <cstdint>
#include <utility>
#include <type_traits>

template <uint32_t stride>
std::pair<uint32_t, uint32_t> bank_coord(uint32_t i, uint32_t j) {
  uint32_t idx = i * stride + j;
  uint32_t bank_i = idx / 32;
  uint32_t bank_j = idx % 32;
  return {bank_i, bank_j};
}

template <uint32_t block_size> uint32_t swizzle_j(uint32_t i, uint32_t j) {
  uint32_t blocks_per_row = 32 / block_size;
  return (i ^ (j / block_size)) % block_size * block_size;
}


template <int B, int M, int S>
struct Swizzle
{
    static_assert(B >= 0, "B must be >= 0");
    static_assert(M >= 0, "M must be >= 0");

    static constexpr int mask = (1 << B) - 1;

    static constexpr int hi_shift = M + (S > 0 ? S : 0);
    static constexpr int lo_shift = M - (S < 0 ? S : 0);

    /// Swizzle a linear element index
    template <typename Index>
    static constexpr Index apply(Index i)
    {
        static_assert(std::is_integral<Index>::value,
                      "Index must be an integral type");

        return i ^ (((i >> hi_shift) & mask) << lo_shift);
    }
};
