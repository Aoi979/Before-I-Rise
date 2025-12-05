#include <cstdint>
#include <utility>
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
