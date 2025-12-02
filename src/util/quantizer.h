#pragma once
#include <cstring>
#include <cuda_fp16.h>

struct W4A16Quantizer {

  static size_t calc_total_buffer_bytes(int rows, int cols,
                                        int BLOCK_SIZE = 64) {
    int transposed_rows = cols;
    int transposed_cols = rows;
    int blocks_per_row = (transposed_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int total_blocks = transposed_rows * blocks_per_row;

    size_t quant_bytes = (size_t)total_blocks * 32;
    size_t table_bytes = (size_t)total_blocks * sizeof(uint32_t);

    return quant_bytes + table_bytes;
  }

  static void unpack_quant_params(uint32_t packed, uint16_t &zero_half,
                                  uint16_t &scale_half) {
    zero_half = static_cast<uint16_t>(packed & 0xFFFFu);
    scale_half = static_cast<uint16_t>((packed >> 16) & 0xFFFFu);
  }

  static void quantize_all(const float *data, int m, int n,
                           unsigned char *quantized_data, uint32_t *table, int BLOCK_SIZE = 64) {

    float *transposed = new float[m * n];
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        transposed[j * m + i] = data[i * n + j];
      }
    }

    int transposed_rows = n;
    int transposed_cols = m;

    int blocks_per_row = (transposed_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int total_blocks = transposed_rows * blocks_per_row;

    int block_id = 0;
    for (int row = 0; row < transposed_rows; ++row) {
      const float *row_data = transposed + row * transposed_cols;

      for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        int start = block_idx * BLOCK_SIZE;
        int end = std::min(start + BLOCK_SIZE, transposed_cols);
        int block_size = end - start;

        unsigned char *block_quantized = quantized_data + block_id * 32;
        float min_val = (block_size > 0) ? row_data[start] : 0.0f;
        float max_val = (block_size > 0) ? row_data[start] : 0.0f;
        for (int i = start + 1; i < end; ++i) {
          min_val = std::min(min_val, row_data[i]);
          max_val = std::max(max_val, row_data[i]);
        }

        float scale = (max_val - min_val) / 15.0f;
        if (scale == 0.0f)
          scale = 1e-8f;
        float zero_point = min_val;

        std::memset(block_quantized, 0, 32);

        for (int i = 0; i < block_size; i += 2) {
          int idx1 = start + i;
          int idx2 = start + i + 1;

          int q1 = static_cast<int>(
              std::floor((row_data[idx1] - zero_point) / scale + 0.5f));
          if (q1 < 0)
            q1 = 0;
          if (q1 > 15)
            q1 = 15;

          int q2 = 0;
          if (idx2 < end) {
            q2 = static_cast<int>(
                std::floor((row_data[idx2] - zero_point) / scale + 0.5f));
            if (q2 < 0)
              q2 = 0;
            if (q2 > 15)
              q2 = 15;
          }

          unsigned char packed =
              static_cast<unsigned char>((q1 & 0x0F) | ((q2 & 0x0F) << 4));
          block_quantized[i / 2] = packed;
        }

        __half zero_h = __float2half(zero_point);
        __half scale_h = __float2half(scale);

        uint16_t zero_half_bits, scale_half_bits;
        std::memcpy(&zero_half_bits, &zero_h, sizeof(zero_half_bits));
        std::memcpy(&scale_half_bits, &scale_h, sizeof(scale_half_bits));

        uint32_t packed = (static_cast<uint32_t>(scale_half_bits) << 16) |
                          static_cast<uint32_t>(zero_half_bits);
        table[block_id] = packed;

        ++block_id;
      }
    }

    delete[] transposed;
  }


    static float dequantize_host(const unsigned char *quantized_data, const uint32_t *table,
                                 int original_row, int original_col,
                                 int m, int n, int BLOCK_SIZE = 64) {
        int transposed_row = original_col;
        int transposed_col = original_row;

        int transposed_rows = n;
        int transposed_cols = m;

        int blocks_per_row = (transposed_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

        int block_idx_in_row = transposed_col / BLOCK_SIZE;
        int offset_in_block = transposed_col % BLOCK_SIZE;

        int global_block_idx = transposed_row * blocks_per_row + block_idx_in_row;

        uint32_t packed_bits = table[global_block_idx];
        uint16_t zero_half_bits, scale_half_bits;
        unpack_quant_params(packed_bits, zero_half_bits, scale_half_bits);

        __half zh, sh;
        std::memcpy(&zh, &zero_half_bits, sizeof(zero_half_bits));
        std::memcpy(&sh, &scale_half_bits, sizeof(scale_half_bits));

        float zero_point = __half2float(zh);
        float scale = __half2float(sh);

        const unsigned char *block_data = quantized_data + global_block_idx * 32;

        int byte_index = offset_in_block / 2;
        int bit_shift = (offset_in_block % 2) * 4;

        uint8_t packed_byte = block_data[byte_index];
        uint8_t quantized_value = (packed_byte >> bit_shift) & 0x0F;

        float dequantized_value = zero_point + static_cast<float>(quantized_value) * scale;
        return dequantized_value;
    }
};