#pragma once

#include <cstddef>

template <int Batch, int Height, int Width, int Channels, int Filters,
          int FilterH, int FilterW, int PadH, int PadW, int StrideH,
          int StrideW, int DilationH, int DilationW>
struct ConvShape {
  static constexpr int n = Batch;
  static constexpr int h = Height;
  static constexpr int w = Width;
  static constexpr int c = Channels;
  static constexpr int k = Filters;
  static constexpr int r = FilterH;
  static constexpr int s = FilterW;
  static constexpr int pad_h = PadH;
  static constexpr int pad_w = PadW;
  static constexpr int stride_h = StrideH;
  static constexpr int stride_w = StrideW;
  static constexpr int dilation_h = DilationH;
  static constexpr int dilation_w = DilationW;
  static constexpr int out_h =
      (h + 2 * pad_h - dilation_h * (r - 1) - 1) / stride_h + 1;
  static constexpr int out_w =
      (w + 2 * pad_w - dilation_w * (s - 1) - 1) / stride_w + 1;
  static constexpr int gemm_m = n * out_h * out_w;
  static constexpr int gemm_n = k;
  static constexpr int gemm_k = r * s * c;
  static constexpr size_t input_elements = static_cast<size_t>(n) * h * w * c;
  static constexpr size_t filter_elements = static_cast<size_t>(k) * r * s * c;
  static constexpr size_t output_elements =
      static_cast<size_t>(gemm_m) * gemm_n;

  static_assert(c % 16 == 0, "input channels must be divisible by 16");
  static_assert(out_h > 0 && out_w > 0, "output must be nonempty");
};
