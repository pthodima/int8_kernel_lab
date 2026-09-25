// cuDNN baseline for one compile-time ConvShape.  Emits one CSV row measuring
// both cuDNN contracts; see cudnn_graph.cuh for why the legacy INT8x32 path is
// not used.
#include "cudnn_graph.cuh"

#include <cstdio>
#include <random>
#include <vector>

int main() {
  using Shape = TestShape;

  // Same generator and seed as conv_lab, so both binaries convolve identical
  // tensors and the timings describe the same problem.
  std::mt19937 rng(20260925);
  std::uniform_int_distribution<int> values(-4, 4);
  std::vector<int8_t> input(Shape::input_elements);
  std::vector<int8_t> filter(Shape::filter_elements);
  for (auto &value : input) value = static_cast<int8_t>(values(rng));
  for (auto &value : filter) value = static_cast<int8_t>(values(rng));

  const CudnnGraphResult f32 =
      run_cudnn_graph<Shape>(CudnnOutput::kFloatOutput, input, filter);
  const CudnnGraphResult i8 =
      run_cudnn_graph<Shape>(CudnnOutput::kInt8Output, input, filter);

  std::printf(
      "case,cudnn_f32_ms,cudnn_int8_ms,cudnn_f32_engines,cudnn_int8_engines,"
      "cudnn_f32_validated,cudnn_int8_validated,cudnn_note\n");
  std::printf("%s,%.6f,%.6f,%d,%d,%d,%d,%s\n", kCaseName, f32.milliseconds,
              i8.milliseconds, f32.engines, i8.engines,
              f32.validated ? 1 : 0, i8.validated ? 1 : 0,
              f32.validated ? (i8.validated ? "both" : i8.note) : f32.note);
  return 0;
}
