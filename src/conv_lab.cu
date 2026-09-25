#include "generic_implicit_gemm.cuh"
#include "spatial_strip.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t elements) {
    check(cudaMalloc(&data_, elements * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data_); }
  T *get() { return data_; }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

 private:
  T *data_ = nullptr;
};

template <class Shape>
int32_t reference_value(const std::vector<int8_t> &input,
                        const std::vector<int8_t> &filter,
                        const std::vector<int32_t> &bias, size_t output_index) {
  const int channel = output_index % Shape::k;
  const int m = output_index / Shape::k;
  const int ox = m % Shape::out_w;
  const int oy = (m / Shape::out_w) % Shape::out_h;
  const int batch = m / (Shape::out_h * Shape::out_w);
  int32_t sum = bias[channel];
  for (int fy = 0; fy < Shape::r; ++fy) {
    const int iy = oy * Shape::stride_h + fy * Shape::dilation_h - Shape::pad_h;
    if (iy < 0 || iy >= Shape::h) continue;
    for (int fx = 0; fx < Shape::s; ++fx) {
      const int ix =
          ox * Shape::stride_w + fx * Shape::dilation_w - Shape::pad_w;
      if (ix < 0 || ix >= Shape::w) continue;
      for (int c = 0; c < Shape::c; ++c) {
        const auto x = static_cast<int32_t>(
            input[((batch * Shape::h + iy) * Shape::w + ix) * Shape::c + c]);
        const auto w = static_cast<int32_t>(
            filter[((channel * Shape::r + fy) * Shape::s + fx) * Shape::c + c]);
        sum += x * w;
      }
    }
  }
  return sum;
}

template <class Shape>
bool validate(const std::vector<int8_t> &input, const std::vector<int8_t> &filter,
              const std::vector<int32_t> &bias,
              const std::vector<int32_t> &output, const char *label) {
  const size_t samples =
      Shape::output_elements <= 65536 ? Shape::output_elements : 2048;
  for (size_t sample = 0; sample < samples; ++sample) {
    const size_t index =
        samples == Shape::output_elements ? sample
                                          : sample * Shape::output_elements / samples;
    const int32_t expected = reference_value<Shape>(input, filter, bias, index);
    if (output[index] != expected) {
      std::cerr << label << " mismatch at output " << index << ": expected "
                << expected << ", got " << output[index] << '\n';
      return false;
    }
  }
  return true;
}

template <class Function>
float measure(Function launch, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) launch();
  check(cudaDeviceSynchronize(), "warmup");
  std::vector<float> samples;
  samples.reserve(iterations);
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "cudaEventCreate start");
  check(cudaEventCreate(&stop), "cudaEventCreate stop");
  for (int i = 0; i < iterations; ++i) {
    check(cudaEventRecord(start), "cudaEventRecord start");
    launch();
    check(cudaEventRecord(stop), "cudaEventRecord stop");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    samples.push_back(ms);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

struct Options {
  std::string strategy = "both";
  bool csv = false;
  int warmup = 20;
  int iterations = 101;
};

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--csv") {
      options.csv = true;
    } else if ((arg == "--strategy" || arg == "--warmup" ||
                arg == "--iterations") &&
               i + 1 < argc) {
      const std::string value = argv[++i];
      if (arg == "--strategy") options.strategy = value;
      if (arg == "--warmup") options.warmup = std::stoi(value);
      if (arg == "--iterations") options.iterations = std::stoi(value);
    } else {
      std::cerr << "usage: conv_lab [--strategy generic|strip|both] [--csv] "
                   "[--warmup N] [--iterations N]\n";
      std::exit(EXIT_FAILURE);
    }
  }
  if ((options.strategy != "generic" && options.strategy != "strip" &&
       options.strategy != "both") ||
      options.warmup < 0 || options.iterations <= 0) {
    std::cerr << "invalid benchmark options\n";
    std::exit(EXIT_FAILURE);
  }
  return options;
}

template <class Shape>
int run(const Options &options) {
  std::mt19937 rng(20260925);
  std::uniform_int_distribution<int> values(-4, 4);
  std::uniform_int_distribution<int> bias_values(-31, 31);
  std::vector<int8_t> input(Shape::input_elements);
  std::vector<int8_t> filter(Shape::filter_elements);
  std::vector<int32_t> bias(Shape::k);
  for (auto &value : input) value = static_cast<int8_t>(values(rng));
  for (auto &value : filter) value = static_cast<int8_t>(values(rng));
  for (auto &value : bias) value = bias_values(rng);

  DeviceBuffer<int8_t> d_input(input.size());
  DeviceBuffer<int8_t> d_filter(filter.size());
  DeviceBuffer<int32_t> d_bias(bias.size());
  DeviceBuffer<int32_t> d_output(Shape::output_elements);
  check(cudaMemcpy(d_input.get(), input.data(), input.size(), cudaMemcpyHostToDevice),
        "copy input");
  check(cudaMemcpy(d_filter.get(), filter.data(), filter.size(), cudaMemcpyHostToDevice),
        "copy filter");
  check(cudaMemcpy(d_bias.get(), bias.data(), bias.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice), "copy bias");

  auto generic = [&] {
#if INT8_LAB_MMA_SM80
    launch_generic_sm80<Shape>(d_input.get(), d_filter.get(), d_bias.get(),
                               d_output.get());
#else
    launch_generic<Shape>(d_input.get(), d_filter.get(), d_bias.get(),
                          d_output.get());
#endif
  };
  const bool strip_eligible = kSpatialStripEligible<Shape>;
  float generic_ms = -1.0f;
  float strip_ms = -1.0f;
  bool generic_correct = true;
  bool strip_correct = true;
  std::vector<int32_t> output(Shape::output_elements);

  if (options.strategy != "strip") {
    generic();
    check(cudaGetLastError(), "generic launch");
    check(cudaDeviceSynchronize(), "generic synchronize");
    check(cudaMemcpy(output.data(), d_output.get(),
                     output.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
          "copy generic output");
    generic_correct = validate<Shape>(input, filter, bias, output, "generic");
    if (generic_correct) generic_ms = measure(generic, options.warmup, options.iterations);
  }

  if constexpr (kSpatialStripEligible<Shape>) {
    if (options.strategy != "generic") {
      auto strip = [&] {
        launch_spatial_strip<Shape>(d_input.get(), d_filter.get(), d_bias.get(),
                                    d_output.get());
      };
      strip();
      check(cudaGetLastError(), "spatial-strip launch");
      check(cudaDeviceSynchronize(), "spatial-strip synchronize");
      check(cudaMemcpy(output.data(), d_output.get(),
                       output.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
            "copy spatial-strip output");
      strip_correct =
          validate<Shape>(input, filter, bias, output, "spatial-strip");
      if (strip_correct) {
        strip_ms = measure(strip, options.warmup, options.iterations);
      }
    }
  }

  const char *winner = "none";
  if (generic_ms >= 0 && strip_ms >= 0) winner = strip_ms < generic_ms ? "strip" : "generic";
  else if (generic_ms >= 0) winner = "generic";
  else if (strip_ms >= 0) winner = "strip";

  if (options.csv) {
    std::cout << "case,n,h,w,c,k,r,s,out_h,out_w,eligible_strip,generic_ms,"
                 "strip_ms,winner,generic_correct,strip_correct\n";
    std::cout << kCaseName << ',' << Shape::n << ',' << Shape::h << ',' << Shape::w
              << ',' << Shape::c << ',' << Shape::k << ',' << Shape::r << ','
              << Shape::s << ',' << Shape::out_h << ',' << Shape::out_w << ','
              << (strip_eligible ? 1 : 0) << ',' << generic_ms << ',' << strip_ms
              << ',' << winner << ',' << (generic_correct ? 1 : 0) << ','
              << (strip_correct ? 1 : 0) << '\n';
  } else {
    std::cout << kCaseName << ": generic ";
    if (generic_ms >= 0) std::cout << std::fixed << std::setprecision(4) << generic_ms << " ms";
    else std::cout << "not run";
    std::cout << ", spatial strip ";
    if (!strip_eligible) std::cout << "ineligible";
    else if (strip_ms >= 0) std::cout << std::fixed << std::setprecision(4) << strip_ms << " ms";
    else std::cout << "failed";
    std::cout << ", winner " << winner << '\n';
  }
  return generic_correct && strip_correct && std::string(winner) != "none" ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
  return run<TestShape>(parse_options(argc, argv));
}
