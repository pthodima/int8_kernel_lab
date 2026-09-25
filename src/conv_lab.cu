#include "dispatch_policy.cuh"
#include "generic_implicit_gemm.cuh"
#include "spatial_halo.cuh"
#include "spatial_strip.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>
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

size_t greatest_common_divisor(size_t a, size_t b) {
  while (b != 0) {
    const size_t next = a % b;
    a = b;
    b = next;
  }
  return a;
}

// A stride that shares a factor with output_elements walks a proper subgroup.
// Because output_elements == gemm_m * k, any stride divisible by k pins every
// sample to output channel 0 -- which is what the old sample * n / count walk
// did, hiding an entire class of epilogue bugs.  A stride coprime to
// output_elements has full period, so consecutive samples advance through
// every output channel and every ox.
size_t full_period_stride(size_t elements) {
  // Coprimality alone is not enough: a stride near elements/3 is coprime but
  // still lands on a handful of output rows, because the quotient it advances
  // the row index by stays nearly constant.  Scaling by 1/phi is the standard
  // low-discrepancy choice and reaches every channel, ox and oy on all the
  // shapes in workloads.csv.
  size_t stride = static_cast<size_t>(static_cast<double>(elements) * 0.6180339887498949);
  stride |= 1;
  while (greatest_common_divisor(stride, elements) != 1) stride += 2;
  return stride;
}

// Indices that a uniform walk is unlikely to land on but that kernels are
// likely to get wrong: the padded border, the first and last pixel of an
// output row (the spatial-strip and halo tap slide), and the output-channel
// tail (the odd-K epilogue).
template <class Shape>
void append_directed_indices(std::vector<size_t> &indices) {
  const int ys[] = {0, 1, Shape::out_h / 2, Shape::out_h - 2, Shape::out_h - 1};
  const int xs[] = {0, 1, 2, Shape::out_w / 2, Shape::out_w - 3, Shape::out_w - 2,
                    Shape::out_w - 1};
  const int ks[] = {0, 1, Shape::k / 2, Shape::k - 2, Shape::k - 1};
  const int ns[] = {0, Shape::n - 1};
  for (int batch : ns) {
    for (int y : ys) {
      for (int x : xs) {
        for (int channel : ks) {
          if (batch < 0 || y < 0 || x < 0 || channel < 0) continue;
          if (y >= Shape::out_h || x >= Shape::out_w || channel >= Shape::k) continue;
          indices.push_back(
              ((static_cast<size_t>(batch) * Shape::out_h + y) * Shape::out_w + x) *
                  Shape::k +
              channel);
        }
      }
    }
  }
}

template <class Shape>
bool validate(const std::vector<int8_t> &input, const std::vector<int8_t> &filter,
              const std::vector<int32_t> &bias,
              const std::vector<int32_t> &output, const char *label,
              size_t requested_samples = 4096) {
  std::vector<size_t> indices;
  if (Shape::output_elements <= 65536) {
    indices.resize(Shape::output_elements);
    for (size_t i = 0; i < Shape::output_elements; ++i) indices[i] = i;
  } else {
    const size_t stride = full_period_stride(Shape::output_elements);
    indices.reserve(requested_samples + 512);
    size_t index = 0;
    for (size_t sample = 0; sample < requested_samples; ++sample) {
      indices.push_back(index);
      index += stride;
      if (index >= Shape::output_elements) index -= Shape::output_elements;
    }
    append_directed_indices<Shape>(indices);
  }
  for (size_t index : indices) {
    const int32_t expected = reference_value<Shape>(input, filter, bias, index);
    if (output[index] != expected) {
      const size_t m = index / Shape::k;
      std::cerr << label << " mismatch at output " << index << " (n="
                << m / (Shape::out_h * Shape::out_w) << " oy="
                << (m / Shape::out_w) % Shape::out_h << " ox=" << m % Shape::out_w
                << " k=" << index % Shape::k << "): expected " << expected
                << ", got " << output[index] << '\n';
      return false;
    }
  }
  return true;
}

// Reports what the sampler actually reached, so a future change to the walk
// cannot silently narrow coverage again.
template <class Shape>
void report_coverage(std::ostream &stream) {
  if (Shape::output_elements <= 65536) {
    stream << "coverage: exhaustive (" << Shape::output_elements << " outputs)\n";
    return;
  }
  const size_t stride = full_period_stride(Shape::output_elements);
  size_t index = 0;
  std::vector<char> channels(Shape::k, 0), xs(Shape::out_w, 0), ys(Shape::out_h, 0);
  for (size_t sample = 0; sample < 4096; ++sample) {
    const size_t m = index / Shape::k;
    channels[index % Shape::k] = 1;
    xs[m % Shape::out_w] = 1;
    ys[(m / Shape::out_w) % Shape::out_h] = 1;
    index += stride;
    if (index >= Shape::output_elements) index -= Shape::output_elements;
  }
  auto count = [](const std::vector<char> &v) {
    return std::count(v.begin(), v.end(), 1);
  };
  stream << "coverage: " << count(channels) << "/" << Shape::k << " channels, "
         << count(xs) << "/" << Shape::out_w << " ox, " << count(ys) << "/"
         << Shape::out_h << " oy\n";
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
  // "both" keeps its name for compatibility but now means "every candidate".
  std::string strategy = "both";
  bool csv = false;
  bool coverage = false;
  int warmup = 20;
  int iterations = 101;
};

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--csv") {
      options.csv = true;
    } else if (arg == "--coverage") {
      options.coverage = true;
    } else if ((arg == "--strategy" || arg == "--warmup" ||
                arg == "--iterations") &&
               i + 1 < argc) {
      const std::string value = argv[++i];
      if (arg == "--strategy") options.strategy = value;
      if (arg == "--warmup") options.warmup = std::stoi(value);
      if (arg == "--iterations") options.iterations = std::stoi(value);
    } else {
      std::cerr << "usage: conv_lab [--strategy generic|strip|halo|both] "
                   "[--csv] [--coverage] [--warmup N] [--iterations N]\n";
      std::exit(EXIT_FAILURE);
    }
  }
  if ((options.strategy != "generic" && options.strategy != "strip" &&
       options.strategy != "halo" && options.strategy != "splitk" &&
       options.strategy != "policy" && options.strategy != "both") ||
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

  std::vector<int32_t> output(Shape::output_elements);
  // Run a candidate once, check it against the CPU oracle, and time it only if
  // it was correct, so a broken kernel can never win a dispatch comparison.
  auto evaluate = [&](auto launch, const char *label, float &milliseconds,
                      bool &correct) {
    launch();
    check(cudaGetLastError(), label);
    check(cudaDeviceSynchronize(), label);
    check(cudaMemcpy(output.data(), d_output.get(),
                     output.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
          label);
    correct = validate<Shape>(input, filter, bias, output, label);
    if (correct) milliseconds = measure(launch, options.warmup, options.iterations);
  };

  const bool strip_eligible = kSpatialStripEligible<Shape>;
  // The halo search space is (stage_k, splits).  stage_k is not a global
  // default: 64 is best when the patch is compact, but stride or a large
  // filter inflates the halo until the staged slab, not the register file,
  // binds occupancy -- a 4x32 patch at stride 2 needs 41 KB and gets 2 CTAs/SM
  // at stage_k 64 against 21 KB and 4 at 32.  Measured on big_3x3_s2, halving
  // it is worth 1.55x; on big_3x3_s1 it costs 10%.  So both are candidates and
  // the dispatch table records which one won.
  using HaloTile64 = HaloTile<INT8_LAB_HALO_TILE_H, INT8_LAB_HALO_TILE_W, 64>;
  using HaloTile32 = HaloTile<INT8_LAB_HALO_TILE_H, INT8_LAB_HALO_TILE_W, 32>;
  const bool halo_eligible =
      kSpatialHaloEligible<Shape, HaloTile64> || kSpatialHaloEligible<Shape, HaloTile32>;
  const bool splitk_eligible =
      (kSpatialHaloEligible<Shape, HaloTile64> && kSplitKUsable<Shape, HaloTile64, 2>) ||
      (kSpatialHaloEligible<Shape, HaloTile32> && kSplitKUsable<Shape, HaloTile32, 2>);
  float generic_ms = -1.0f, strip_ms = -1.0f, halo_ms = -1.0f;
  bool generic_correct = true, strip_correct = true, halo_correct = true;
  int halo_stage_k = 0;
  float halo_splitk_ms = -1.0f;
  int halo_splitk_splits = 0, halo_splitk_stage_k = 0;
  bool halo_splitk_correct = true;

  if (options.strategy == "generic" || options.strategy == "both") {
    evaluate(generic, "generic", generic_ms, generic_correct);
  }
  if constexpr (kSpatialStripEligible<Shape>) {
    if (options.strategy == "strip" || options.strategy == "both") {
      evaluate([&] {
        launch_spatial_strip<Shape>(d_input.get(), d_filter.get(),
                                    d_bias.get(), d_output.get());
      }, "spatial-strip", strip_ms, strip_correct);
    }
  }

  // Split-K stays a separate candidate rather than a mode of the halo, because
  // it trades a zeroing pass plus atomics for CTA count: a dispatch policy has
  // to see both numbers and the configuration that produced each.
  auto try_halo = [&](auto stage_tag, auto split_tag) {
    constexpr int kStageK = decltype(stage_tag)::value;
    constexpr int kSplits = decltype(split_tag)::value;
    using Tile = HaloTile<INT8_LAB_HALO_TILE_H, INT8_LAB_HALO_TILE_W, kStageK>;
    if constexpr (kSpatialHaloEligible<Shape, Tile> &&
                  kSplitKUsable<Shape, Tile, kSplits>) {
      const bool wanted = kSplits == 1
                              ? (options.strategy == "halo" ||
                                 options.strategy == "both")
                              : (options.strategy == "halo" ||
                                 options.strategy == "splitk" ||
                                 options.strategy == "both");
      if (wanted) {
        float ms = -1.0f;
        bool ok = true;
        evaluate([&] {
          launch_spatial_halo<Shape, Tile, kSplits>(
              d_input.get(), d_filter.get(), d_bias.get(), d_output.get());
        }, kSplits == 1 ? "spatial-halo" : "spatial-halo-splitk", ms, ok);
        if constexpr (kSplits == 1) {
          if (!ok) halo_correct = false;
          if (ms >= 0 && (halo_ms < 0 || ms < halo_ms)) {
            halo_ms = ms;
            halo_stage_k = kStageK;
          }
        } else {
          if (!ok) halo_splitk_correct = false;
          if (ms >= 0 && (halo_splitk_ms < 0 || ms < halo_splitk_ms)) {
            halo_splitk_ms = ms;
            halo_splitk_splits = kSplits;
            halo_splitk_stage_k = kStageK;
          }
        }
      }
    }
  };
  auto try_stage = [&](auto stage_tag) {
    try_halo(stage_tag, std::integral_constant<int, 1>{});
    try_halo(stage_tag, std::integral_constant<int, 2>{});
    try_halo(stage_tag, std::integral_constant<int, 4>{});
    try_halo(stage_tag, std::integral_constant<int, 8>{});
    try_halo(stage_tag, std::integral_constant<int, 16>{});
  };
  try_stage(std::integral_constant<int, 64>{});
  try_stage(std::integral_constant<int, 32>{});

  // The policy is evaluated as one more candidate so the sweep records its
  // regret: how much slower its single compile-time choice is than the best
  // any candidate achieved.  A policy is only worth shipping if that stays
  // near 1.00x.
  constexpr DispatchDecision kPolicy = select_conv_kernel<Shape>();
  float policy_ms = -1.0f;
  bool policy_correct = true;
  if (options.strategy == "policy" || options.strategy == "both") {
    evaluate([&] {
      launch_by_policy<Shape>(d_input.get(), d_filter.get(), d_bias.get(),
                              d_output.get());
    }, "policy", policy_ms, policy_correct);
  }

  const char *winner = "none";
  float best = -1.0f;
  if (generic_ms >= 0) { winner = "generic"; best = generic_ms; }
  if (strip_ms >= 0 && (best < 0 || strip_ms < best)) { winner = "strip"; best = strip_ms; }
  if (halo_ms >= 0 && (best < 0 || halo_ms < best)) { winner = "halo"; best = halo_ms; }
  if (halo_splitk_ms >= 0 && (best < 0 || halo_splitk_ms < best)) {
    winner = "halo_splitk"; best = halo_splitk_ms;
  }

  if (options.coverage) report_coverage<Shape>(std::cerr);

  if (options.csv) {
    std::cout << "case,n,h,w,c,k,r,s,pad_h,pad_w,stride_h,stride_w,dilation_h,"
                 "dilation_w,out_h,out_w,gemm_m,gemm_n,gemm_k,eligible_strip,"
                 "eligible_halo,eligible_splitk,generic_ms,strip_ms,halo_ms,"
                 "halo_stage_k,halo_splitk_ms,halo_splits,halo_splitk_stage_k,"
                 "winner,best_ms,policy_algorithm,policy_config,policy_ms,"
                 "policy_regret,generic_correct,strip_correct,halo_correct,"
                 "halo_splitk_correct,policy_correct\n";
    std::cout << kCaseName << ',' << Shape::n << ',' << Shape::h << ','
              << Shape::w << ',' << Shape::c << ',' << Shape::k << ','
              << Shape::r << ',' << Shape::s << ',' << Shape::pad_h << ','
              << Shape::pad_w << ',' << Shape::stride_h << ','
              << Shape::stride_w << ',' << Shape::dilation_h << ','
              << Shape::dilation_w << ',' << Shape::out_h << ','
              << Shape::out_w << ',' << Shape::gemm_m << ',' << Shape::gemm_n
              << ',' << Shape::gemm_k << ',' << (strip_eligible ? 1 : 0) << ','
              << (halo_eligible ? 1 : 0) << ',' << (splitk_eligible ? 1 : 0)
              << ',' << generic_ms << ','
              << strip_ms << ',' << halo_ms << ',' << halo_stage_k << ','
              << halo_splitk_ms << ',' << halo_splitk_splits << ','
              << halo_splitk_stage_k << ',' << winner << ',' << best << ','
              << algorithm_name(kPolicy.algorithm) << ',';
    if (kPolicy.algorithm == ConvAlgorithm::kGeneric)
      std::cout << kPolicy.cta_m << 'x' << kPolicy.cta_n << '/' << kPolicy.warp_m
                << 'x' << kPolicy.warp_n << "/k" << kPolicy.stage_k;
    else
      std::cout << kPolicy.halo_tile_h << 'x' << kPolicy.halo_tile_w << "/k"
                << kPolicy.halo_stage_k << "/s" << kPolicy.halo_splits;
    std::cout << ',' << policy_ms << ','
              << (policy_ms > 0 && best > 0 ? policy_ms / best : -1.0f) << ','
              << (generic_correct ? 1 : 0) << ',' << (strip_correct ? 1 : 0)
              << ',' << (halo_correct ? 1 : 0) << ','
              << (halo_splitk_correct ? 1 : 0) << ','
              << (policy_correct ? 1 : 0) << '\n';
  } else {
    auto show = [&](const char *label, bool eligible, float milliseconds) {
      std::cout << label << ' ';
      if (!eligible) std::cout << "ineligible";
      else if (milliseconds >= 0)
        std::cout << std::fixed << std::setprecision(4) << milliseconds << " ms";
      else std::cout << "not run";
      std::cout << ", ";
    };
    std::cout << kCaseName << ": ";
    show("generic", true, generic_ms);
    show("strip", strip_eligible, strip_ms);
    show("halo", halo_eligible, halo_ms);
    if (halo_stage_k) std::cout << "(k" << halo_stage_k << ") ";
    std::cout << "splitk";
    if (halo_splitk_ms >= 0)
      std::cout << "x" << halo_splitk_splits << "(k" << halo_splitk_stage_k
                << ") " << std::fixed << std::setprecision(4)
                << halo_splitk_ms << " ms, ";
    else std::cout << " ineligible, ";
    std::cout << "winner " << winner << ", policy "
              << algorithm_name(kPolicy.algorithm) << ' ';
    if (policy_ms >= 0)
      std::cout << std::fixed << std::setprecision(4) << policy_ms << " ms ("
                << (best > 0 ? policy_ms / best : 0) << "x best)";
    else std::cout << "not run";
    std::cout << '\n';
  }
  return generic_correct && strip_correct && halo_correct &&
                 halo_splitk_correct && policy_correct &&
                 std::string(winner) != "none"
             ? 0
             : 1;
}

}  // namespace

int main(int argc, char **argv) {
  return run<TestShape>(parse_options(argc, argv));
}
