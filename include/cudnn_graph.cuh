#pragma once

// cuDNN baseline built on the graph (backend) API.
//
// This replaces a legacy CUDNN_DATA_INT8x32 / NCHW_VECT_C /
// IMPLICIT_PRECOMP_GEMM harness.  That enum is CUDNN_DEPRECATED_ENUM in cuDNN
// 9 and has no tensor-core kernel on recent architectures: profiling an sm_120
// run of it showed cuDNN falling back to
// `cnn::conv2d_grouped_direct_kernel<...int...>`, a scalar direct convolution
// taking ~1.34 ms of GPU time for a 14x14 layer -- roughly 100x off a real
// IMMA kernel, and identical for every shape because it was a fixed fallback
// cost rather than compute.  Any ratio measured against it flattered the lab
// by two orders of magnitude.
//
// Two contracts are measured, because they answer different questions:
//
//   kFloatOutput  int8 NHWC x int8 KRSC -> float NHWK, INT32 accumulate.
//                 Four output bytes per element, exactly like conv_lab's
//                 INT32, so write traffic matches.  This is the like-for-like
//                 kernel comparison.
//
//   kInt8Output   the same convolution into a virtual tensor, then a
//                 per-channel scale, then int8 NHWK.  One output byte per
//                 element.  This is the production requantized path, and the
//                 gap between the two columns is what conv_lab's INT32
//                 contract costs -- which is shape dependent, near zero when
//                 the reduction is deep and close to 2x when it is shallow.
//
// cuDNN's graph API reports zero engine configurations for an INT32 output
// tensor, so exact contract parity with conv_lab is not available.
// Neither graph applies a bias; conv_lab does.

#include <cudnn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

enum class CudnnOutput { kFloatOutput, kInt8Output };

struct CudnnGraphResult {
  float milliseconds = -1.0f;
  int engines = 0;
  bool validated = false;
  const char *note = "";
};

namespace cudnn_graph_detail {

inline int64_t kAlignment = 16;

inline bool set_tensor(cudnnBackendDescriptor_t *descriptor, int64_t uid,
                       cudnnDataType_t type, const int64_t dims[4],
                       const int64_t strides[4], bool virtual_tensor) {
  auto ok = [](cudnnStatus_t s) { return s == CUDNN_STATUS_SUCCESS; };
  if (!ok(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, descriptor)))
    return false;
  bool good =
      ok(cudnnBackendSetAttribute(*descriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID,
                                  CUDNN_TYPE_INT64, 1, &uid)) &&
      ok(cudnnBackendSetAttribute(*descriptor, CUDNN_ATTR_TENSOR_DATA_TYPE,
                                  CUDNN_TYPE_DATA_TYPE, 1, &type)) &&
      ok(cudnnBackendSetAttribute(*descriptor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                                  CUDNN_TYPE_INT64, 1, &kAlignment)) &&
      ok(cudnnBackendSetAttribute(*descriptor, CUDNN_ATTR_TENSOR_DIMENSIONS,
                                  CUDNN_TYPE_INT64, 4, dims)) &&
      ok(cudnnBackendSetAttribute(*descriptor, CUDNN_ATTR_TENSOR_STRIDES,
                                  CUDNN_TYPE_INT64, 4, strides));
  if (good && virtual_tensor) {
    bool is_virtual = true;
    good = ok(cudnnBackendSetAttribute(*descriptor, CUDNN_ATTR_TENSOR_IS_VIRTUAL,
                                       CUDNN_TYPE_BOOLEAN, 1, &is_virtual));
  }
  return good && ok(cudnnBackendFinalize(*descriptor));
}

// NHWC-packed strides for a [N, C, H, W] logical shape.
inline void nhwc_strides(int64_t n, int64_t c, int64_t h, int64_t w,
                         int64_t out[4]) {
  (void)n;
  out[0] = h * w * c;
  out[1] = 1;
  out[2] = w * c;
  out[3] = c;
}

template <class Shape>
int32_t reference(const std::vector<int8_t> &input,
                  const std::vector<int8_t> &filter, size_t index) {
  const int channel = index % Shape::k;
  const int m = index / Shape::k;
  const int ox = m % Shape::out_w;
  const int oy = (m / Shape::out_w) % Shape::out_h;
  const int batch = m / (Shape::out_h * Shape::out_w);
  int32_t sum = 0;
  for (int fy = 0; fy < Shape::r; ++fy) {
    const int iy = oy * Shape::stride_h + fy * Shape::dilation_h - Shape::pad_h;
    if (iy < 0 || iy >= Shape::h) continue;
    for (int fx = 0; fx < Shape::s; ++fx) {
      const int ix = ox * Shape::stride_w + fx * Shape::dilation_w - Shape::pad_w;
      if (ix < 0 || ix >= Shape::w) continue;
      for (int c = 0; c < Shape::c; ++c)
        sum += static_cast<int32_t>(
                   input[((batch * Shape::h + iy) * Shape::w + ix) * Shape::c + c]) *
               static_cast<int32_t>(
                   filter[((channel * Shape::r + fy) * Shape::s + fx) * Shape::c + c]);
    }
  }
  return sum;
}

}  // namespace cudnn_graph_detail

// Builds the graph, walks every heuristic engine configuration, validates each
// against the CPU oracle, and returns the median time of the fastest one that
// produced correct results.  An engine that fails validation is never timed,
// so a broken plan cannot win.
template <class Shape>
CudnnGraphResult run_cudnn_graph(CudnnOutput mode,
                                 const std::vector<int8_t> &host_input,
                                 const std::vector<int8_t> &host_filter,
                                 int iterations = 21, int warmup = 10) {
  using namespace cudnn_graph_detail;
  CudnnGraphResult result;
  const float scale = 1.0f / 512.0f;  // keeps a requantized result in range
  const bool int8_out = mode == CudnnOutput::kInt8Output;

  cudnnHandle_t handle = nullptr;
  if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
    result.note = "cudnnCreate failed";
    return result;
  }

  int64_t x_dims[4] = {Shape::n, Shape::c, Shape::h, Shape::w};
  int64_t w_dims[4] = {Shape::k, Shape::c, Shape::r, Shape::s};
  int64_t y_dims[4] = {Shape::n, Shape::k, Shape::out_h, Shape::out_w};
  int64_t x_str[4], w_str[4], y_str[4];
  nhwc_strides(Shape::n, Shape::c, Shape::h, Shape::w, x_str);
  nhwc_strides(Shape::k, Shape::c, Shape::r, Shape::s, w_str);
  nhwc_strides(Shape::n, Shape::k, Shape::out_h, Shape::out_w, y_str);

  cudnnBackendDescriptor_t xd, wd, yd, vd, sd, cd, conv_op, pw_desc, pw_op, graph;
  if (!set_tensor(&xd, 'x', CUDNN_DATA_INT8, x_dims, x_str, false) ||
      !set_tensor(&wd, 'w', CUDNN_DATA_INT8, w_dims, w_str, false) ||
      !set_tensor(&yd, 'y', int8_out ? CUDNN_DATA_INT8 : CUDNN_DATA_FLOAT,
                  y_dims, y_str, false)) {
    result.note = "tensor descriptor rejected";
    return result;
  }
  // The convolution writes into a virtual float tensor when a requantizing
  // pointwise stage follows it, and straight into y otherwise.
  cudnnBackendDescriptor_t conv_dst = yd;
  if (int8_out) {
    if (!set_tensor(&vd, 'v', CUDNN_DATA_FLOAT, y_dims, y_str, true)) {
      result.note = "virtual tensor rejected";
      return result;
    }
    conv_dst = vd;
    int64_t s_dims[4] = {1, Shape::k, 1, 1};
    int64_t s_str[4] = {Shape::k, 1, 1, 1};
    if (!set_tensor(&sd, 's', CUDNN_DATA_FLOAT, s_dims, s_str, false)) {
      result.note = "scale tensor rejected";
      return result;
    }
  }

  int64_t spatial = 2;
  int64_t padding[2] = {Shape::pad_h, Shape::pad_w};
  int64_t dilation[2] = {Shape::dilation_h, Shape::dilation_w};
  int64_t filter_stride[2] = {Shape::stride_h, Shape::stride_w};
  cudnnDataType_t compute = CUDNN_DATA_INT32;
  cudnnConvolutionMode_t conv_mode = CUDNN_CROSS_CORRELATION;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &cd);
  cudnnBackendSetAttribute(cd, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &compute);
  cudnnBackendSetAttribute(cd, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &conv_mode);
  cudnnBackendSetAttribute(cd, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial);
  cudnnBackendSetAttribute(cd, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, padding);
  cudnnBackendSetAttribute(cd, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, padding);
  cudnnBackendSetAttribute(cd, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, dilation);
  cudnnBackendSetAttribute(cd, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, filter_stride);
  if (cudnnBackendFinalize(cd) != CUDNN_STATUS_SUCCESS) {
    result.note = "convolution descriptor rejected";
    return result;
  }

  double alpha = 1.0, beta = 0.0;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &conv_op);
  cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xd);
  cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &wd);
  cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv_dst);
  cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cd);
  cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_DOUBLE, 1, &alpha);
  cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_DOUBLE, 1, &beta);
  if (cudnnBackendFinalize(conv_op) != CUDNN_STATUS_SUCCESS) {
    result.note = "convolution operation rejected";
    return result;
  }

  cudnnBackendDescriptor_t ops[2] = {conv_op, nullptr};
  int64_t op_count = 1;
  if (int8_out) {
    cudnnPointwiseMode_t pointwise = CUDNN_POINTWISE_MUL;
    cudnnDataType_t precision = CUDNN_DATA_FLOAT;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR, &pw_desc);
    cudnnBackendSetAttribute(pw_desc, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &pointwise);
    cudnnBackendSetAttribute(pw_desc, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &precision);
    if (cudnnBackendFinalize(pw_desc) != CUDNN_STATUS_SUCCESS) {
      result.note = "pointwise descriptor rejected";
      return result;
    }
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &pw_op);
    cudnnBackendSetAttribute(pw_op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pw_desc);
    cudnnBackendSetAttribute(pw_op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &vd);
    cudnnBackendSetAttribute(pw_op, CUDNN_ATTR_OPERATION_POINTWISE_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &sd);
    cudnnBackendSetAttribute(pw_op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yd);
    if (cudnnBackendFinalize(pw_op) != CUDNN_STATUS_SUCCESS) {
      result.note = "pointwise operation rejected";
      return result;
    }
    ops[1] = pw_op;
    op_count = 2;
  }

  cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &graph);
  cudnnBackendSetAttribute(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
  cudnnBackendSetAttribute(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, op_count, ops);
  if (cudnnBackendFinalize(graph) != CUDNN_STATUS_SUCCESS) {
    result.note = "operation graph rejected";
    return result;
  }

  cudnnBackendDescriptor_t heuristics;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heuristics);
  cudnnBackendSetAttribute(heuristics, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph);
  cudnnBackendHeurMode_t heuristic_mode = CUDNN_HEUR_MODE_A;
  cudnnBackendSetAttribute(heuristics, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &heuristic_mode);
  if (cudnnBackendFinalize(heuristics) != CUDNN_STATUS_SUCCESS) {
    result.note = "heuristics rejected";
    return result;
  }

  constexpr int kMaxConfigs = 12;
  cudnnBackendDescriptor_t configs[kMaxConfigs];
  for (int i = 0; i < kMaxConfigs; ++i)
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &configs[i]);
  int64_t returned = 0;
  cudnnBackendGetAttribute(heuristics, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                           CUDNN_TYPE_BACKEND_DESCRIPTOR, kMaxConfigs, &returned, configs);
  result.engines = static_cast<int>(returned);
  if (returned == 0) {
    result.note = "no engine supports this shape and contract";
    return result;
  }

  const size_t output_bytes =
      static_cast<size_t>(Shape::output_elements) * (int8_out ? 1 : 4);
  int8_t *d_input = nullptr, *d_filter = nullptr;
  void *d_output = nullptr;
  float *d_scale = nullptr;
  cudaMalloc(&d_input, host_input.size());
  cudaMalloc(&d_filter, host_filter.size());
  cudaMalloc(&d_output, output_bytes);
  cudaMemcpy(d_input, host_input.data(), host_input.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter, host_filter.data(), host_filter.size(), cudaMemcpyHostToDevice);
  if (int8_out) {
    std::vector<float> host_scale(Shape::k, scale);
    cudaMalloc(&d_scale, Shape::k * sizeof(float));
    cudaMemcpy(d_scale, host_scale.data(), Shape::k * sizeof(float), cudaMemcpyHostToDevice);
  }

  std::vector<char> host_output(output_bytes);
  float best = -1.0f;
  for (int i = 0; i < returned; ++i) {
    cudnnBackendDescriptor_t plan = nullptr, pack = nullptr;
    void *workspace = nullptr;
    if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan) != CUDNN_STATUS_SUCCESS)
      continue;
    cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
    cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &configs[i]);
    if (cudnnBackendFinalize(plan) != CUDNN_STATUS_SUCCESS) {
      cudnnBackendDestroyDescriptor(plan);
      continue;
    }
    int64_t workspace_bytes = 0, one = 0;
    cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &one, &workspace_bytes);
    if (workspace_bytes > 0 && cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) {
      cudnnBackendDestroyDescriptor(plan);
      continue;
    }
    void *pointers[4] = {d_input, d_filter, d_output, d_scale};
    int64_t uids[4] = {'x', 'w', 'y', 's'};
    const int64_t pointer_count = int8_out ? 4 : 3;
    if (int8_out) {
      pointers[2] = d_scale;  pointers[3] = d_output;
      uids[2] = 's';          uids[3] = 'y';
    }
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &pack);
    cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, pointer_count, pointers);
    cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, pointer_count, uids);
    cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace);
    if (cudnnBackendFinalize(pack) == CUDNN_STATUS_SUCCESS &&
        cudnnBackendExecute(handle, plan, pack) == CUDNN_STATUS_SUCCESS) {
      cudaDeviceSynchronize();
      cudaMemcpy(host_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost);
      const size_t elements = Shape::output_elements;
      size_t stride = static_cast<size_t>(elements * 0.6180339887498949) | 1;
      size_t index = 0;
      bool correct = true;
      for (int probe = 0; probe < 512 && correct; ++probe) {
        const int32_t exact = reference<Shape>(host_input, host_filter, index);
        if (int8_out) {
          int expected = static_cast<int>(std::lrintf(exact * scale));
          expected = std::max(-128, std::min(127, expected));
          // One unit of slack: the rounding mode of the requantize stage is
          // cuDNN's to choose, and it is not part of what is being measured.
          if (std::abs(static_cast<int>(reinterpret_cast<int8_t *>(host_output.data())[index]) -
                       expected) > 1)
            correct = false;
        } else if (reinterpret_cast<float *>(host_output.data())[index] !=
                   static_cast<float>(exact)) {
          correct = false;
        }
        index = (index + stride) % elements;
      }
      if (correct) {
        for (int t = 0; t < warmup; ++t) cudnnBackendExecute(handle, plan, pack);
        cudaDeviceSynchronize();
        std::vector<float> samples;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        for (int t = 0; t < iterations; ++t) {
          cudaEventRecord(start);
          cudnnBackendExecute(handle, plan, pack);
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          float ms = 0;
          cudaEventElapsedTime(&ms, start, stop);
          samples.push_back(ms);
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        std::sort(samples.begin(), samples.end());
        const float median = samples[samples.size() / 2];
        if (best < 0 || median < best) best = median;
        result.validated = true;
      }
    }
    if (pack) cudnnBackendDestroyDescriptor(pack);
    cudnnBackendDestroyDescriptor(plan);
    if (workspace) cudaFree(workspace);
  }

  cudaFree(d_input);
  cudaFree(d_filter);
  cudaFree(d_output);
  if (d_scale) cudaFree(d_scale);
  cudnnDestroy(handle);
  result.milliseconds = best;
  if (!result.validated) result.note = "no engine produced correct results";
  return result;
}
