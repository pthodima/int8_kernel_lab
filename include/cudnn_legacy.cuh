#pragma once

#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdexcept>

inline void cudnn_lab_check(cudnnStatus_t status, const char *what) {
  if (status != CUDNN_STATUS_SUCCESS) throw std::runtime_error(what);
}

class CudnnLegacyInt8x32 {
 public:
  CudnnLegacyInt8x32() {
    cudnn_lab_check(cudnnCreate(&h_), "cudnnCreate");
    cudnn_lab_check(cudnnCreateTensorDescriptor(&x_), "cudnnCreateTensorDescriptor");
    cudnn_lab_check(cudnnCreateTensorDescriptor(&y_), "cudnnCreateTensorDescriptor");
    cudnn_lab_check(cudnnCreateFilterDescriptor(&w_), "cudnnCreateFilterDescriptor");
    cudnn_lab_check(cudnnCreateConvolutionDescriptor(&c_), "cudnnCreateConvolutionDescriptor");
  }
  ~CudnnLegacyInt8x32() {
    if (workspace_) cudaFree(workspace_);
    if (reordered_) cudaFree(reordered_);
    if (c_) cudnnDestroyConvolutionDescriptor(c_);
    if (w_) cudnnDestroyFilterDescriptor(w_);
    if (y_) cudnnDestroyTensorDescriptor(y_);
    if (x_) cudnnDestroyTensorDescriptor(x_);
    if (h_) cudnnDestroy(h_);
  }
  template <class S> void initialize(const int8_t *x, const int8_t *w, int8_t *y) {
    static_assert(S::c % 32 == 0 && S::k % 32 == 0,
                  "cuDNN INT8x32 baseline requires C and K multiples of 32");
    x_ptr_ = x; y_ptr_ = y;
    cudnn_lab_check(cudnnSetTensor4dDescriptor(x_, CUDNN_TENSOR_NCHW_VECT_C,
      CUDNN_DATA_INT8x32, S::n, S::c, S::h, S::w), "set input");
    cudnn_lab_check(cudnnSetFilter4dDescriptor(w_, CUDNN_DATA_INT8x32,
      CUDNN_TENSOR_NCHW_VECT_C, S::k, S::c, S::r, S::s), "set filter");
    cudnn_lab_check(cudnnSetConvolution2dDescriptor(c_, S::pad_h, S::pad_w,
      S::stride_h, S::stride_w, S::dilation_h, S::dilation_w,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_INT32), "set convolution");
    cudnn_lab_check(cudnnSetConvolutionMathType(c_, CUDNN_TENSOR_OP_MATH), "set math");
    cudnn_lab_check(cudnnSetTensor4dDescriptor(y_, CUDNN_TENSOR_NCHW_VECT_C,
      CUDNN_DATA_INT8x32, S::n, S::k, S::out_h, S::out_w), "set output");
    cudaMalloc(&reordered_, S::filter_elements);
    cudnn_lab_check(cudnnReorderFilterAndBias(h_, w_, CUDNN_DEFAULT_REORDER,
      const_cast<int8_t *>(w), reordered_, 0, nullptr, nullptr), "reorder filter");
    cudnn_lab_check(cudnnSetConvolutionReorderType(c_, CUDNN_NO_REORDER), "set reorder");
    algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    cudnn_lab_check(cudnnGetConvolutionForwardWorkspaceSize(h_, x_, w_, c_, y_, algo_, &bytes_), "workspace size");
    if (bytes_) cudaMalloc(&workspace_, bytes_);
  }
  void run() {
    constexpr float alpha = 1, beta = 0;
    cudnn_lab_check(cudnnConvolutionForward(h_, &alpha, x_, x_ptr_, w_, reordered_, c_,
      algo_, workspace_, bytes_, &beta, y_, y_ptr_), "cudnnConvolutionForward");
  }
 private:
  cudnnHandle_t h_{}; cudnnTensorDescriptor_t x_{}, y_{}; cudnnFilterDescriptor_t w_{};
  cudnnConvolutionDescriptor_t c_{}; cudnnConvolutionFwdAlgo_t algo_{};
  const int8_t *x_ptr_{}; int8_t *y_ptr_{}; void *reordered_{}, *workspace_{}; size_t bytes_{};
};
