#include "cudnn_legacy.cuh"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

template <class S> std::vector<int8_t> pack_x(const std::vector<int8_t>& x) {
  std::vector<int8_t> p(x.size());
  for(int n=0;n<S::n;n++) for(int c=0;c<S::c;c++) for(int h=0;h<S::h;h++) for(int w=0;w<S::w;w++)
    p[(c%32)+32*(w+S::w*(h+S::h*(c/32+S::c/32*n)))] = x[((n*S::h+h)*S::w+w)*S::c+c];
  return p;
}
template <class S> std::vector<int8_t> pack_w(const std::vector<int8_t>& w) {
  std::vector<int8_t> p(w.size());
  for(int k=0;k<S::k;k++) for(int c=0;c<S::c;c++) for(int r=0;r<S::r;r++) for(int s=0;s<S::s;s++)
    p[(c%32)+32*(s+S::s*(r+S::r*(c/32+S::c/32*k)))] = w[((k*S::r+r)*S::s+s)*S::c+c];
  return p;
}
int main(int argc,char**) {
  using S=TestShape; std::mt19937 g(20260925); std::uniform_int_distribution<int> d(-4,4);
  std::vector<int8_t> x(S::input_elements), w(S::filter_elements); for(auto&v:x)v=d(g); for(auto&v:w)v=d(g);
  auto px=pack_x<S>(x), pw=pack_w<S>(w); int8_t *dx,*dw,*dy; cudaMalloc(&dx,px.size()); cudaMalloc(&dw,pw.size()); cudaMalloc(&dy,S::output_elements);
  cudaMemcpy(dx,px.data(),px.size(),cudaMemcpyHostToDevice); cudaMemcpy(dw,pw.data(),pw.size(),cudaMemcpyHostToDevice);
  CudnnLegacyInt8x32 c; c.initialize<S>(dx,dw,dy); for(int i=0;i<20;i++)c.run(); cudaDeviceSynchronize();
  std::vector<float>a; cudaEvent_t b,e; cudaEventCreate(&b);cudaEventCreate(&e);for(int i=0;i<101;i++){cudaEventRecord(b);c.run();cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,b,e);a.push_back(ms);}std::sort(a.begin(),a.end());
  std::printf("case,cudnn_ms\n%s,%.6f\n",kCaseName,a[a.size()/2]); cudaEventDestroy(b);cudaEventDestroy(e);cudaFree(dx);cudaFree(dw);cudaFree(dy);
}
