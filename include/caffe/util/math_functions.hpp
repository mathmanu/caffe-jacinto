#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <tuple>

#include <glog/logging.h>

#include "caffe/common.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

#define FP16_MAX_THR 65000.F

namespace caffe {

class Blob;

inline
bool is_pow2(unsigned int x) {
  return ((x & (x - 1)) == 0);
}

template <typename T>
void clean_last_element(T* x, cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(x, 0, sizeof(T), stream));
//  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Caffe gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_cpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

// y[i] = max(a * x[i], b * y[i])
template <typename Dtype>
void caffe_cpu_eltwise_max(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

// y[i] = min(a * x[i], b * y[i])
template <typename Dtype>
void caffe_cpu_eltwise_min(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

template <typename Dtype>
void caffe_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype *X);

inline void caffe_memset(const size_t N, const int alpha, void* X) {
  memset(X, alpha, N);  // NOLINT(caffe/alt_fn)
}

template <typename Dtype>
void caffe_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_sqr(const int N, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

unsigned int caffe_rng_rand();

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b);

template <typename Ftype>
void caffe_rng_uniform(int n, Ftype a, Ftype b, Blob* blob);

template <typename Ftype>
void caffe_rng_uniform(int n, Ftype a, Ftype b, Ftype* r);

template <typename Ftype>
void caffe_rng_gaussian(int n, Ftype a, Ftype b, Blob* blob);

template <typename Ftype>
void caffe_rng_gaussian(int n, Ftype a, Ftype b, Ftype* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r);

template <typename Dtype>
void caffe_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_log(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_abs(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y);

template <typename Dtype>
Dtype caffe_cpu_strided_dot(const int n, const Dtype* x, const int incx,
    const Dtype* y, const int incy);

template <typename Dtype>
int caffe_cpu_hamming_distance(const int n, const Dtype* x, const Dtype* y);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
float caffe_cpu_asum(const int n, const Dtype* x);

template <typename Dtype>
float caffe_cpu_sumsq(const int n, const Dtype* x);

template <typename Dtype>
Dtype caffe_cpu_amax(const int n, const Dtype* x);

// the branchless, type-safe version from
// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template<typename Dtype>
inline int8_t caffe_sign(Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

// The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
//   in include/caffe/util/mkl_alternate.hpp authored by @Rowland Depp.
// Please refer to commit 7e8ef25c7 of the boost-eigen branch.
// Git cherry picking that commit caused a conflict hard to resolve and
//   copying that file in convenient for code reviewing.
// So they have to be pasted here temporarily.
#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void caffe_cpu_##name(const int n, const Dtype* x, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(x); CHECK(y); \
    for (int i = 0; i < n; ++i) { \
      operation; \
    } \
  }
#define DEFINE_CAFFE_CPU_2NARY_FUNC(name, operation, threshold) \
  template<typename Dtype> \
  void caffe_cpu_##name(const int n, const Dtype* x, Dtype* y, Dtype threshold) { \
    CHECK_GT(n, 0); CHECK(x); CHECK(y); \
    for (int i = 0; i < n; ++i) { \
      operation; \
    } \
  }
  
// output is 1 for the positives, 0 for zero, and -1 for the negatives
DEFINE_CAFFE_CPU_UNARY_FUNC(sign,          y[i] = caffe_sign<Dtype>(x[i]));
DEFINE_CAFFE_CPU_UNARY_FUNC(if_zero,       y[i] = ((x[i] <= Dtype(0) && x[i] >= Dtype(-0) ) ? 1 : 0) );
DEFINE_CAFFE_CPU_UNARY_FUNC(if_nonzero,    y[i] = ((x[i] > Dtype(0) || x[i] < Dtype(-0) ) ? 1 : 0) );
DEFINE_CAFFE_CPU_UNARY_FUNC(eltwise_multi, y[i] = (y[i]*x[i]) );

DEFINE_CAFFE_CPU_2NARY_FUNC(zerout,        y[i] = ((x[i] <= Dtype(threshold) && x[i] >= Dtype(-threshold)) ? Dtype(0) : x[i]), threshold );

// This returns a nonzero value if the input has its sign bit set.
// The name sngbit is meant to avoid conflicts with std::signbit in the macro.
// The extra parens are needed because CUDA < 6.5 defines signbit as a macro,
// and we don't want that to expand here when CUDA headers are also included.
DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit, \
    y[i] = static_cast<bool>((std::signbit)(x[i])));

DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]));

template <typename Dtype>
void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y, void* handle = nullptr);

template <typename Dtype>
void caffe_gpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y, int group = 0);

template <typename Dtype>
void caffe_gpu_set(const size_t N, const Dtype alpha, Dtype *X);

inline void caffe_gpu_memset(const size_t N, const int alpha, void* X, int group = 0) {
  cudaStream_t stream = Caffe::thread_stream(group);
  CUDA_CHECK_ARG2(cudaMemsetAsync(X, alpha, N, stream),
      stream, Caffe::current_device());  // NOLINT(caffe/alt_fn)
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename Dtype>
void caffe_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype* X);

template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype* X, cublasHandle_t cublas_handle);

template <typename Dtype>
void caffe_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_incr(const int N, const Dtype* a, Dtype* b);

template <typename Dtype>
void caffe_gpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_square(const int N, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_abs(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_log(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

// caffe_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].
void caffe_gpu_rng_uniform(const int n, unsigned int* r);

// caffe_gpu_rng_uniform with four arguments generates floats in the range
// (a, b] (strictly greater than a, less than or equal to b) due to the
// specification of curandGenerateUniform.  With a = 0, b = 1, just calls
// curandGenerateUniform; with other limits will shift and scale the outputs
// appropriately after calling curandGenerateUniform.
template <typename Dtype>
void caffe_gpu_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void caffe_gpu_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                            Dtype* r);

template <typename Dtype>
void caffe_gpu_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype, typename Mtype>
void caffe_gpu_dot(const int n, const Dtype* x, const Dtype* y, Mtype* out);

//template <typename Dtype>
//uint32_t caffe_gpu_hamming_distance(const int n, const Dtype* x,
//                                    const Dtype* y);

// TODO group
template <typename Dtype, typename Mtype>
void caffe_gpu_asum(const int n, const Dtype* x, Mtype* y, int group);

template <typename Dtype, typename Mtype>
void caffe_gpu_sumsq(const int n, const Dtype* x, Mtype* s, int group);

template <typename Dtype>
void caffe_gpu_amax(const int n, const Dtype* x, float* y, int group);

template<typename Dtype>
void caffe_gpu_sign(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_sgnbit(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_fabs(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

template <typename T_IN, typename T_OUT>
void caffe_gpu_convert(const unsigned int n, const T_IN* in, T_OUT* out);

template <typename Dtype>
float caffe_gpu_max_norm1(const int n, const int m, const Dtype* x);


// y[i] = max(a * x[i], b * y[i])
template <typename Dtype>
void caffe_gpu_eltwise_max(const int n, const Dtype alpha, const Dtype* x,
    const Dtype beta, Dtype* y);

// y[i] = min(a * x[i], b * y[i])
template <typename Dtype>
void caffe_gpu_eltwise_min(const int n, const Dtype alpha, const Dtype* x,
    const Dtype beta, Dtype* y);

template<typename Dtype>
void caffe_gpu_eltwise_multi(const int n, const Dtype* x, Dtype* y);

//float16 is too small range for count. so use a int return type always.
template <typename Dtype>
int caffe_gpu_count_zero(const int N, const Dtype* x, Dtype threshold);

template <typename Dtype>
Dtype caffe_gpu_min(const int N, const Dtype* x);

template <typename Dtype>
Dtype caffe_gpu_max(const int N, const Dtype* x);

template<typename Dtype>
void caffe_gpu_if_nonzero(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_if_zero(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_set(const int N, Dtype* x, Dtype value);

template<typename Dtype>
void caffe_gpu_zerout(const int n, const Dtype* x, Dtype* y, Dtype threshold);

#define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(name, operation) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
  CUDA_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void caffe_gpu_##name<float>(const int n, const float* x, float* y) { \
  cudaStream_t stream = Caffe::thread_stream(); \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(n, x, y); \
  CUDA_CHECK(cudaStreamSynchronize(stream)); \
} \
template <> \
void caffe_gpu_##name<double>(const int n, const double* x, double* y) { \
  cudaStream_t stream = Caffe::thread_stream(); \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(n, x, y); \
  CUDA_CHECK(cudaStreamSynchronize(stream)); \
} \
template <> \
void caffe_gpu_##name<float16>(const int n, const float16* x, float16* y) { \
  cudaStream_t stream = Caffe::thread_stream(); \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float16><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(n, x, y); \
  CUDA_CHECK(cudaStreamSynchronize(stream)); \
}


#define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC_AUX(name, operation) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
  CUDA_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void caffe_gpu_##name<float>(const int n, const float* x, float* y, void* handle) { \
  cublasHandle_t cublas_handle = \
      handle == nullptr ? Caffe::cublas_handle() : reinterpret_cast<cublasHandle_t>(handle); \
  cudaStream_t stream; \
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream)); \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(n, x, y); \
  CUDA_CHECK(cudaStreamSynchronize(stream)); \
} \
template <> \
void caffe_gpu_##name<double>(const int n, const double* x, double* y, void* handle) { \
  cublasHandle_t cublas_handle = \
      handle == nullptr ? Caffe::cublas_handle() : reinterpret_cast<cublasHandle_t>(handle); \
  cudaStream_t stream; \
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream)); \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(n, x, y); \
  CUDA_CHECK(cudaStreamSynchronize(stream)); \
} \
template <> \
void caffe_gpu_##name<float16>(const int n, const float16* x, float16* y, void* handle) { \
  cublasHandle_t cublas_handle = \
      handle == nullptr ? Caffe::cublas_handle() : reinterpret_cast<cublasHandle_t>(handle); \
  cudaStream_t stream; \
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream)); \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float16><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(n, x, y); \
  CUDA_CHECK(cudaStreamSynchronize(stream)); \
}

#define DEFINE_AND_INSTANTIATE_GPU_1NARY_FUNC(name, operation, value) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, Dtype* x, Dtype value) { \
  CUDA_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void caffe_gpu_##name<float>(const int n, float* x, float value) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, \
    Caffe::thread_stream()>>>\
      (n, x, value); \
} \
template <> \
void caffe_gpu_##name<double>(const int n, double* x, double value) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, \
    Caffe::thread_stream()>>>\
      (n, x, value); \
} \
template <> \
void caffe_gpu_##name<float16>(const int n, float16* x, float16 value) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float16><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, \
    Caffe::thread_stream()>>>\
      (n, x, value); \
}

#define DEFINE_AND_INSTANTIATE_GPU_X2NARY_FUNC(name, operation, threshold) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y, Dtype threshold) { \
  CUDA_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void caffe_gpu_##name<float>(const int n, const float *x, float *y, float threshold) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, \
    Caffe::thread_stream()>>>\
      (n, x, y, threshold); \
} \
template <> \
void caffe_gpu_##name<double>(const int n, const double *x, double *y, double threshold) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, \
    Caffe::thread_stream()>>>\
      (n, x, y, threshold); \
} \
template <> \
void caffe_gpu_##name<float16>(const int n, const float16 *x, float16 *y, float16 threshold) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float16><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, \
    Caffe::thread_stream()>>>\
      (n, x, y, threshold); \
}


template <typename T_IN, typename T_OUT>
inline void caffe_cpu_convert(const int n, const T_IN *in, T_OUT *out) {
  for (int i = 0; i < n; ++i) {
    out[i] = static_cast<T_OUT>(in[i]);
  }
}

template <typename T_IN, typename T_OUT>
inline void caffe_convert(bool use_gpu, const int n, const T_IN* in, T_OUT* out) {
  if (use_gpu) {
    caffe_gpu_convert(n, in, out);
  } else {
    caffe_cpu_convert(n, in, out);
  }
}

template <typename Dtype> Dtype caffe_cpu_max(const int n, const Dtype* x);
template <typename Dtype> Dtype caffe_cpu_min(const int n, const Dtype* x);
template <typename Dtype> Dtype caffe_cpu_count_zero(const int N, const Dtype* x, Dtype threshold);

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
