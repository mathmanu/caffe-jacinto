#include <algorithm>
#include <device_launch_parameters.h>

#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_math_functions.cuh"


#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/count.h>

#include <cuda_fp16.h>

namespace caffe {


template<typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template<>
void caffe_gpu_mul<float>(const int N, const float* a, const float* b, float* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_mul<double>(const int N, const double* a, const double* b, double* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_mul<float16>(const int N, const float16* a, const float16* b, float16* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}




template<typename Dtype>
__global__ void square_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * a[index];
  }
}

template<>
void caffe_gpu_square<float>(const int N, const float* a, float* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  square_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, a, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_square<double>(const int N, const double* a, double* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  square_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, a, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_square<float16>(const int N, const float16* a, float16* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  square_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, a, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template<>
void caffe_gpu_div<float>(const int N, const float* a, const float* b, float* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_div<double>(const int N, const double* a, const double* b, double* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_div<float16>(const int N, const float16* a, const float16* b, float16* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template<>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, a, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, a, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_abs<float16>(const int N, const float16* a, float16* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, a, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template<>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, a, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double> <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, a, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_exp<float16>(const int N, const float16* a, float16* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, a, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template<>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, a, y);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, a, y);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_log<float16>(const int N, const float16* a, float16* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, a, y);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template<>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, a, alpha, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, a, alpha, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_powx<float16>(const int N, const float16* a,
    const float16 alpha, float16* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, a, alpha, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign,
    y[index] = (Dtype(0) < x[index]) - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

__global__ void popc_kernel(const int n, const float* a,
    const float* b, uint8_t* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = __popc(static_cast<uint32_t>(a[index]) ^
                      static_cast<uint32_t>(b[index]));
  }
}

__global__ void popcll_kernel(const int n, const double* a,
    const double* b, uint8_t* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = __popcll(static_cast<uint64_t>(a[index]) ^
                        static_cast<uint64_t>(b[index]));
  }
}

__global__ void popch_kernel(const int n, const half* a,
    const half* b, uint8_t* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = __popc(a[index].x() ^ b[index].x());
  }
}

template<typename T, typename TR>
__global__
void convert_kernel(const unsigned int n, const T* in, TR* out) {
  CUDA_KERNEL_LOOP(i, n) {
    out[i] = in[i];
  }
}
template<>
__global__
void convert_kernel(const unsigned int n, const half2* in, float2* out) {
  CUDA_KERNEL_LOOP(i, n) {
    out[i] = __half22float2(in[i]);
  }
}
template<>
__global__
void convert_kernel(const unsigned int n, const float2* in, half2* out) {
  CUDA_KERNEL_LOOP(i, n) {
    out[i] = float22half2_clip(in[i]);
  }
}

template<typename T, typename TR>
void caffe_gpu_convert(const unsigned int N, const T* in, TR* out) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  convert_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, in, out);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_convert<float, float16>(const unsigned int n,
    const float* in, float16* out) {
  cudaStream_t stream = Caffe::thread_stream();
  const unsigned int n2 = even(n) / 2;
  // NOLINT_NEXT_LINE(whitespace/operators)
  convert_kernel<<<CAFFE_GET_BLOCKS_HALF(n2), CAFFE_CUDA_NUM_THREADS_HALF, 0, stream>>>
      (n2, reinterpret_cast<const float2*>(in), reinterpret_cast<half2*>(out));
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_convert<float16, float>(const unsigned int n,
    const float16* in, float* out) {
  cudaStream_t stream = Caffe::thread_stream();
  const unsigned int n2 = even(n) / 2;
  // NOLINT_NEXT_LINE(whitespace/operators)
  convert_kernel<<<CAFFE_GET_BLOCKS_HALF(n2), CAFFE_CUDA_NUM_THREADS_HALF, 0, stream>>>
      (n2, reinterpret_cast<const half2*>(in), reinterpret_cast<float2*>(out));
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template void caffe_gpu_convert<double, float16>(const unsigned int n,
    const double* in, float16* out);
template void caffe_gpu_convert<float16, double>(const unsigned int n,
    const float16* in, double* out);
template void caffe_gpu_convert<double, float>(const unsigned int n,
    const double* in, float* out);
template void caffe_gpu_convert<float, double>(const unsigned int n,
    const float* in, double* out);
template<>
void caffe_gpu_convert<float, float>(const unsigned int n,
    const float* in, float* out) {
  caffe_copy(n, in, out);
}
template<>
void caffe_gpu_convert<double, double>(const unsigned int n,
    const double* in, double* out) {
  caffe_copy(n, in, out);
}
template<>
void caffe_gpu_convert<float16, float16>(const unsigned int n,
    const float16* in, float16* out) {
  caffe_copy(n, in, out);
}

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::curand_stream()));
}
template<>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
    float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::curand_stream()));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template<>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
    double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::curand_stream()));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template<>
void caffe_gpu_rng_uniform<float16>(const int n, const float16 a,
    const float16 b, float16* r) {
  GPUMemory::Workspace rf(n * sizeof(float), Caffe::current_device());
  float* rfp = static_cast<float*>(rf.data());
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), rfp, n));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::curand_stream()));
  const float range = b - a;
  if (range != 1.F) {
    caffe_gpu_scal(n, range, rfp);
  }
  if (a != static_cast<float16>(0)) {
    caffe_gpu_add_scalar(n, static_cast<float>(a), rfp);
  }
  caffe_gpu_convert(n, rfp, r);
}

template<>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma, float* r) {
  CURAND_CHECK(curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::curand_stream()));
}

template<>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma, double* r) {
  CURAND_CHECK(curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::curand_stream()));
}

template<>
void caffe_gpu_rng_gaussian(const int n, const float16 mu, const float16 sigma, float16* r) {
  GPUMemory::Workspace rf(n * sizeof(float), Caffe::current_device());
  float* rfp = static_cast<float*>(rf.data());
  CURAND_CHECK(curandGenerateNormal(Caffe::curand_generator(), rfp, n, mu, sigma));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::curand_stream()));
  caffe_gpu_convert(n, rfp, r);
}

template<typename Dtype>
__global__ void caffe_gpu_eltwise_max_kernel(const int N, const Dtype alpha, const Dtype* x,
    const Dtype beta, Dtype* y) {
  CUDA_KERNEL_LOOP(index, N) {
    y[index] = max(alpha * x[index], beta * y[index]);
  }
}

template<>
void caffe_gpu_eltwise_max<float>(const int N, const float alpha, const float* x,
    const float beta, float* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_eltwise_max_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>
      (N, alpha, x, beta, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_eltwise_max<double>(const int N,
    const double alpha, const double* x, const double beta, double* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_eltwise_max_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>
      (N, alpha, x, beta, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_eltwise_max<float16>(const int N,
    const float16 alpha, const float16* x, const float16 beta, float16* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_eltwise_max_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>
      (N, alpha, x, beta, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename Dtype>
__global__ void caffe_gpu_eltwise_min_kernel(const int N,
    const Dtype alpha, const Dtype* x, const Dtype beta, Dtype* y) {
  CUDA_KERNEL_LOOP(index, N) {
    y[index] = min(alpha * x[index], beta * y[index]);
  }
}

template<>
void caffe_gpu_eltwise_min<float>(const int N,
    const float alpha, const float* x, const float beta, float* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_eltwise_min_kernel<float> <<<CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, alpha, x, beta, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}
template<>
void caffe_gpu_eltwise_min<double>(const int N,
    const double alpha, const double* x, const double beta, double* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_eltwise_min_kernel<double> <<<CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, alpha, x, beta, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}
template<>
void caffe_gpu_eltwise_min<float16>(const int N,
    const float16 alpha, const float16* x, const float16 beta, float16* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_eltwise_min_kernel<float16> <<<CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, alpha, x, beta, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}


DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(if_zero, y[index] = ((x[index] <= Dtype(0) && x[index] >= Dtype(-0) ) ? 1 : 0) );
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(if_nonzero, y[index] = ((x[index] > Dtype(0) || x[index] < Dtype(-0) ) ? 1 : 0) )
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(eltwise_multi, y[index] = y[index]*x[index] )

DEFINE_AND_INSTANTIATE_GPU_1NARY_FUNC(set, x[index] = value, value )

DEFINE_AND_INSTANTIATE_GPU_X2NARY_FUNC(zerout, y[index] = ((x[index] <= Dtype(threshold) && x[index] >= Dtype(-threshold)) ? Dtype(0) : x[index]), threshold )

template <typename Dtype>
struct CheckZeroFunctor {
  CheckZeroFunctor(const Dtype threshold) : threshold(threshold) {
  }
  __host__ __device__ bool operator()(const Dtype& x) {
    return (x<=(threshold) && x>=(-threshold));
  }
  const Dtype threshold;
};

//use an int return type always.
template <typename Dtype>
int caffe_gpu_count_zero(const int N, const Dtype* x, Dtype threshold) {
  CheckZeroFunctor<Dtype> check_zero(threshold);
  thrust::device_ptr<const Dtype> pWrapper(x);
  int count = thrust::count_if(pWrapper, pWrapper+N, check_zero);
  return count;
}


template
int caffe_gpu_count_zero<float>(const int N, const float* x, float threshold);
template
int caffe_gpu_count_zero<double>(const int N, const double* x, double threshold);
#if 0
template
int caffe_gpu_count_zero<float16>(const int N, const float16* x, float16 threshold);
#else
template<>
int caffe_gpu_count_zero<float16>(const int N, const float16* x, float16 threshold) {
  CHECK(false) << "caffe_gpu_count_zero<float16>() is disabled due to compilation error";
  return 0;
}
#endif

struct transform_op_float16_to_float {
  __host__ __device__ float operator()(const float16 v1) {
    return float(v1);
  }  
};

template <typename Dtype>
Dtype caffe_gpu_min(const int N, const Dtype* x) {
  thrust::device_ptr<const Dtype> pWrapper(x);
  const thrust::device_ptr<const Dtype> min_val = thrust::min_element(pWrapper, pWrapper+N);
  return *min_val;
}
template
float caffe_gpu_min(const int N, const float* x);
template
double caffe_gpu_min(const int N, const double* x);
#if 0
//Avoid float16 errors, probably because thrust::greater(), thrust::lesser() 
//are not available for float16: provide a different specializer here
template<>
float16 caffe_gpu_min(const int N, const float16* x){
  thrust::device_ptr<const float16> pWrapper(x);
  float min_val = thrust::transform_reduce(pWrapper, pWrapper+N, transform_op_float16_to_float(), 
    0, thrust::less<float>());
  return min_val;
}
#else
template<>
float16 caffe_gpu_min(const int N, const float16* x){
  CHECK(false) << "caffe_gpu_min<float16>() is disabled due to compilation error";
  return 0;
}
#endif

template <typename Dtype>
Dtype caffe_gpu_max(const int N, const Dtype* x) {
  thrust::device_ptr<const Dtype> pWrapper(x);
  const thrust::device_ptr<const Dtype> max_val = thrust::max_element(pWrapper, pWrapper+N);
  return *max_val;
}
template
float caffe_gpu_max(const int N, const float* x);
template
double caffe_gpu_max(const int N, const double* x);
#if 0
//Avoid float16 errors, probably because thrust::greater(), thrust::lesser() 
//are not available for float16: provide a different specializer here
template<> 
float16 caffe_gpu_max(const int N, const float16* x) {
  thrust::device_ptr<const float16> pWrapper(x);
   thrust::transform_reduce(pWrapper, pWrapper+N, transform_op_float16_to_float(), 
    0., thrust::greater<float>());
  return 0;
}
#else
template<>
float16 caffe_gpu_max(const int N, const float16* x){
  CHECK(false) << "caffe_gpu_max<float16>() is disabled due to compilation error";
  return 0;
}
#endif

}  // namespace caffe
