#ifndef CAFFE_BASE_CONV_LAYER_CUH_
#define CAFFE_BASE_CONV_LAYER_CUH_

#include <curand_kernel.h>

namespace caffe {

// Returns a random number in (0,1].
// Even though the repetitive initialization of a curand state might look
// suboptimal, the performance is actually nearly the same as when using global
// states.
__device__ __forceinline__ double
RandUniform_device(const int index) {
  curandState state;
  curand_init( (unsigned long long) clock() + index, 0, 0, &state);
  return curand_uniform_double(&state);
}

}  // namespace caffe

#endif  // CAFFE_BASE_CONV_LAYER_CUH_
