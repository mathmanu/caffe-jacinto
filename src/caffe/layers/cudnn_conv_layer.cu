#ifdef USE_CUDNN
#include <algorithm>

#include "caffe/filler.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/solver.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void CuDNNConvolutionLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  //this->Quantize_gpu(bottom, top);
  const Ftype* weight = this->blobs_[0]->template gpu_data<Ftype>();
  shared_ptr<GPUMemory::Workspace>& ws = GPUMemory::workspace_[Caffe::current_device()];
  if (use_v7grouping()) {
    for (int i = 0; i < bottom.size(); ++i) {
      const Ftype *bottom_data = bottom[i]->gpu_data<Ftype>();
      Ftype *top_data = top[i]->mutable_gpu_data<Ftype>();
      // Forward through cuDNN in parallel over groups.
      CUDNN_CHECK(cudnnConvolutionForward(Caffe::cudnn_handle(0),
          cudnn::dataType<Ftype>::one, fwd_bottom_descs_[i], bottom_data,
          fwd_filter_desc_, weight,
          fwd_conv_descs_[i], fwd_algo_[i], ws->data(), ws->size(),
          cudnn::dataType<Ftype>::zero, fwd_top_descs_[i], top_data));
      if (this->bias_term_) {
        const Ftype *bias_data = this->blobs_[1]->template gpu_data<Ftype>();
        CUDNN_CHECK(cudnnAddTensor(Caffe::cudnn_handle(0),
            cudnn::dataType<Ftype>::one,
            fwd_bias_desc_, bias_data,
            cudnn::dataType<Ftype>::one,
            fwd_top_descs_[i], top_data));
      }
      CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(0)));
    }  // end of for i
  } else {
    // "old" path
    for (int i = 0; i < bottom.size(); ++i) {
      const Ftype* bottom_data = bottom[i]->gpu_data<Ftype>();
      Ftype* top_data = top[i]->mutable_gpu_data<Ftype>();
      // Forward through cuDNN in parallel over groups.
      const size_t gsize = ws->size() / ws_groups();
      CHECK(is_even(gsize));
      for (int g = 0; g < groups(); ++g) {
        void* pspace = static_cast<unsigned char*>(ws->data()) + gsize * idxg(g);
        // Filters.
        CUDNN_CHECK(cudnnConvolutionForward(Caffe::cudnn_handle(idxg(g)),
            cudnn::dataType<Ftype>::one, fwd_bottom_descs_[i], bottom_data + bottom_offset_ * g,
            fwd_filter_desc_, weight + this->weight_offset_ * g,
            fwd_conv_descs_[i], fwd_algo_[i], pspace, gsize,
            cudnn::dataType<Ftype>::zero, fwd_top_descs_[i], top_data + top_offset_ * g));
      }
      // NOLINT_NEXT_LINE(whitespace/operators)
      for (int ig = 0; ig < ws_groups(); ++ig) {
        CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(ig)));
      }

      if (this->bias_term_) {
        const Ftype* bias_data = this->blobs_[1]->template gpu_data<Ftype>();
        for (int g = 0; g < groups(); ++g) {
          CUDNN_CHECK(cudnnAddTensor(Caffe::cudnn_handle(idxg(g)),
              cudnn::dataType<Ftype>::one,
              fwd_bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Ftype>::one,
              fwd_top_descs_[i], top_data + top_offset_ * g));
        }
        // Synchronize the work across groups, each of which went into its own stream
        // NOLINT_NEXT_LINE(whitespace/operators)
        for (int g = 0; g < ws_groups(); ++g) {
          CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(g)));
        }
      }
    }  // end of for i
  }

  ++fwd_count_;
  
  this->Quantize_gpu(bottom, top);    
}

template <typename Ftype, typename Btype>
void CuDNNConvolutionLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  propagate_down_ = propagate_down;
  shared_ptr<GPUMemory::Workspace>& ws = GPUMemory::workspace_[Caffe::current_device()];
  if (use_v7grouping()) {
    // compute dE/dB = sum_c(dE/dy)
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Btype *bias_diff = this->blobs_[1]->template mutable_gpu_diff<Btype>();
      for (int i = 0; i < top.size(); ++i) {
        Btype *top_diff = top[i]->mutable_gpu_diff<Btype>();
        // in parallel over groups
        CUDNN_CHECK(cudnnConvolutionBackwardBias(Caffe::cudnn_handle(0),
            cudnn::dataType<Btype>::one, bwd_top_descs_[i], top_diff,
            cudnn::dataType<Btype>::one, bwd_bias_desc_, bias_diff));
        CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(0)));
      }  // end of i
    }  // end of dB

    // compute dE/dW = dY * X
    if (this->param_propagate_down_[0]) {
      Btype *weight_diff = this->blobs_[0]->template mutable_gpu_diff<Btype>();
      for (int i = 0; i < top.size(); ++i) {
        Btype *top_diff = top[i]->mutable_gpu_diff<Btype>();
        const Btype *bottom_data = bottom[i]->gpu_data<Btype>();
        // Gradient w.r.t. weights.
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(Caffe::cudnn_handle(0),
            cudnn::dataType<Btype>::one, bwd_bottom_descs_[i], bottom_data,
            bwd_top_descs_[i], top_diff,
            bwd_conv_filter_descs_[i], bwd_filter_algo_[i], ws->data(), ws->size(),
            cudnn::dataType<Btype>::one, bwd_filter_desc_, weight_diff));
        CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(0)));
      }  // end of i
    }

    // Backward propagate grad wrt bottom data dE/dX= dE/dY * W
    const Btype *weight = this->blobs_[0]->template gpu_data<Btype>();
    for (int i = 0; i < top.size(); ++i) {
      if (propagate_down[i]) {
        Btype *top_diff = top[i]->mutable_gpu_diff<Btype>();
        Btype *bottom_diff = bottom[i]->mutable_gpu_diff<Btype>();
        CUDNN_CHECK(cudnnConvolutionBackwardData(Caffe::cudnn_handle(0),
            cudnn::dataType<Btype>::one, bwd_filter_desc_, weight,
            bwd_top_descs_[i], top_diff,
            bwd_conv_data_descs_[i],
            bwd_data_algo_[i], ws->data(), ws->size(),
            cudnn::dataType<Btype>::zero, bwd_bottom_descs_[i], bottom_diff));
        CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(0)));
      }  // end if propagate down
    }  // end for i
  } else {
    // "old" path
    const size_t gsize = ws->size() / ws_groups();
    // compute dE/dB = sum_c(dE/dy)
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Btype* bias_diff = this->blobs_[1]->template mutable_gpu_diff<Btype>();
      for (int i = 0; i < top.size(); ++i) {
        Btype* top_diff = top[i]->mutable_gpu_diff<Btype>();
        // in parallel over groups
        for (int g = 0; g < groups(); ++g) {
          CUDNN_CHECK(cudnnConvolutionBackwardBias(Caffe::cudnn_handle(idxg(g)),
              cudnn::dataType<Btype>::one, bwd_top_descs_[i], top_diff + top_offset_ * g,
              cudnn::dataType<Btype>::one, bwd_bias_desc_, bias_diff + bias_offset_ * g));
        }  // end of groups
        // Synchronize the work across groups, each of which went into its own stream
        // NOLINT_NEXT_LINE(whitespace/operators)
        for (int g = 0; g < ws_groups(); ++g) {
          CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(g)));
        }
      }  // end of i
    }  // end of dB

    // compute dE/dW = dY * X
    if (this->param_propagate_down_[0]) {
      Btype* weight_diff = this->blobs_[0]->template mutable_gpu_diff<Btype>();
      for (int i = 0; i < top.size(); ++i) {
        Btype* top_diff = top[i]->mutable_gpu_diff<Btype>();
        const Btype* bottom_data = bottom[i]->gpu_data<Btype>();
        // Backward through cuDNN in parallel over groups and gradients.
        for (int g = 0; g < groups(); ++g) {
          unsigned char* pspace = static_cast<unsigned char*>(ws->data()) + gsize * idxg(g);
          // Gradient w.r.t. weights.
          CUDNN_CHECK(cudnnConvolutionBackwardFilter(Caffe::cudnn_handle(idxg(g)),
              cudnn::dataType<Btype>::one,
              bwd_bottom_descs_[i], bottom_data + bottom_offset_ * g,
              bwd_top_descs_[i], top_diff + top_offset_ * g,
              bwd_conv_filter_descs_[i], bwd_filter_algo_[i], pspace, gsize,
              cudnn::dataType<Btype>::one,
              bwd_filter_desc_, weight_diff + this->weight_offset_ * g));
        }  // end of groups
        // Synchronize the work across groups, each of which went into its own stream
        // NOLINT_NEXT_LINE(whitespace/operators)
        for (int g = 0; g < ws_groups(); ++g) {
          CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(g)));
        }
      }  // end of i
    }

    // Backward propagate grad wrt bottom data dE/dX= dE/dY * W
    const Btype* weight = this->blobs_[0]->template gpu_data<Btype>();
    for (int i = 0; i < top.size(); ++i) {
      if (propagate_down[i]) {
        // Backward in parallel over groups
        for (int g = 0; g < groups(); ++g) {
          Btype* top_diff = top[i]->mutable_gpu_diff<Btype>();
          Btype* bottom_diff = bottom[i]->mutable_gpu_diff<Btype>();
          unsigned char* pspace = static_cast<unsigned char*>(ws->data()) + gsize * idxg(g);
          CUDNN_CHECK(cudnnConvolutionBackwardData(Caffe::cudnn_handle(idxg(g)),
              cudnn::dataType<Btype>::one,
              bwd_filter_desc_, weight + this->weight_offset_ * g,
              bwd_top_descs_[i], top_diff + top_offset_ * g,
              bwd_conv_data_descs_[i],
              bwd_data_algo_[i], pspace, gsize,
              cudnn::dataType<Btype>::zero,
              bwd_bottom_descs_[i], bottom_diff + bottom_offset_ * g));
        }
        // Synchronize the work across groups.
        // NOLINT_NEXT_LINE(whitespace/operators)
        for (int g = 0; g < ws_groups(); ++g) {
          CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(g)));
        }
      }  // end if propagate down
    }  // end for i
  }

  ++bwd_count_;
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
