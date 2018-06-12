#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_deconv_layer.hpp"

namespace caffe {

__global__ void sync_deconv_groups() {}

template<typename Ftype, typename Btype>
void CuDNNDeconvolutionLayer<Ftype, Btype>::Forward_gpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  //this->Quantize_gpu(bottom, top);
  const Ftype* weight = this->blobs_[0]->template gpu_data<Ftype>();
  for (int i = 0; i < bottom.size(); ++i) {
    const Ftype* bottom_data = bottom[i]->gpu_data<Ftype>();
    Ftype* top_data = top[i]->mutable_gpu_data<Ftype>();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionBackwardData(
          handle_[g],
          cudnn::dataType<Ftype>::one,
          filter_desc_,
          weight + this->weight_offset_ * g,
          bottom_descs_[i],
          bottom_data + bottom_offset_ * g,
          conv_descs_[i],
          bwd_data_algo_[i],
          workspace[g],
          workspace_bwd_data_sizes_[i],
          cudnn::dataType<Ftype>::zero,
          top_descs_[i],
          top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Ftype* bias_data = this->blobs_[1]->template gpu_data<Ftype>();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
                                   cudnn::dataType<Ftype>::one,
                                   bias_desc_,
                                   bias_data + bias_offset_ * g,
                                   cudnn::dataType<Ftype>::one,
                                   top_descs_[i],
                                   top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_deconv_groups<<<1, 1>>>();  // FIXME
  }
  this->Quantize_gpu(bottom, top);  
}

template<typename Ftype, typename Btype>
void CuDNNDeconvolutionLayer<Ftype, Btype>::Backward_gpu(
    const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  const Btype* weight = NULL;
  Btype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->template gpu_data<Btype>();
    weight_diff = this->blobs_[0]->template mutable_gpu_diff<Btype>();
  }
  Btype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->template mutable_gpu_diff<Btype>();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Btype* top_diff = top[i]->gpu_diff<Btype>();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0 * this->group_ + g],
                                                 cudnn::dataType<Btype>::one,
                                                 top_descs_[i],
                                                 top_diff + top_offset_ * g,
                                                 cudnn::dataType<Btype>::one,
                                                 bias_desc_,
                                                 bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Btype* bottom_data = bottom[i]->gpu_data<Btype>();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
            handle_[1 * this->group_ + g],
            cudnn::dataType<Btype>::one,
            top_descs_[i],
            top_diff + top_offset_ * g,
            bottom_descs_[i],
            bottom_data + bottom_offset_ * g,
            conv_descs_[i],
            bwd_filter_algo_[i],
            workspace[1 * this->group_ + g],
            workspace_bwd_filter_sizes_[i],
            cudnn::dataType<Btype>::one,
            filter_desc_,
            weight_diff + this->weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->template gpu_data<Btype>();
        }
        Btype* bottom_diff = bottom[i]->mutable_gpu_diff<Btype>();
        CUDNN_CHECK(
            cudnnConvolutionForward(handle_[2 * this->group_ + g],
                                    cudnn::dataType<Btype>::one,
                                    top_descs_[i],
                                    top_diff + top_offset_ * g,
                                    filter_desc_,
                                    weight + this->weight_offset_ * g,
                                    conv_descs_[i],
                                    fwd_algo_[i],
                                    workspace[2 * this->group_ + g],
                                    workspace_fwd_sizes_[i],
                                    cudnn::dataType<Btype>::zero,
                                    bottom_descs_[i],
                                    bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_deconv_groups<<<1, 1>>>();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(CuDNNDeconvolutionLayer);

}  // namespace caffe
#endif
