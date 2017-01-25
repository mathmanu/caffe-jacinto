#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_weighted_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SoftmaxLossForwardProbSumGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* prob_sum_data,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
    	prob_sum_data[index] = 0;
    	counts[index] = 0;
    } else {
    	prob_sum_data[index] = prob_data[n * dim + label_value * spatial_dim + s];
    	counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  iter_count_++;
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  iou_accuracy_layer_->Forward(iou_accuracy_bottom_vec_, iou_accuracy_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
                                                        valid_count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts, int num_labels) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if ((has_ignore_label_ && label_value == ignore_label_) ||
        (label_value<0 || label_value >= num_labels)) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardOHEMGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts, const Dtype* prob_data, Dtype prob_threshold, int num_labels) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    bool ignore_pixel = (has_ignore_label_ && label_value == ignore_label_) ||
    		(prob_data[n * dim + label_value * spatial_dim + s] > prob_threshold) ||
            (label_value<0 || label_value >= num_labels);

    if (ignore_pixel) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

/*
template <typename Dtype>
__global__ void AssignLabelWeightsGPUKernel(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, int num_labels, const Dtype* label_weights) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if ((has_ignore_label_ && label_value == ignore_label_) ||
        (label_value<0 || label_value >= num_labels)) {
      //nothing to do
    } else {
      const int label_weight = label_weights[label_value];
      bottom_diff[n * dim + label_value * spatial_dim + s] *= label_weight;
    }
  }
}
*/

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::AssignLabelWeights_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  /*
  //this GPU implementation is not working correctly - so call the CPU version instead.

  int num_labels = bottom[0]->shape(softmax_axis_);
  if(this->layer_param_.loss_param().has_num_label_weights()) {
    num_labels = std::min(num_labels, this->layer_param_.loss_param().num_label_weights());
  }
  //LOG(INFO) << "num_labels = " << num_labels;

  vector<Dtype> label_weights_cur(num_labels, Dtype(1.0));
  const Dtype *iou_class_data = iou_class_.cpu_data();
  Dtype iou_mean = *iou_mean_.cpu_data();
  for (int i = 0; i < num_labels; ++i) {
    if(iou_class_data[i] > 0) {
      Dtype weight = (iou_class_data[i] + 1.0 - iou_mean);
      weight = pow(weight, 4);//2
      label_weights_cur[i] = std::max<Dtype>(std::min<Dtype>(1.0 / weight, 10.0), 0.1);
    } else {
      label_weights_cur[i] = 1.0;
    }
  }

  Dtype* label_weights_data_cpu =  label_weights_blob_.mutable_cpu_data();
  for (int i = 0; i < num_labels; ++i) {
    label_weights_data_cpu[i] = label_weights_data_cpu[i] * 0.99 + label_weights_cur[i] * 0.01;
    if((iter_count_ % 1000) == 0) {
      LOG(INFO) << " label_weights [" << i << "] = " << label_weights_cur[i] << ", "<< label_weights_data_cpu[i];
    }
  }

  const Dtype* label_weights_data_gpu =  label_weights_blob_.gpu_data();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  AssignLabelWeightsGPUKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, num_labels,
        label_weights_data_gpu);
  */

  this->AssignLabelWeights_cpu(bottom, top);
  bottom[0]->gpu_diff();
  label_weights_blob_.gpu_data();
}

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    Dtype bootstrap_samples_fraction = this->layer_param_.loss_param().bootstrap_samples_fraction();
    Dtype bootstrap_prob_threshold = this->layer_param_.loss_param().bootstrap_prob_threshold();

    int num_labels = bottom[0]->shape(softmax_axis_);
    if(this->layer_param_.loss_param().has_num_label_weights()) {
      num_labels = std::min(num_labels, this->layer_param_.loss_param().num_label_weights());
    }

    if(prob_threshold_ > 0) {
		// NOLINT_NEXT_LINE(whitespace/operators)
		SoftmaxLossBackwardOHEMGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
			outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts, prob_data, prob_threshold_, num_labels);
    } else {
		// NOLINT_NEXT_LINE(whitespace/operators)
		SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
			outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts, num_labels);
    }

    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if ((normalization_ == LossParameter_NormalizationMode_VALID && has_ignore_label_) ||
        bootstrap_samples_fraction > 0 || bootstrap_prob_threshold > 0) {
      caffe_gpu_asum(prob_.count(), counts, &valid_count);
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0] /
                              get_normalizer(normalization_, valid_count);
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);

    if(this->layer_param_.loss_param().assign_label_weights()) {
      this->AssignLabelWeights_gpu(bottom, top);
    }

    if(bootstrap_prob_threshold > 0 || bootstrap_samples_fraction > 0) {
        if(bootstrap_prob_threshold > 0) {
			prob_threshold_ = bootstrap_prob_threshold;
		} else if(bootstrap_samples_fraction > 0) {
			int count_low = ceil(bootstrap_samples_fraction * prob_.count() * 0.75);
			int count_high = ceil(bootstrap_samples_fraction * prob_.count() * 1.25);
			if(valid_count < count_low) {
                //LOG(WARNING) << "prob_.count()=" << prob_.count() << " valid_count=" << valid_count << " count_low" << count_low
                //    << " prob_threshold_=" << prob_threshold_ << " increasing";
				prob_threshold_ = prob_threshold_ + 0.001;
			}
			if(valid_count > count_high) {
                //LOG(WARNING) << "prob_.count()=" << prob_.count() << " valid_count=" << valid_count << " count_high" << count_high
                //    << " prob_threshold_=" << prob_threshold_ << " decreasing";
				prob_threshold_ = prob_threshold_ - 0.001;
			}
			Dtype prob_threshold_min = 0.0;
			Dtype prob_threshold_max = 1.0;
			prob_threshold_ = std::min(std::max(prob_threshold_, prob_threshold_min), prob_threshold_max);

	        LOG_EVERY_N(INFO, 1000) << " prob_threshold: " << prob_threshold_ << ", bootstrap_count: " << valid_count
	            << " (count_low:" << count_low << ", count_high:" << count_high << ")";
		}
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithWeightedLossLayer);

}  // namespace caffe
