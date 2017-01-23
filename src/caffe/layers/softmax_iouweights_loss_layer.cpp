#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_iouweights_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossIOUWeightsLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_param.set_name(this->layer_param_.name()+"_Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  LayerParameter iou_accuracy_param(this->layer_param_);
  if(iou_accuracy_param.loss_weight_size()!=0 && iou_accuracy_param.loss_weight_size()!=2) {
    int num_needed = 2 - iou_accuracy_param.loss_weight_size();
    for(int i=0; i<num_needed; i++) {
      iou_accuracy_param.add_loss_weight(1.0);
    }
  }
  iou_accuracy_param.set_type("IOUAccuracy");
  iou_accuracy_param.set_name(this->layer_param_.name()+"_IOUAccuracy");
  iou_accuracy_layer_ = LayerRegistry<Dtype>::CreateLayer(iou_accuracy_param);
  iou_accuracy_bottom_vec_.clear();
  iou_accuracy_bottom_vec_.push_back(bottom[0]);
  iou_accuracy_bottom_vec_.push_back(bottom[1]);
  iou_accuracy_top_vec_.clear();
  iou_accuracy_top_vec_.push_back(&iou_mean_);
  iou_accuracy_top_vec_.push_back(&iou_class_);
  iou_accuracy_layer_->SetUp(iou_accuracy_bottom_vec_, iou_accuracy_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  Dtype bootstrap_prob_threshold = this->layer_param_.loss_param().bootstrap_prob_threshold();
  Dtype bootstrap_samples_fraction = this->layer_param_.loss_param().bootstrap_samples_fraction();
  prob_threshold_ = (bootstrap_prob_threshold > 0? bootstrap_prob_threshold :
		  (bootstrap_samples_fraction>0? 0.5: 0));
  iter_count_ = -1;

  if(this->layer_param_.loss_param().assign_label_weights()) {
    softmax_axis_ =
        bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
    int num_labels = bottom[0]->shape(softmax_axis_);
    if(this->layer_param_.loss_param().has_num_label_weights()) {
      num_labels = std::min(num_labels, this->layer_param_.loss_param().num_label_weights());
    }
    int label_weights_blob_shape_arr[4] = {1, num_labels, 1, 1};
    vector<int> label_weights_blob_shape(label_weights_blob_shape_arr, label_weights_blob_shape_arr+4);
    label_weights_blob_.Reshape(label_weights_blob_shape);
    Dtype* label_weights_data_cpu =  label_weights_blob_.mutable_cpu_data();
    for (int i = 0; i < num_labels; ++i) {
      label_weights_data_cpu[i] = 1.0;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossIOUWeightsLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  iou_accuracy_layer_->Reshape(iou_accuracy_bottom_vec_, iou_accuracy_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithLossIOUWeightsLayer<Dtype>::AssignLabelWeights_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

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

  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype *label_data = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      Dtype selected_weight;
      const int label_value = static_cast<int>(label_data[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        selected_weight = 0;
      } else if(label_value < num_labels){
        selected_weight = label_weights_data_cpu[label_value];
      } else {
        selected_weight = 0.0;
      }
      for(int l = 0; l<num_labels; l++) {
        bottom_diff[i * dim + l * inner_num_ + j] *= selected_weight;
      }
    }
  }
}

template <typename Dtype>
Dtype SoftmaxWithLossIOUWeightsLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SoftmaxWithLossIOUWeightsLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  iter_count_++;
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  iou_accuracy_layer_->Forward(iou_accuracy_bottom_vec_, iou_accuracy_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      ++count;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossIOUWeightsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    Dtype bootstrap_samples_fraction = this->layer_param_.loss_param().bootstrap_samples_fraction();
    Dtype bootstrap_prob_threshold = this->layer_param_.loss_param().bootstrap_prob_threshold();

    int num_labels = bottom[0]->shape(softmax_axis_);
    if(this->layer_param_.loss_param().has_num_label_weights()) {
      num_labels = std::min(num_labels, this->layer_param_.loss_param().num_label_weights());
    }

    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        bool ignore_pixel = (has_ignore_label_ && label_value == ignore_label_) ||
        		(prob_threshold_ > 0 && prob_data[i * dim + label_value * inner_num_ + j] > prob_threshold_) ||
        		(label_value<0 || label_value >= num_labels);

        if (ignore_pixel) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          ++count;
        }
      }
    }

    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);

    if(this->layer_param_.loss_param().assign_label_weights()) {
      this->AssignLabelWeights_cpu(bottom, top);
    }

    if(bootstrap_prob_threshold > 0 || bootstrap_samples_fraction > 0) {
        if(bootstrap_prob_threshold > 0) {
			prob_threshold_ = bootstrap_prob_threshold;
		} else if(bootstrap_samples_fraction > 0) {
			int count_low = ceil(bootstrap_samples_fraction * prob_.count() * 0.75);
			int count_high = ceil(bootstrap_samples_fraction * prob_.count() * 1.25);
			if(count < count_low) {
				prob_threshold_ = prob_threshold_ + 0.001;
			}
			if(count > count_high) {
				prob_threshold_ = prob_threshold_ - 0.001;
			}
			Dtype prob_threshold_min = 1e-10;
			Dtype prob_threshold_max = 1.0;
			prob_threshold_ = std::min(std::max(prob_threshold_, prob_threshold_min), prob_threshold_max);

	        LOG_EVERY_N(INFO, 1000) << " prob_threshold: " << prob_threshold_ << ", bootstrap_count: " << count
	            << " (count_low:" << count_low << ", count_high:" << count_high << ")";
		}
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossIOUWeightsLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossIOUWeightsLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLossIOUWeights);

}  // namespace caffe
