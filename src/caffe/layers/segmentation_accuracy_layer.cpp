#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/segmentation_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/solver.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void SegmentationAccuracyLayer<Ftype, Btype>::LayerSetUp(
  const vector<Blob*>& bottom, const vector<Blob*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Ftype, typename Btype>
void SegmentationAccuracyLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  
  num_predictions_ = bottom[0]->shape(label_axis_);
  num_labels_ = bottom[0]->shape(label_axis_);
  
  //confusion_matrix. first two indices (batch, channels) are size 1.
  vector<int> top_shape_confusion_matrix(4, 1);
  top_shape_confusion_matrix[2] = num_labels_;
  top_shape_confusion_matrix[3] = num_predictions_;
  top[0]->Reshape(top_shape_confusion_matrix);
  
  confusion_matrix_ = vector<vector<float>>(num_labels_, vector<float>(num_predictions_, 0.0));
}

template <typename Ftype, typename Btype>
void SegmentationAccuracyLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  // IOUAccuracy layer should not be used as a loss function as it doesn't have backward.
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  const Ftype* bottom_label = bottom[1]->cpu_data<Ftype>();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);

  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }

      if(label_value < 0 || label_value >= num_labels_ || num_labels_ < 0) {
        //LOG(INFO) << "Invalid label_value: " << label_value;
        continue;
      }

      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);

      std::vector<std::pair<Ftype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }
      std::partial_sort(bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Ftype, int> >());
      
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          int prediction_value = bottom_data_vector[k].second;
          if(prediction_value < 0 || prediction_value >= num_labels) {
            //LOG(INFO) << "Invalid label_value: " << label_value;
            continue;
          }
          confusion_matrix_[label_value][prediction_value] += 1;
          break;
        }
      }
    }
  }

  for (int i = 0; i < num_labels_; ++i) {
    for (int j = 0; j < num_predictions_; ++j) {
      top[0]->mutable_cpu_data<Ftype>()[i*num_labels_+j] = confusion_matrix_[i][j];
    }
  }
}


INSTANTIATE_CLASS_FB(SegmentationAccuracyLayer);
REGISTER_LAYER_CLASS(SegmentationAccuracy);

}  // namespace caffe
