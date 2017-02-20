#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/iou_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void IOUAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }

  image_iter_ = 0;
  history_size_ = this->layer_param_.accuracy_param().history_size();
}

template <typename Dtype>
void IOUAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(1);  // Accuracy is a scalar; 0 axes.
  top_shape[0] = 1;
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    int num_labels = bottom[0]->shape(label_axis_);
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = num_labels;
    top[1]->Reshape(top_shape_per_class);
  }

  const int num_labels = bottom[0]->shape(label_axis_);
  int num_predictions = num_labels;
  confusion_matrix_ = vector<vector<Dtype>>(num_labels, vector<Dtype>(num_predictions, 0.0));
  batch_size_ = bottom[0]->shape(0);
}

template <typename Dtype>
void IOUAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);

  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }

      if(label_value < 0 || label_value >= num_labels || num_labels < 0) {
        //LOG(INFO) << "Invalid label_value: " << label_value;
        continue;
      }

      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);

      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }
      std::partial_sort(bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());

      int prediction_value = bottom_data_vector[0].second;

      if(prediction_value < 0 || prediction_value >= num_labels || prediction_value < 0) {
        //LOG(INFO) << "Invalid label_value: " << label_value;
        continue;
      }

      confusion_matrix_[label_value][prediction_value] += 1;

      ++count;
    }
  }

  vector<Dtype> iou(num_labels, 0);
  Dtype mean_iou = 0, count_iou = 0;
  for (int i = 0; i < num_labels; ++i) {
    bool valid = false;
    iou[i] = getIOUScoreForLabel(confusion_matrix_, i, valid);
    if(valid) {
      mean_iou += iou[i];
      count_iou++;
    }
  }
  mean_iou = count_iou>0? (mean_iou / count_iou) : 0;

  //LOG(INFO) << "MeanIOU: " << mean_iou;
  top[0]->mutable_cpu_data()[0] = mean_iou;
  for (int i = 0; i < std::min<Dtype>(num_labels, top[1]->count()); ++i) {
    //LOG(INFO) << "IOU[" << i << "]: " << iou[i];
    if (top.size() > 1) {
      top[1]->mutable_cpu_data()[i] = iou[i];
    }
  }
  // IOUAccuracy layer should not be used as a loss function.

  //Reduce the effect of old samples by scaling down confusion matrix.
  //image_iter_ has to be increased by batch_size_ after forward/backward.
  for(int b=0; b<batch_size_; b++) {
    if(history_size_ == 0) {
      for (int i = 0; i < confusion_matrix_.size(); ++i) {
        for (int j = 0; j < confusion_matrix_[i].size(); ++j) {
          confusion_matrix_[i][j] = 0;
        }
      }
    } else {
      if((image_iter_ % history_size_) == 0) {
        for (int i = 0; i < confusion_matrix_.size(); ++i) {
          for (int j = 0; j < confusion_matrix_[i].size(); ++j) {
            confusion_matrix_[i][j] = confusion_matrix_[i][j] / 2;
          }
        }
      }
    }
    image_iter_++;
  }
}

//Information: See definition of IU (a.k.a. IOU) here: https://arxiv.org/abs/1411.4038
template <typename Dtype>
Dtype IOUAccuracyLayer<Dtype>::getIOUScoreForLabel(vector<vector<Dtype> >& confusion_matrix, int label, bool& valid) {
  Dtype ture_pos = confusion_matrix[label][label];
  Dtype label_count = 0;
  for(int j=0; j<confusion_matrix[label].size(); j++) {
    label_count += confusion_matrix[label][j];
  }
  Dtype label_observed = 0;
  for(int j=0; j<confusion_matrix[label].size(); j++) {
    label_observed += confusion_matrix[j][label];
  }
  Dtype den = label_count + label_observed - ture_pos;
  valid = (den>0);
  Dtype score = (valid? (ture_pos/den) : 0);

  return score;
}

INSTANTIATE_CLASS(IOUAccuracyLayer);
REGISTER_LAYER_CLASS(IOUAccuracy);

}  // namespace caffe
