#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_label_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {


template <typename Ftype, typename Btype>
ImageLabelDataLayer<Ftype, Btype>::ImageLabelDataLayer(const LayerParameter& param) :
  BasePrefetchingDataLayer<Ftype, Btype>(param) {
}

template <typename Ftype, typename Btype>
ImageLabelDataLayer<Ftype, Btype>::~ImageLabelDataLayer() {
  this->StopInternalThread();
}

template<typename Ftype, typename Btype>
void ImageLabelDataLayer<Ftype, Btype>::InitializePrefetch() {
  if (layer_inititialized_flag_.is_set()) {
    return;
  }
  data_layer_->InitializePrefetch();
  label_layer_->InitializePrefetch();

  //prefetch of this not required as this doesn't read data directly.
  //BasePrefetchingDataLayer<Ftype, Btype>::InitializePrefetch();

  layer_inititialized_flag_.set();
}

template<typename Ftype, typename Btype>
void ImageLabelDataLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  CHECK(this->layer_param_.image_label_data_param().has_backend()) << "ImageLabelDataParameter should specify backend";

  LayerParameter data_param(this->layer_param_);
  data_param.mutable_transform_param()->set_crop_size(this->layer_param_.transform_param().crop_size());
  data_param.mutable_transform_param()->set_mirror(this->layer_param_.transform_param().mirror());
  data_param.mutable_data_param()->set_source(this->layer_param_.image_label_data_param().image_list_path());
  data_param.mutable_data_param()->set_batch_size(this->layer_param_.image_label_data_param().batch_size());
  data_param.mutable_data_param()->set_threads(this->layer_param_.image_label_data_param().threads());
  data_param.mutable_data_param()->set_parser_threads(this->layer_param_.data_param().threads());
  data_param.mutable_data_param()->set_backend(static_cast<DataParameter_DB>(this->layer_param_.image_label_data_param().backend()));

  LayerParameter label_param(this->layer_param_);
  label_param.mutable_transform_param()->set_crop_size(this->layer_param_.transform_param().crop_size());
  label_param.mutable_transform_param()->set_mirror(this->layer_param_.transform_param().mirror());
  label_param.mutable_data_param()->set_source(this->layer_param_.image_label_data_param().label_list_path());
  label_param.mutable_data_param()->set_batch_size(this->layer_param_.image_label_data_param().batch_size());
  label_param.mutable_data_param()->set_threads(this->layer_param_.image_label_data_param().threads());
  label_param.mutable_data_param()->set_parser_threads(this->layer_param_.data_param().threads());
  label_param.mutable_data_param()->set_backend(static_cast<DataParameter_DB>(this->layer_param_.image_label_data_param().backend()));

  //Create the internal layers
  data_layer_.reset(new DataLayerExtended<Ftype, Btype>(data_param));
  label_layer_.reset(new DataLayerExtended<Ftype, Btype>(label_param));

  //Populate bottom and top
  vector<Blob*> data_bottom_vec;
  vector<Blob*> data_top_vec;
  data_top_vec.push_back(top[0]);
  data_layer_->LayerSetUp(data_bottom_vec, data_top_vec);

  vector<Blob*> label_bottom_vec;
  vector<Blob*> label_top_vec;
  label_top_vec.push_back(top[1]);
  label_layer_->LayerSetUp(label_bottom_vec, label_top_vec);

  BasePrefetchingDataLayer<Ftype, Btype>::LayerSetUp(bottom, top);
}

// This function is called on prefetch thread
template <typename Ftype, typename Btype>
void ImageLabelDataLayer<Ftype, Btype>::load_batch(Batch<Ftype>* batch, int thread_id, size_t queue_id) {
  vector<Blob*> data_bottom_vec;
  vector<Blob*> data_top_vec;
  data_top_vec.push_back(&batch->data_);
  data_layer_->Forward(data_bottom_vec, data_top_vec);

  vector<Blob*> label_bottom_vec;
  vector<Blob*> label_top_vec;
  label_top_vec.push_back(&batch->label_);
  label_layer_->Forward(label_bottom_vec, label_top_vec);

  //batch->data_.CopyFrom(*data, true, true);
  batch->data_.gpu_data();
  batch->data_.cpu_data();

  //batch->label_.CopyFrom(*label, true, true);
  batch->label_.gpu_data();
  batch->label_.cpu_data();
}

template<typename Ftype, typename Btype>
void ImageLabelDataLayer<Ftype, Btype>::InternalThreadEntryN(size_t thread_id) {
#ifndef CPU_ONLY
  //const bool use_gpu_transform = this->is_gpu_transform();
#endif
  static thread_local bool iter0 = this->phase_ == TRAIN;
  if (iter0 && this->net_inititialized_flag_ != nullptr) {
    this->net_inititialized_flag_->wait();
  } else {  // nothing to wait -> initialize and start pumping
    std::lock_guard<std::mutex> lock(this->mutex_in_);
    this->InitializePrefetch();
    start_reading();
    iter0 = false;
  }
  try {
    while (!this->must_stop(thread_id)) {
      const size_t qid = this->queue_id(thread_id);
#ifndef CPU_ONLY
      shared_ptr<Batch<Ftype>> batch = this->prefetches_free_[qid]->pop();

      CHECK_EQ((size_t) -1, batch->id());
      load_batch(batch.get(), thread_id, qid);
      /*if (Caffe::mode() == Caffe::GPU) {
        if (!use_gpu_transform) {
          batch->data_.async_gpu_push();
        }
        if (this->output_labels_) {
          batch->label_.async_gpu_push();
        }
        CUDA_CHECK(cudaStreamSynchronize(Caffe::th_stream_aux(Caffe::STREAM_ID_ASYNC_PUSH)));
      }*/

      this->prefetches_full_[qid]->push(batch);
#else
      shared_ptr<Batch<Ftype>> batch = this->prefetches_free_[qid]->pop();
      load_batch(batch.get(), thread_id, qid);
      this->prefetches_full_[qid]->push(batch);
#endif

      if (iter0) {
        if (this->net_iteration0_flag_ != nullptr) {
          this->net_iteration0_flag_->wait();
        }
        std::lock_guard<std::mutex> lock(this->mutex_out_);
        if (this->net_inititialized_flag_ != nullptr) {
          this->net_inititialized_flag_ = nullptr;  // no wait on the second round
          this->InitializePrefetch();
          start_reading();
        }
        //if (this->auto_mode_) {
        //  break;
        //}  // manual otherwise, thus keep rolling
        iter0 = false;
      }
    }
  } catch (boost::thread_interrupted&) {
     LOG(INFO) << "InternalThreadEntryN was interrupted" << std::endl;
  }
}

INSTANTIATE_CLASS_FB(ImageLabelDataLayer);
REGISTER_LAYER_CLASS(ImageLabelData);

}  // namespace caffe
#endif  // USE_OPENCV
