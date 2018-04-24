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
ImageLabelDataLayer<Ftype, Btype>::ImageLabelDataLayer(const LayerParameter& param, size_t solver_rank) :
  QuantizedLayer<Ftype, Btype>(param), solver_rank_(solver_rank) {
}

template <typename Ftype, typename Btype>
ImageLabelDataLayer<Ftype, Btype>::~ImageLabelDataLayer() {
}

template<typename Ftype, typename Btype>
void ImageLabelDataLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  CHECK(this->layer_param_.image_label_data_param().has_backend()) << "ImageLabelDataParameter should specify backend";

  //Hang is observed when using default number of threads. Set the number
  bool has_threads = this->layer_param_.image_label_data_param().has_threads();
  int input_threads = this->layer_param_.image_label_data_param().threads();
  int threads = has_threads? std::min<int>(std::max<int>(input_threads, 1), 8) : 2;

  int num_mean_values = this->layer_param_.transform_param().mean_value_size();

  unsigned int random_seed = this->layer_param_.transform_param().has_random_seed()? 
     this->layer_param_.transform_param().random_seed() : caffe_rng_rand();
  
  LayerParameter data_param(this->layer_param_);
  data_param.mutable_transform_param()->set_crop_size(this->layer_param_.transform_param().crop_size());
  data_param.mutable_transform_param()->set_mirror(this->layer_param_.transform_param().mirror());
  data_param.mutable_transform_param()->clear_mean_value();
  for(int i=0; i<num_mean_values; i++) {
    data_param.mutable_transform_param()->add_mean_value(this->layer_param_.transform_param().mean_value(i));
  }
  data_param.mutable_transform_param()->set_random_seed(random_seed);  
  data_param.mutable_data_param()->set_shuffle(this->layer_param_.image_label_data_param().shuffle());    
  data_param.mutable_data_param()->set_source(this->layer_param_.image_label_data_param().image_list_path());
  data_param.mutable_data_param()->set_batch_size(this->layer_param_.image_label_data_param().batch_size());
  data_param.mutable_data_param()->set_backend(static_cast<DataParameter_DB>(this->layer_param_.image_label_data_param().backend()));
  if(has_threads) {
    data_param.mutable_data_param()->set_threads(threads);
    data_param.mutable_data_param()->set_parser_threads(threads);
  }
  
  LayerParameter label_param(this->layer_param_);
  label_param.mutable_transform_param()->set_crop_size(this->layer_param_.transform_param().crop_size());
  label_param.mutable_transform_param()->set_mirror(this->layer_param_.transform_param().mirror());
  //label doesn't have mean value - add zeros, just in case there is a rand sequence mismatch some where
  label_param.mutable_transform_param()->clear_mean_value();
  for(int i=0; i<num_mean_values; i++) {
    label_param.mutable_transform_param()->add_mean_value(0);
  }
  label_param.mutable_transform_param()->set_random_seed(random_seed);  
  label_param.mutable_data_param()->set_shuffle(this->layer_param_.image_label_data_param().shuffle());   
  label_param.mutable_data_param()->set_source(this->layer_param_.image_label_data_param().label_list_path());
  label_param.mutable_data_param()->set_batch_size(this->layer_param_.image_label_data_param().batch_size());
  label_param.mutable_data_param()->set_backend(static_cast<DataParameter_DB>(this->layer_param_.image_label_data_param().backend()));
  if(has_threads) {
    label_param.mutable_data_param()->set_threads(threads);
    label_param.mutable_data_param()->set_parser_threads(threads);
  }
  
  //Create the internal layers
  data_layer_.reset(new DataLayer<Ftype, Btype>(data_param, solver_rank_));
  label_layer_.reset(new DataLayer<Ftype, Btype>(label_param, solver_rank_));

  //Populate bottom and top
  vector<Blob*> data_bottom_vec;
  vector<Blob*> data_top_vec;
  data_top_vec.push_back(top[0]);
  data_layer_->LayerSetUp(data_bottom_vec, data_top_vec);

  const int crop_size = this->layer_param_.transform_param().crop_size();
  int data_height = crop_size? crop_size : top[0]->height();
  int data_width = crop_size? crop_size : top[0]->width();
  vector<int> data_shape = {top[0]->num(), top[0]->channels(), data_height, data_width};
  top[0]->Reshape(data_shape);

  vector<Blob*> label_bottom_vec;
  vector<Blob*> label_top_vec;
  label_top_vec.push_back(top[1]);
  label_layer_->LayerSetUp(label_bottom_vec, label_top_vec);

  int label_channels = 1;
  vector<int> label_shape = {top[1]->num(), label_channels, data_height, data_width};
  top[1]->Reshape(label_shape);
}

template<typename Ftype, typename Btype>
void ImageLabelDataLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  vector<Blob*> data_bottom_vec;
  vector<Blob*> data_top_vec;
  data_top_vec.push_back(top[0]);
  data_layer_->Reshape(data_bottom_vec, data_top_vec);	
  
  vector<Blob*> label_bottom_vec;
  vector<Blob*> label_top_vec;
  label_top_vec.push_back(top[1]);
  label_layer_->Reshape(label_bottom_vec, label_top_vec);  
}
  
template<typename Ftype, typename Btype>
void ImageLabelDataLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {

  vector<Blob*> data_bottom_vec;
  vector<Blob*> data_top_vec;
  data_top_vec.push_back(top[0]);
  
  vector<Blob*> label_bottom_vec;
  vector<Blob*> label_top_vec;
  label_top_vec.push_back(top[1]);

  data_layer_->Forward(data_bottom_vec, data_top_vec);
  label_layer_->Forward(label_bottom_vec, label_top_vec);  
  
  this->Quantize_cpu(bottom, top);
}

INSTANTIATE_CLASS_FB(ImageLabelDataLayer);
REGISTER_LAYER_CLASS_R(ImageLabelData);

}  // namespace caffe
#endif  // USE_OPENCV
