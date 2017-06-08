#ifndef CAFFE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ImageLabelDataLayer : public Layer<Dtype> {
 public:
  explicit ImageLabelDataLayer(const LayerParameter& param);
  virtual ~ImageLabelDataLayer();
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) override;

  const char* type() const override { return "ImageLabelData"; }
  int ExactNumBottomBlobs() const override { return 0; }
  int ExactNumTopBlobs() const override { return 2; }
  
  void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) override;
    
  void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) override;
	  
  void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) override {}
  void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) override {}
	  	  
  void InitRand(int seed) {
    const unsigned int rng_seed = seed? seed : caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  }

  unsigned int Rand() const {
    CHECK(rng_);
    caffe::rng_t *rng =
        static_cast<caffe::rng_t *>(rng_->generator());
    // this doesn't actually produce a uniform distribution
    return static_cast<unsigned int>((*rng)());
  }

  unsigned int Rand(int n) const {
    CHECK_GT(n, 0);
    return Rand() % n;
  }

  //bool ShareInParallel() const override {
  //  return false;
  //}

protected:
  shared_ptr<DataLayer<Dtype>> data_layer_, label_layer_;
  shared_ptr<DataTransformer<Dtype>>  data_transformer_;
  bool needs_rand_;
  shared_ptr<Caffe::RNG> rng_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
