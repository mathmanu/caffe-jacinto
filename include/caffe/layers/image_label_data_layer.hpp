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
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Ftype, typename Btype>
class ImageLabelDataLayer : public Layer<Ftype, Btype> {
 public:
  explicit ImageLabelDataLayer(const LayerParameter& param);
  virtual ~ImageLabelDataLayer();
  void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

  const char* type() const override { return "ImageLabelData"; }
  int ExactNumBottomBlobs() const override { return 0; }
  int ExactNumTopBlobs() const override { return 2; }
  
  void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
    
  void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
	  
  void Backward_cpu(const vector<Blob*>& top, const vector<bool>& propagate_down,
      const vector<Blob*>& bottom) override {}
  void Backward_gpu(const vector<Blob*>& top, const vector<bool>& propagate_down,
      const vector<Blob*>& bottom) override {}
	  	  
  shared_ptr<DataLayer<Ftype,Btype>> data_layer_, label_layer_;
  shared_ptr<DataTransformer<Ftype>>  data_transformer_;
  bool needs_rand_;
  //bool ShareInParallel() const override {
  //  return false;
  //}  
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
