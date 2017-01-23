#ifndef CAFFE_IMAGE_LABEL_DATA_LAYER_H
#define CAFFE_IMAGE_LABEL_DATA_LAYER_H

#include <random>
#include <vector>

#include <opencv2/core/core.hpp>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template<typename Dtype>
class ImageLabelDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageLabelDataLayer(const LayerParameter &param);

  virtual ~ImageLabelDataLayer();

  virtual void DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                              const vector<Blob<Dtype> *> &top);

  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }

  virtual inline const char *type() const { return "ImageLabelData"; }

  virtual inline int ExactNumBottomBlobs() const { return 0; }

  virtual inline int ExactNumTopBlobs() const { return -1; }

  virtual inline int MaxTopBlobs() const { return 3; }

  virtual inline int MinTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;

  virtual void ShuffleImages();

  virtual void SampleScale(cv::Mat *image, cv::Mat *label);

  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::string> image_lines_;
  vector<std::string> label_lines_;
  int lines_id_;

  Blob<Dtype> transformed_label_;

  int label_margin_h_;
  int label_margin_w_;

  std::mt19937 *rng_;
};

} // namspace caffe

#endif //CAFFE_IMAGE_LABEL_DATA_LAYER_H
