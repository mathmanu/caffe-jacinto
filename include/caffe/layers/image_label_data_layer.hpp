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

  virtual void DataLayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

  const char *type() const override { return "ImageLabelData"; }
  int ExactNumBottomBlobs() const override { return 0; }
  int ExactNumTopBlobs() const override { return 2; }

 protected:

  void ShuffleImages();
  void SampleScale(cv::Mat *image, cv::Mat *label);

  void ResizeTo(
      const cv::Mat& img,
      cv::Mat* img_temp,
      const cv::Mat& label,
      cv::Mat* label_temp,
      const cv::Size& size
  );

  virtual void load_batch(Batch<Dtype>* batch);
  
  bool ShareInParallel() const override { return false; }

  vector<shared_ptr<Blob<Dtype>>> transformed_data_, transformed_label_;
  vector<shared_ptr<DataTransformer<Dtype>>> data_transformers_;  
  shared_ptr<Caffe::RNG> prefetch_rng_;  
  vector<std::string> image_lines_;
  vector<std::string> label_lines_;
  int lines_id_;

  int label_margin_h_;
  int label_margin_w_;

  std::mt19937 *rng_;

  int epoch_;
};

} // namspace caffe

#endif //CAFFE_IMAGE_LABEL_DATA_LAYER_H
