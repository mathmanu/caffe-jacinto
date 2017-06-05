#ifndef CAFFE_IMAGE_LABEL_LIST_DATA_LAYER_H
#define CAFFE_IMAGE_LABEL_LIST_DATA_LAYER_H

#include <random>
#include <vector>

#include <opencv2/core/core.hpp>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Ftype, typename Btype>
class ImageLabelListDataLayer : public BasePrefetchingDataLayer<Ftype, Btype> {
 public:
  explicit ImageLabelListDataLayer(const LayerParameter &param);
  virtual ~ImageLabelListDataLayer();
  void DataLayerSetUp(const vector<Blob*> &bottom, const vector<Blob*> &top) override;

  const char *type() const override { return "ImageLabelData"; }
  int ExactNumBottomBlobs() const override { return 0; }
  int ExactNumTopBlobs() const override { return 2; }
  bool is_gpu_transform() const override { return false; }

 protected:
  void InternalThreadEntryN(size_t thread_id) override;
  
  void ShuffleImages();
  void SampleScale(cv::Mat *image, cv::Mat *label);
  void ResizeTo(
      const cv::Mat& img,
      cv::Mat* img_temp,
      const cv::Mat& label,
      cv::Mat* label_temp,
      const cv::Size& size
  );

  void load_batch(Batch<Ftype>* batch, int thread_id, size_t queue_id) override;
  void start_reading() override {
    DLOG(INFO) << "[" << this->target_device_ << "] Start Reading.";  
  }
  
  bool ShareInParallel() const override {
    return false;
  }

  vector<shared_ptr<TBlob<Ftype>>> transformed_data_, transformed_label_;
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

#endif //CAFFE_IMAGE_LABEL_LIST_DATA_LAYER_H
