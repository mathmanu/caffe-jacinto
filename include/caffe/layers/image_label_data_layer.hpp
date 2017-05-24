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

template<typename Ftype, typename Btype>
class ImageLabelDataLayer : public BasePrefetchingDataLayer<Ftype, Btype> {
 public:

  explicit ImageLabelDataLayer(const LayerParameter &param);

  virtual ~ImageLabelDataLayer();

  virtual void DataLayerSetUp(const vector<Blob*> &bottom,
                              const vector<Blob*> &top);

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

  void ResizeTo(
      const cv::Mat& img,
      cv::Mat* img_temp,
      const cv::Mat& label,
      cv::Mat* label_temp,
      const cv::Size& size
  );

  virtual void load_batch(Batch<Ftype>* batch, int thread_id, size_t queue_id) override;
  void start_reading() override {}
  vector<std::string> image_lines_;
  vector<std::string> label_lines_;
  int lines_id_;

  TBlob<Ftype> transformed_data_, transformed_label_;
  int label_margin_h_;
  int label_margin_w_;

  std::mt19937 *rng_;
};

} // namspace caffe

#endif //CAFFE_IMAGE_LABEL_DATA_LAYER_H
