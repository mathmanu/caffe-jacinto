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
class DataLayerExtended : public DataLayer<Ftype, Btype> {
public:
  DataLayerExtended(const LayerParameter& param) : DataLayer<Ftype, Btype> (param) {}
  ~DataLayerExtended() {}
};


/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Ftype, typename Btype>
class ImageLabelDataLayer : public BasePrefetchingDataLayer<Ftype, Btype> {
 public:
  explicit ImageLabelDataLayer(const LayerParameter& param);
  virtual ~ImageLabelDataLayer();
  void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

  const char* type() const override { return "ImageLabelData"; }
  int ExactNumBottomBlobs() const override { return 0; }
  int ExactNumTopBlobs() const override { return 2; }
  void start_reading() override {}
  bool ShareInParallel() const override {
    return false;
  }
  Flag* layer_inititialized_flag() override {
    return this->phase_ == TRAIN ? &layer_inititialized_flag_ : nullptr;
  }
  void InitializePrefetch() override;
  void InternalThreadEntryN(size_t thread_id);

 protected:
  void load_batch(Batch<Ftype>* batch, int thread_id, size_t queue_id = 0UL) override;
  //bool is_gpu_transform() const { return true; }

  shared_ptr<DataLayerExtended<Ftype,Btype>> data_layer_, label_layer_;
  Flag layer_inititialized_flag_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
