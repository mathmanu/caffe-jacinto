#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>
#include <mutex>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/batch_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/quantized_layer.hpp"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template<typename Ftype, typename Btype>
class BaseDataLayer : public QuantizedLayer<Ftype, Btype> {
 public:
  BaseDataLayer(const LayerParameter& param, size_t transf_num);
  virtual ~BaseDataLayer() {}
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  virtual void DataLayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) = 0;

  // Data layers should be shared by multiple solvers in parallel
  bool ShareInParallel() const override { return true; }

  // Data layers have no bottoms, so reshaping is trivial.
  void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) override {}

  void Backward_cpu(const vector<Blob*>& top, const vector<bool>& propagate_down,
      const vector<Blob*>& bottom) override {}
  void Backward_gpu(const vector<Blob*>& top, const vector<bool>& propagate_down,
      const vector<Blob*>& bottom) override {}

  virtual bool is_gpu_transform() const { return false; }

 protected:
  TransformationParameter transform_param_;
  bool output_labels_;
};

template<typename Ftype, typename Btype>
class BasePrefetchingDataLayer : public BaseDataLayer<Ftype, Btype>, public InternalThread {
 public:
  BasePrefetchingDataLayer(const LayerParameter& param, size_t solver_rank);
  virtual ~BasePrefetchingDataLayer();
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

  DataTransformer<Btype>* bdt(int id) {
    return bwd_data_transformers_.at(id).get();
  }
  DataTransformer<Ftype>* fdt(int id) {
    return fwd_data_transformers_.at(id).get();
  }

  bool is_gpu_transform() const override {
    // If user omitted this setting we deduce it from Ftype
    const bool use_gpu = this->transform_param_.use_gpu_transform();
    const bool use_rand_resize =
        this->transform_param_.has_img_rand_resize_lower() ||
        this->transform_param_.has_img_rand_resize_upper() ||
        this->transform_param_.has_rand_resize_ratio_lower() ||
        this->transform_param_.has_rand_resize_ratio_upper() ||
        this->transform_param_.has_vertical_stretch_lower() ||
        this->transform_param_.has_vertical_stretch_upper() ||
        this->transform_param_.has_horizontal_stretch_lower() ||
        this->transform_param_.has_horizontal_stretch_upper();
    return use_gpu && Caffe::mode() == Caffe::GPU && !use_rand_resize;
  }

  int batch_size() const {
    return batch_size_;
  }

 protected:
  void InternalThreadEntry() override;
  void InternalThreadEntryN(size_t thread_id) override;

  virtual void ResizeQueues();
  virtual void InitializePrefetch();
  virtual void load_batch(Batch* batch, int thread_id, size_t queue_id) = 0;
  virtual void start_reading() = 0;
  virtual size_t queue_id(size_t thread_id) const {
    return thread_id;
  }
  virtual bool auto_mode() const {
    return auto_mode_;
  }

  size_t batch_id(int thread_id) {
    size_t id = batch_ids_[thread_id];
    batch_ids_[thread_id] += this->threads_num();
    return id;
  }

  static size_t threads(const LayerParameter& param);
  static size_t parser_threads(const LayerParameter& param);
  static bool auto_mode(const LayerParameter& param);

  std::vector<size_t> batch_ids_;
  bool auto_mode_;
  size_t parsers_num_, transf_num_, queues_num_;

  // These two are for delayed init only
  std::vector<Blob*> bottom_init_;
  std::vector<Blob*> top_init_;

  // Use Btype as a transformer type (i.e. better float 32)
  // since data layers don't have bottom channels
  std::vector<shared_ptr<DataTransformer<Btype>>> bwd_data_transformers_;
  // TransformGPU may do this in-place
  std::vector<shared_ptr<DataTransformer<Ftype>>> fwd_data_transformers_;

  shared_ptr<BatchTransformer<Ftype, Btype>> batch_transformer_;
  std::vector<int> last_shape_;
  int batch_size_;
  Flag iter0_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
