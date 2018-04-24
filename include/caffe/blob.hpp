#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/tensor.hpp"
#include "caffe/type.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"

const int kMaxBlobAxes = 32;

namespace caffe {

template<typename Dtype>
class TBlob;

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 *        This is template-less implementation made for mixed precision.
 *        Instances can be converted to any other supported Type.
 *
 * TODO(dox): more thorough description.
 */
class Blob {
 public:
  void Swap(Blob& other) noexcept {
    std::swap(data_tensor_, other.data_tensor_);
    std::swap(diff_tensor_, other.diff_tensor_);
    std::swap(connectivity_, other.connectivity_);	
    std::swap(shape_data_, other.shape_data_);
    std::swap(shape_, other.shape_);
    std::swap(count_, other.count_);
  }

 protected:
  Blob(Type data_type, Type diff_type)
      : data_tensor_(make_shared<Tensor>(data_type)),
        diff_tensor_(make_shared<Tensor>(diff_type)),
        count_(0) {}
  explicit Blob(Type dtype)
      : Blob(dtype, dtype) {}

 public:
  virtual ~Blob() {}

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int channels, const int height, const int width);
  void Reshape(const int num);

  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  void Reshape(const vector<int>& shape);
  void Reshape(const BlobShape& shape);

  void ReshapeLike(const Blob* other) {
// TODO   if (this->shape() != other->shape()) {
      Reshape(other->shape());
//    }
  }

  void ReshapeLike(const Blob& other) {
    ReshapeLike(&other);
  }

  Type data_type() const {
    return data_tensor_->type();
  }

  Type diff_type() const {
    return diff_tensor_->type();
  }

  bool diff_equals(const Blob& other) const {
    return diff_tensor_ == other.diff_tensor_;
  }

  void allocate_data(bool on_gpu = true) {
    data_tensor_->current_memory(on_gpu);
  }

  void allocate_diff(bool on_gpu = true) {
    diff_tensor_->current_memory(on_gpu);
  }

  /**
   * @brief Creates an instance of a Blob with given Dtype.
   */
  template<typename D, typename DI = D>
  static shared_ptr<Blob> create() {
    return shared_ptr<Blob>(new Blob(tp<D>(), tp<DI>()));
  }

  /**
   * @brief Creates an instance of a Blob with given Type.
   */
  static shared_ptr<Blob> create(Type data_type, Type diff_type) {
    return shared_ptr<Blob>(new Blob(data_type, diff_type));
  }

  /// @brief Creates an instance of a Blob with given type Dtype and given shape.
  template<typename D, typename DI = D>
  static shared_ptr<Blob> create(const vector<int>& shape) {
    shared_ptr<Blob> ptr = create<D, DI>();
    ptr->Reshape(shape);
    return ptr;
  }

  template<typename D, typename DI = D>
  static shared_ptr<Blob> create(int N) {
    shared_ptr<Blob> ptr = create<D, DI>();
    vector<int> shape;
    shape.push_back(N);
    ptr->Reshape(shape);
    return ptr;
  }


  /// @brief Deprecated; use <code>create(const vector<int>& shape)</code>.
  template<typename D, typename DI = D>
  static shared_ptr<Blob> create(int num, int channels, int height, int width) {
    shared_ptr<Blob> ptr = create<D, DI>();
    ptr->Reshape(num, channels, height, width);
    return ptr;
  }

  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  void CopyFrom(const Blob& source, bool copy_diff = false, bool reshape = false,
      Packing pk_from = NCHW, Packing pk_to = NCHW, int group = 0);

  void CopyDataFrom(const Blob& source, bool reshape = false,
      Packing pk_from = NCHW, Packing pk_to = NCHW, int group = 0) {
    CopyFrom(source, false, reshape, pk_from, pk_to, group);
  }

  void CopyDiffFrom(const Blob& source, bool reshape = false,
      Packing pk_from = NCHW, Packing pk_to = NCHW, int group = 0) {
    CopyFrom(source, true, reshape, pk_from, pk_to, group);
  }

  bool is_data_empty() const {
    return data_tensor_->is_empty();
  }

  bool is_diff_empty() const {
    return diff_tensor_->is_empty();
  }

  std::string shape_string() const {
    std::ostringstream stream;
    for (size_t i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }

  const vector<int>& shape() const { return shape_; }

  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }

  int num_axes() const {
    return shape_.size();
  }

  int count() const {
    return count_;
  }

  size_t sizeof_data() const {
    return data_tensor_->size_of();
  }

  size_t sizeof_diff() const {
    return diff_tensor_->size_of();
  }

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }

  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes()) << "axis " << axis_index << " out of range for " << num_axes()
                                      << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes()) << "axis " << axis_index << " out of range for " << num_axes()
                                     << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  int num() const { return LegacyShape(0); }

  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  int channels() const { return LegacyShape(1); }

  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  int height() const { return LegacyShape(2); }

  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  int width() const { return LegacyShape(3); }

  int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4) << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }

  size_t offset(size_t n, size_t c = 0, size_t h = 0, size_t w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }

  int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if ((int)indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }

  template<typename Dtype>
  void set_cpu_data(Dtype* data) {
    CHECK_NOTNULL(data);
    convert_data(tp<Dtype>());
    CHECK(is_type<Dtype>(data_type()));
    data_tensor_->mutable_synced_mem()->set_cpu_data(data);
  }

  template<typename Dtype>
  void set_cpu_diff(Dtype* diff) {
    CHECK_NOTNULL(diff);
    convert_diff(tp<Dtype>());
    CHECK(is_type<Dtype>(diff_type()));
    diff_tensor_->mutable_synced_mem()->set_cpu_data(diff);
  }

  template<typename Dtype>
  const Dtype* cpu_data() const {
    convert_data(tp<Dtype>());
    return static_cast<const Dtype*>(data_tensor_->synced_mem()->cpu_data());
  }

  template<typename Dtype>
  const Dtype* cpu_diff() const {
    convert_diff(tp<Dtype>());
    return static_cast<const Dtype*>(diff_tensor_->synced_mem()->cpu_data());
  }

  template<typename Dtype>
  Dtype* mutable_cpu_data_c(bool copy_from_gpu) {
    convert_data(tp<Dtype>());
    return static_cast<Dtype*>(data_tensor_->mutable_synced_mem()->mutable_cpu_data(copy_from_gpu));
  }

  template<typename Dtype>
  Dtype* mutable_cpu_data() {  // Keeping PyCaffe intact
    return mutable_cpu_data_c<Dtype>(true);
  }

  template<typename Dtype>
  Dtype* mutable_cpu_diff_c(bool copy_from_gpu) {
    convert_diff(tp<Dtype>());
    return static_cast<Dtype*>(diff_tensor_->mutable_synced_mem()->mutable_cpu_data(copy_from_gpu));
  }

  template<typename Dtype>
  Dtype* mutable_cpu_diff() {
    return mutable_cpu_diff_c<Dtype>(true);
  }

  // Element-wise accessor. Might be slow due to syncing from GPU to CPU.
  // Currently it's used in tests only. We better keep it this way.
  float data_at(const int n, const int c, const int h, const int w) const {
    return at(offset(n, c, h, w), data_type(), data_tensor_->synced_mem()->cpu_data());
  }
  float diff_at(const int n, const int c, const int h, const int w) const {
    return at(offset(n, c, h, w), diff_type(), diff_tensor_->synced_mem()->cpu_data());
  }
  float data_at(const vector<int>& index) const {
    return at(offset(index), data_type(), data_tensor_->synced_mem()->cpu_data());
  }
  float diff_at(const vector<int>& index) const {
    return at(offset(index), diff_type(), diff_tensor_->synced_mem()->cpu_data());
  }
  float data_at(int index) const {
    return at(index, data_type(), data_tensor_->synced_mem()->cpu_data());
  }
  float diff_at(int index) const {
    return at(index, diff_type(), diff_tensor_->synced_mem()->cpu_data());
  }

  void Update();

  /// @brief Compute the maximum of absolute values (L_\infinity norm) of the data.
  float amax_data(int group = 0) const {
    return data_tensor_->amax(group);
  }

  /// @brief Compute the maximum of absolute values (L_\infinity norm) of the diff.
  float amax_diff(int group = 0) const {
    return diff_tensor_->amax(group);
  }

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  float asum_data(int group = 0) const {
    return data_tensor_->asum(group);
  }

  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  float asum_diff(int group = 0) const {
    return diff_tensor_->asum(group);
  }

  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  float sumsq_data(int group = 0) const {
    return data_tensor_->sumsq(group);
  }

  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  float sumsq_diff(int group = 0) const {
    return diff_tensor_->sumsq(group);
  }

  /// @brief Scale the blob data by a constant factor.
  void scale_data(float scale, void* handle = nullptr) {
    data_tensor_->scale(scale, handle);
  }

  /// @brief Scale the blob diff by a constant factor.
  void scale_diff(float scale, void* handle = nullptr) {
    diff_tensor_->scale(scale, handle);
  }

  /// @brief Set all the blob's data elements to a value.
  void set_data(float value) {
    data_tensor_->set(value);
  }

  /// @brief Set all the blob's diff elements to a value.
  void set_diff(float value) {
    diff_tensor_->set(value);
  }

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);

  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const Blob& other);

  void ToProto(BlobProto* proto, bool store_in_old_format, bool write_diff = false, bool write_data = true) const;
  void ToProtoBVLC(BlobProto* proto, bool write_diff = false, bool write_data = true) const;

  void FromProto(const BlobProto& proto, bool reshape = true, bool ignore_shape_mismatch = false);
  bool ShapeEquals(const BlobProto& other);
  std::string to_string(int indent = 0) const;  // debug helper

  // These ones are to be used with care: they don't convert.
  void* current_mutable_data_memory(bool is_gpu, bool flush = true) {
    return data_tensor_->current_mutable_memory(is_gpu, flush);
  }

  void* current_mutable_diff_memory(bool is_gpu, bool flush = true) {
    return diff_tensor_->current_mutable_memory(is_gpu, flush);
  }

  const void* current_data_memory(bool is_gpu) const {
    return data_tensor_->current_memory(is_gpu);
  }

  const void* current_diff_memory(bool is_gpu) const {
    return diff_tensor_->current_memory(is_gpu);
  }

  bool is_data_on_gpu() const {
    return data_tensor_->is_gpu_head();
  }

  bool is_diff_on_gpu() const {
    return diff_tensor_->is_gpu_head();
  }

  size_t gpu_memory_data_use(bool own_only = false) const;
  size_t gpu_memory_diff_use(bool own_only = false) const;

  void set_gpu_data(void* data) {
    CHECK_NOTNULL(data);
    data_tensor_->mutable_synced_mem()->set_gpu_data(data);
  }

  void set_gpu_diff(void* diff) {
    CHECK_NOTNULL(diff);
    diff_tensor_->mutable_synced_mem()->set_gpu_data(diff);
  }

  template<typename Dtype>
  const Dtype* gpu_data() const {
    convert_data(tp<Dtype>());
    return static_cast<const Dtype*>(data_tensor_->synced_mem()->gpu_data());
  }

  template<typename Dtype>
  const Dtype* gpu_diff() const {
    convert_diff(tp<Dtype>());
    return static_cast<const Dtype*>(diff_tensor_->synced_mem()->gpu_data());
  }

  template<typename Dtype>
  Dtype* mutable_gpu_data_c(bool copy_from_cpu) {
    convert_data(tp<Dtype>());
    return static_cast<Dtype*>(data_tensor_->mutable_synced_mem()->mutable_gpu_data(copy_from_cpu));
  }

  template<typename Dtype>
  Dtype* mutable_gpu_data() {  // Keeping PyCaffe intact
    return mutable_gpu_data_c<Dtype>(true);
  }

  template<typename Dtype>
  Dtype* mutable_gpu_diff_c(bool copy_from_cpu) {
    convert_diff(tp<Dtype>());
    return static_cast<Dtype*>(diff_tensor_->mutable_synced_mem()->mutable_gpu_data(copy_from_cpu));
  }

  template<typename Dtype>
  Dtype* mutable_gpu_diff() {
    return mutable_gpu_diff_c<Dtype>(true);
  }

  const int* gpu_shape() const;

  // Element-wise mutator. Might be slow due to syncing from GPU to CPU.
  template<typename Dtype>
  void set_value_at(bool set_data, int idx, Dtype val) {
    void* ptr = set_data ? current_mutable_data_memory(false, false) :
                current_mutable_diff_memory(false, false);
    CHECK_NOTNULL(ptr);
    const Type dtype = set_data ? data_type() : diff_type();
    if (is_type<float>(dtype)) {
      static_cast<float*>(ptr)[idx] = static_cast<float>(val);
    } else if (is_type<float16>(dtype)) {
      static_cast<float16*>(ptr)[idx] = static_cast<float16>(val);
    } else if (is_type<double>(dtype)) {
      static_cast<double*>(ptr)[idx] = static_cast<double>(val);
    } else {
      LOG(FATAL) << "Unknown data or diff: " << Type_Name(dtype);
    }
  }
  
  void cpu_eltwise_multi(int count, Type dtype, const void* X, void* Y);
  void gpu_eltwise_multi(int count, Type dtype, const void* X, void* Y);
  float cpu_max(int count, Type dtype, const void* X, const int start_index) const;
  float gpu_max(int count, Type dtype, const void* X, const int start_index) const;
  float max(const int start_index, const int count) const;
  
  float cpu_min(int count, Type dtype, const void* X, const int start_index) const;
  float gpu_min(int count, Type dtype, const void* X, const int start_index) const;
  float min(const int start_index, const int count) const;
    
  int cpu_count_zero(int count, Type dtype, const void* X, float threshold, const int start_index) const;
  int gpu_count_zero(int count, Type dtype, const void* X, float threshold, const int start_index) const;
  int count_zero(float threshold, const int start_index, const int count) const;
  int count_zero_connectivity(float threshold, const int start_index, const int count) const;
  void cpu_if_nonzero(int count, Type dtype, const void* X, void* Y) const;
  void gpu_if_nonzero(int count, Type dtype, const void* X, void* Y) const;
  void cpu_set(int count, Type dtype, void* X, float val);
  void gpu_set(int count, Type dtype, void* X, float val);
  
  void cpu_zerout(int count, Type dtype, const void* X, void* Y, float threshold, const int start_index);
  void gpu_zerout(int count, Type dtype, const void* X, void* Y, float threshold, const int start_index);
  void zerout(float threshold, const int start_index, const int count);

  //For sparse operation
  inline const shared_ptr<Tensor>& connectivity() const {
    return connectivity_;
  }
  void InitializeConnectivity(float val = 1.0);
  void ComputeSparseDiff();
  void ComputeSparseData();
  void StoreSparseModeConnectivity(const SparseMode mode);
  
  DISABLE_COPY_MOVE_AND_ASSIGN(Blob);

 protected:
  mutable shared_ptr<Tensor> data_tensor_;
  mutable shared_ptr<Tensor> diff_tensor_;
  shared_ptr<SyncedMemory> shape_data_;
  vector<int> shape_;
  int count_;

  //For sparse operation  
  mutable shared_ptr<Tensor> connectivity_;

  bool is_current_data_valid() const {
    return data_tensor_->is_current_valid();
  }

  bool is_current_diff_valid() const {
    return diff_tensor_->is_current_valid();
  }

  void convert_data(Type new_data_type) const {
    data_tensor_->convert(new_data_type);
  }

  void convert_diff(Type new_diff_type) const {
    diff_tensor_->convert(new_diff_type);
  }

  static float at(int offset, Type dtype, const void* data);
  static void cpu_axpy(int count, Type dtype, float alpha, const void* X, void* Y);
  static void gpu_axpy(int count, Type dtype, float alpha, const void* X, void* Y);

  static void check_integrity(bool do_data, Type current_type, Type new_type) {
    CHECK_EQ(current_type, new_type)
      << "Attempt to change TBlob native " << (do_data ? "data" : "diff")
      << " type from " << Type_Name(current_type) << " to " << Type_Name(new_type);
  }
};  // class Blob


/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 *        This is template implementation made for simpler instantiation.
 *        Instances can be converted to any other supported Type.
 *
 * TODO(dox): more thorough description.
 */
template<typename Dtype>
class TBlob : public Blob {
 public:
  TBlob()
      : Blob(tp<Dtype>()) {}

  /// @brief Deprecated; use <code>TBlob(const vector<int>& shape)</code>.
  TBlob(const int num, const int channels, const int height, const int width)
      : Blob(tp<Dtype>()) {
    Reshape(num, channels, height, width);
  }

  explicit TBlob(const vector<int>& shape)
      : Blob(tp<Dtype>()) {
    Reshape(shape);
  }

  // Shadowing parent's implementations and calling them with Dtype by default.
  // We might get rid of shadowing but overall changes would be too dramatic
  // in this case. These are the shortcuts allowing to keep current
  // code intact (pretty much).
  template<typename T = Dtype>
  const T* cpu_data() const {
    check_integrity(true, data_type(), tp<T>());
    return Blob::cpu_data<T>();
  }

  template<typename T = Dtype>
  const T* cpu_diff() const {
    check_integrity(false, diff_type(), tp<T>());
    return Blob::cpu_diff<T>();
  }

  template<typename T = Dtype>
  T* mutable_cpu_data(bool copy_from_gpu = true) {
    check_integrity(true, data_type(), tp<T>());
    return Blob::mutable_cpu_data_c<T>(copy_from_gpu);
  }

  template<typename T = Dtype>
  T* mutable_cpu_diff(bool copy_from_gpu = true) {
    check_integrity(false, diff_type(), tp<T>());
    return Blob::mutable_cpu_diff_c<T>(copy_from_gpu);
  }

  template<typename T = Dtype>
  const T* gpu_data() const {
    check_integrity(true, data_type(), tp<T>());
    return Blob::gpu_data<T>();
  }

  template<typename T = Dtype>
  const T* gpu_diff() const {
    check_integrity(false, diff_type(), tp<T>());
    return Blob::gpu_diff<T>();
  }

  template<typename T = Dtype>
  T* mutable_gpu_data(bool copy_from_cpu = true) {
    check_integrity(true, data_type(), tp<T>());
    return Blob::mutable_gpu_data_c<T>(copy_from_cpu);
  }

  template<typename T = Dtype>
  T* mutable_gpu_diff(bool copy_from_cpu = true) {
    check_integrity(false, diff_type(), tp<T>());
    return Blob::mutable_gpu_diff_c<T>(copy_from_cpu);
  }

  DISABLE_COPY_MOVE_AND_ASSIGN(TBlob);
};  // class TBlob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
