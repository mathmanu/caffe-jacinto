#include <climits>
#include <vector>

#include "caffe/blob.hpp"

namespace caffe {

size_t Blob::cpu_memory_data_use() const {
  return data_tensor_->cpu_memory_use();
}
size_t Blob::cpu_memory_diff_use() const {
  return diff_tensor_->cpu_memory_use();
}
#ifndef CPU_ONLY
size_t Blob::gpu_memory_data_use() const {
  return data_tensor_->gpu_memory_use();
}
size_t Blob::gpu_memory_diff_use() const {
  return diff_tensor_->gpu_memory_use();
}
#endif

void Blob::Reshape(const int num, const int channels, const int height,
    const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

void Blob::Reshape(const int n) {
  vector<int> shape(1);
  shape[0] = n;
  Reshape(shape);
}

void Blob::Reshape(const vector<int>& shape) {
  std::lock_guard<std::mutex> lock(reshape_mutex_);
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_ = make_shared<SyncedMemory>(shape.size() * sizeof(int));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    if (count_ != 0) {
      CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    }
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }
  if (!data_tensor_) { // might be moved
    data_tensor_ = make_shared<Tensor>(last_data_type_);
  }
  if (!diff_tensor_) { // might be moved
    diff_tensor_ = make_shared<Tensor>(last_diff_type_);
  }
  data_tensor_->Reshape(count_);
  diff_tensor_->Reshape(count_);
  CHECK(is_current_data_valid());
  CHECK(is_current_diff_valid());
  /*
  if(!connectivity_) {
    connectivity_ = make_shared<Tensor>(last_data_type_);
  }
  connectivity_->Reshape(count_); 
  initialize_connectivity();    
  */
}

void Blob::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

#ifndef CPU_ONLY
const int* Blob::gpu_shape() const {
  CHECK(shape_data_);
  return static_cast<const int*>(shape_data_->gpu_data());
}

#endif

void Blob::ShareData(const Blob& other) {
  CHECK_EQ(count(), other.count());
#ifdef DEBUG
#ifndef CPU_ONLY
  const shared_ptr<SyncedMemory>& mem = data_tensor_->synced_mem();
  const shared_ptr<SyncedMemory>& other_mem = other.data_tensor_->synced_mem();
  if (mem && other_mem) {
    const int this_device = mem->gpu_device();
    const int other_device = other_mem->gpu_device();
    if (this_device >= 0 && other_device >= 0) {
      CHECK_EQ(this_device, other_device);
    }
  }
#endif
#endif
  data_tensor_ = other.data_tensor_;
  CHECK(data_type() == other.data_type());
  CHECK(is_current_data_valid());
}

void Blob::ShareDiff(const Blob& other) {
  CHECK_EQ(count(), other.count());
#ifdef DEBUG
#ifndef CPU_ONLY
  const shared_ptr<SyncedMemory>& mem = diff_tensor_->synced_mem();
  const shared_ptr<SyncedMemory>& other_mem = other.diff_tensor_->synced_mem();
  if (mem && other_mem) {
    const int this_device = mem->gpu_device();
    const int other_device = other_mem->gpu_device();
    if (this_device >= 0 && other_device >= 0) {
      CHECK_EQ(this_device, other_device);
    }
  }
#endif
#endif
  diff_tensor_ = other.diff_tensor_;
  CHECK(diff_type() == other.diff_type());
  CHECK(is_current_diff_valid());
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as TBlob<float> or TBlob<double> -- hence we do not define it for
// TBlob<int> or TBlob<unsigned int>.
void Blob::Update() {
  convert_diff(data_type());  // align data&diff types
  shared_ptr<SyncedMemory>& data_mem = data_tensor_->mutable_synced_mem();
  const shared_ptr<SyncedMemory>& diff_mem = diff_tensor_->synced_mem();
  const shared_ptr<SyncedMemory>& connectivity_mem = connectivity_->synced_mem();
  
  // We will perform update based on where the data is located.
  switch (data_mem->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    if(sparse_mode_ != SPARSE_NONE && connectivity_!=NULL) {
      cpu_eltwise_multi(count_, data_type(),
	  		connectivity_mem->cpu_data(), diff_mem->mutable_cpu_data() );	  
    }
    cpu_axpy(count_, data_type(), -1.F,
        diff_mem->cpu_data(), data_mem->mutable_cpu_data());
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    if(sparse_mode_ != SPARSE_NONE && connectivity_!=NULL) {
      gpu_eltwise_multi(count_, data_type(),
	  		connectivity_mem->gpu_data(), diff_mem->mutable_gpu_data() );
    }
    gpu_axpy(count_, data_type(), -1.F,
        diff_mem->gpu_data(), data_mem->mutable_gpu_data());
#else
    NO_GPU;
#endif
    break;
    default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
  CHECK(is_current_data_valid());
  CHECK(is_current_diff_valid());
}

float Blob::at(int offset, Type dtype, const void* data) {
  if (is_type<float>(dtype)) {
    return static_cast<const float*>(data)[offset];
#ifndef CPU_ONLY
  } else if (is_type<float16>(dtype)) {
    return static_cast<const float16*>(data)[offset];
#endif
  } else if (is_type<double>(dtype)) {
    return static_cast<const double*>(data)[offset];
  }
  LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
  return 0.F;
}

float Blob::cpu_sumsq(int count, Type dtype, const void* data) {
  if (is_type<float>(dtype)) {
    return caffe_cpu_dot(count, static_cast<const float*>(data),
        static_cast<const float*>(data));
#ifndef CPU_ONLY
  } else if (is_type<float16>(dtype)) {
    return caffe_cpu_dot(count, static_cast<const float16*>(data),
        static_cast<const float16*>(data));
#endif
  } else if (is_type<double>(dtype)) {
    return caffe_cpu_dot(count, static_cast<const double*>(data),
        static_cast<const double*>(data));
  }
  LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
  return 0.F;
}

#ifndef CPU_ONLY
float Blob::gpu_sumsq(int count, Type dtype, const void* data) {
  if (is_type<float>(dtype)) {
    float sumsq;
    caffe_gpu_dot(count, static_cast<const float*>(data),
        static_cast<const float*>(data), &sumsq);
    return sumsq;
  } else if (is_type<float16>(dtype)) {
    float16 sumsq;
    caffe_gpu_dot(count, static_cast<const float16*>(data),
        static_cast<const float16*>(data), &sumsq);
    return sumsq;
  } else if (is_type<double>(dtype)) {
    double sumsq;
    caffe_gpu_dot(count, static_cast<const double*>(data),
        static_cast<const double*>(data), &sumsq);
    return sumsq;
  }
  LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
  return 0.F;
}
#endif

void Blob::cpu_axpy(int count, Type dtype, float alpha, const void* X, void* Y) {
  if (is_type<float>(dtype)) {
    caffe_axpy(count, alpha, static_cast<const float*>(X),
        static_cast<float*>(Y));
#ifndef CPU_ONLY
  } else if (is_type<float16>(dtype)) {
    caffe_axpy(count, static_cast<float16>(alpha),
        static_cast<const float16*>(X), static_cast<float16*>(Y));
#endif
  } else if (is_type<double>(dtype)) {
    caffe_axpy(count, static_cast<double>(alpha),
        static_cast<const double*>(X), static_cast<double*>(Y));
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
  }
}

#ifndef CPU_ONLY
void Blob::gpu_axpy(int count, Type dtype, float alpha, const void* X, void* Y) {
  if (is_type<float>(dtype)) {
    caffe_gpu_axpy(count, alpha, static_cast<const float*>(X),
        static_cast<float*>(Y));
  } else if (is_type<float16>(dtype)) {
    caffe_gpu_axpy_extfp16(count, static_cast<float>(alpha),
        static_cast<const float16*>(X), static_cast<float16*>(Y));
  } else if (is_type<double>(dtype)) {
    caffe_gpu_axpy(count, static_cast<double>(alpha),
        static_cast<const double*>(X), static_cast<double*>(Y));
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
  }
}
#endif

float Blob::sumsq_data() const {
  const shared_ptr<SyncedMemory>& data_mem = data_tensor_->synced_mem();
  if (!data_mem || count_ <= 0) {
    return 0.F;
  }
  if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
    return gpu_sumsq(count_, data_type(), data_mem->gpu_data());
#else
    NO_GPU;
#endif
  }
  return cpu_sumsq(count_, data_type(), data_mem->cpu_data());
}

float Blob::sumsq_diff() const {
  const shared_ptr<SyncedMemory>& diff_mem = diff_tensor_->synced_mem();
  if (!diff_mem || count_ <= 0) {
    return 0.F;
  }
  if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
    return gpu_sumsq(count_, diff_type(), diff_mem->gpu_data());
#else
    NO_GPU;
#endif
  }
  return cpu_sumsq(count_, diff_type(), diff_mem->cpu_data());
}

float Blob::amax_data() const {
  if (!data_tensor_ || count_ <= 0) {
    return 0.F;
  }
  if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
    return data_tensor_->gpu_amax();
#else
    NO_GPU;
#endif
  }
  return data_tensor_->cpu_amax();
}

float Blob::amax_diff() const {
  if (!diff_tensor_ || count_ <= 0) {
    return 0.F;
  }
  if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
    return diff_tensor_->gpu_amax();
#else
    NO_GPU;
#endif
  }
  return diff_tensor_->cpu_amax();
}

bool Blob::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D TBlob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal TBlob::num(), TBlob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

void Blob::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  const shared_ptr<Tensor>& srct = copy_diff ? source.diff_tensor_ : source.data_tensor_;
  shared_ptr<Tensor>& dstt = copy_diff ? diff_tensor_ : data_tensor_;
  shared_ptr<SyncedMemory>& dst = dstt->mutable_synced_mem();
  if (srct == dstt) {
    return;
  }
  const shared_ptr<SyncedMemory>& src = srct->synced_mem();
  if (src->head() != SyncedMemory::UNINITIALIZED) {
    const bool is_gpu = Caffe::mode() == Caffe::GPU;
    Type src_data_type = copy_diff ? source.diff_type() : source.data_type();
    Type dst_data_type = copy_diff ? diff_type() : data_type();
    Tensor::copy_helper(is_gpu, count_,
        is_gpu ? src->gpu_data() : src->cpu_data(),
        src_data_type,
        is_gpu ? dst->mutable_gpu_data(false) : dst->mutable_cpu_data(false),
        dst_data_type);
    dst->validate();
  }
}

void Blob::FromProto(const BlobProto& proto, bool reshape, bool ignore_shape_mismatch) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D TBlob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else if(!ignore_shape_mismatch) {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  if (proto.double_data_size() > 0) {
    CHECK_EQ(count_, proto.double_data_size());
    for (int i = 0; i < count_; ++i) {
      set_value_at(true, i, proto.double_data(i));
    }
    data_tensor_->invalidate_others();
  } else if (proto.data_size() > 0) {
    CHECK_EQ(count_, proto.data_size());
    for (int i = 0; i < count_; ++i) {
      set_value_at(true, i, proto.data(i));
    }
    data_tensor_->invalidate_others();
  } else if (proto.has_raw_data() > 0) {
    CHECK(proto.has_raw_data_type()) << "Missing raw data type";
    Type raw_type = proto.raw_data_type();
    Type dt = data_tensor_->type();
    const ::std::string& hd = proto.raw_data();
    CHECK_EQ(count_ * tsize(raw_type), hd.size());
    switch (raw_type) {
      case FLOAT:
        caffe_copy<float>(count_, reinterpret_cast<const float*>(&hd.front()),
            mutable_cpu_data<float>());
        break;
#ifndef CPU_ONLY
      case FLOAT16:
        caffe_copy<float16>(count_, reinterpret_cast<const float16*>(&hd.front()),
            mutable_cpu_data<float16>());
        break;
#endif
      case DOUBLE:
        caffe_copy<double>(count_, reinterpret_cast<const double*>(&hd.front()),
            mutable_cpu_data<double>());
        break;
      default:
        LOG(FATAL) << "Unsupported raw type " << Type_Name(raw_type);
    }
    data_tensor_->convert(dt);  // we have to restore its original type
    data_tensor_->invalidate_others();
  }
  // copy diff
  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_diff_size());
    for (int i = 0; i < count_; ++i) {
      set_value_at(false, i, proto.double_diff(i));
    }
    diff_tensor_->invalidate_others();
  } else if (proto.diff_size() > 0) {
    CHECK_EQ(count_, proto.diff_size());
    for (int i = 0; i < count_; ++i) {
      set_value_at(false, i, proto.diff(i));
    }
    diff_tensor_->invalidate_others();
  } else if (proto.has_raw_diff() > 0) {
    CHECK(proto.has_raw_diff_type()) << "Missing raw diff type";
    Type raw_type = proto.raw_diff_type();
    Type dt = diff_tensor_->type();
    const ::std::string& hd = proto.raw_diff();
    CHECK_EQ(count_ * tsize(raw_type), hd.size());
    switch (raw_type) {
      case FLOAT:
        caffe_copy<float>(count_, reinterpret_cast<const float*>(&hd.front()),
            mutable_cpu_diff<float>());
        break;
#ifndef CPU_ONLY
      case FLOAT16:
        caffe_copy<float16>(count_, reinterpret_cast<const float16*>(&hd.front()),
            mutable_cpu_diff<float16>());
        break;
#endif
      case DOUBLE:
        caffe_copy<double>(count_, reinterpret_cast<const double*>(&hd.front()),
            mutable_cpu_diff<double>());
        break;
      default:
        LOG(FATAL) << "Unsupported raw type " << Type_Name(raw_type);
    }
    diff_tensor_->convert(dt);  // we have to restore its original type
    diff_tensor_->invalidate_others();
  }
}

template<typename Dtype>
void Blob::ToProto(BlobProto* proto, bool write_diff) const {
  CHECK(is_current_data_valid());
  CHECK(is_current_diff_valid());
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  const void* pdata = cpu_data<Dtype>();
  Type dt = data_type();
  proto->set_raw_data_type(dt);
  proto->set_raw_data(pdata, count_ * tsize(dt));

  if (write_diff) {
    const void* pdiff = cpu_diff<Dtype>();
    dt = diff_type();
    proto->set_raw_diff_type(dt);
    proto->set_raw_diff(pdiff, count_ * tsize(dt));
  }
}

std::string Blob::to_string(int indent) const {  // debug helper
  const std::string idt(indent, ' ');
  std::ostringstream os;
  os << idt << "Blob " << this << ", count_: " << count_
      << ", data type: " << Type_Name(data_type())
      << ", diff type: " << Type_Name(diff_type()) << std::endl;
  os << idt << "shape_:";
  for (size_t i = 0; i < shape_.size(); ++i) {
    os << " " << shape_[i];
  }
  os << std::endl;
  if (data_tensor_) {
    os << idt << "Data " << data_tensor_->to_string(indent + 2);
  }
  if (diff_tensor_) {
    os << idt << "Diff " << diff_tensor_->to_string(indent + 2);
  }
  os << std::endl;
  return os.str();
}

template void Blob::ToProto<float>(BlobProto*, bool) const;
template void Blob::ToProto<double>(BlobProto*, bool) const;
#ifndef CPU_ONLY
template void Blob::ToProto<float16>(BlobProto*, bool) const;
#endif



void Blob::cpu_eltwise_multi(int count, Type dtype, const void* X, void* Y) {
  if (is_type<float>(dtype)) {
    caffe_cpu_eltwise_multi(count, static_cast<const float*>(X), static_cast<float*>(Y));
#ifndef CPU_ONLY
  } else if (is_type<float16>(dtype)) {
    caffe_cpu_eltwise_multi(count, static_cast<const float16*>(X), static_cast<float16*>(Y));
#endif
  } else if (is_type<double>(dtype)) {
    caffe_cpu_eltwise_multi(count, static_cast<const double*>(X), static_cast<double*>(Y));
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
  }
}

#ifndef CPU_ONLY
void Blob::gpu_eltwise_multi(int count, Type dtype, const void* X, void* Y) {
  if (is_type<float>(dtype)) {
    caffe_gpu_eltwise_multi(count, static_cast<const float*>(X), static_cast<float*>(Y));
  } else if (is_type<float16>(dtype)) {
    caffe_gpu_eltwise_multi(count, static_cast<const float16*>(X), static_cast<float16*>(Y));
  } else if (is_type<double>(dtype)) {
    caffe_gpu_eltwise_multi(count, static_cast<const double*>(X), static_cast<double*>(Y));
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
  }
}
#endif

float Blob::cpu_max(int count, Type dtype, const void* X) const {
  if (is_type<float>(dtype)) {
    return caffe_cpu_max(count, static_cast<const float*>(X));
#ifndef CPU_ONLY
  } else if (is_type<float16>(dtype)) {
    return caffe_cpu_max(count, static_cast<const float16*>(X));
#endif
  } else if (is_type<double>(dtype)) {
    return caffe_cpu_max(count, static_cast<const double*>(X));
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
	return 0;
  }
  return 0;
}

#ifndef CPU_ONLY
float Blob::gpu_max(int count, Type dtype, const void* X) const {
  if (is_type<float>(dtype)) {
    return caffe_gpu_max(count, static_cast<const float*>(X));
#ifndef CPU_ONLY
  } else if (is_type<float16>(dtype)) {
    return caffe_gpu_max(count, static_cast<const float16*>(X));
#endif
  } else if (is_type<double>(dtype)) {
    return caffe_gpu_max(count, static_cast<const double*>(X));
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
	return 0;
  }
  return 0;  
}
#endif

float Blob::max() const {
    const shared_ptr<SyncedMemory>& data_mem = data_tensor_->synced_mem();
	if (!data_tensor_) {
		return 0;
	}
	// We will perform update based on where the data is located.
	switch (data_mem->head()) {
	case SyncedMemory::HEAD_AT_CPU:
	{
		// perform computation on CPU
		auto max_val = cpu_max(this->count(), data_type(), data_mem->cpu_data());
		return max_val;
	}
	case SyncedMemory::HEAD_AT_GPU:
	case SyncedMemory::SYNCED:
	{
#ifndef CPU_ONLY
		// perform computation on GPU
		float max_val = gpu_max(this->count(), data_type(), data_mem->gpu_data());
		return max_val;
#else
		NO_GPU;
#endif
		return 0;
	}
	default:
		LOG(WARNING)<< "Syncedmem not initialized.";
		return 0;
	}
	return 0;
}


float Blob::cpu_min(int count, Type dtype, const void* X) const {
  if (is_type<float>(dtype)) {
    return caffe_cpu_min(count, static_cast<const float*>(X));
#ifndef CPU_ONLY
  } else if (is_type<float16>(dtype)) {
    return caffe_cpu_min(count, static_cast<const float16*>(X));
#endif
  } else if (is_type<double>(dtype)) {
    return caffe_cpu_min(count, static_cast<const double*>(X));
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
	return 0;
  }
  return 0;
}

#ifndef CPU_ONLY
float Blob::gpu_min(int count, Type dtype, const void* X) const {
  if (is_type<float>(dtype)) {
    return caffe_gpu_min(count, static_cast<const float*>(X));
#ifndef CPU_ONLY
  } else if (is_type<float16>(dtype)) {
    return caffe_gpu_min(count, static_cast<const float16*>(X));
#endif
  } else if (is_type<double>(dtype)) {
    return caffe_gpu_min(count, static_cast<const double*>(X));
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
	return 0;
  }
  return 0;
}
#endif

float Blob::min() const {
    const shared_ptr<SyncedMemory>& data_mem = data_tensor_->synced_mem();
	if (!data_tensor_) {
		return 0;
	}
	// We will perform update based on where the data is located.
	switch (data_mem->head()) {
	case SyncedMemory::HEAD_AT_CPU:
	{
		// perform computation on CPU
		auto min_val = cpu_min(this->count(), data_type(), data_mem->cpu_data());
		return min_val;
	}
	case SyncedMemory::HEAD_AT_GPU:
	case SyncedMemory::SYNCED:
	{
#ifndef CPU_ONLY
		// perform computation on GPU
		float min_val = gpu_min(this->count(), data_type(), data_mem->gpu_data());
		return min_val;
#else
		NO_GPU;
		return 0;
#endif
	}
	default:
		LOG(WARNING)<< "Syncedmem not initialized.";
		return 0;
	}
}

void Blob::cpu_if_nonzero(int count, Type dtype, const void* X, void* Y) const {
  if (is_type<float>(dtype)) {
    caffe_cpu_if_nonzero(count, static_cast<const float*>(X), static_cast<float*>(Y));
#ifndef CPU_ONLY
  } else if (is_type<float16>(dtype)) {
    caffe_cpu_if_nonzero(count, static_cast<const float16*>(X), static_cast<float16*>(Y));
#endif
  } else if (is_type<double>(dtype)) {
    caffe_cpu_if_nonzero(count, static_cast<const double*>(X), static_cast<double*>(Y));
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
  }
}

#ifndef CPU_ONLY
void Blob::gpu_if_nonzero(int count, Type dtype, const void* X, void* Y) const {
  if (is_type<float>(dtype)) {
    caffe_gpu_if_nonzero(count, static_cast<const float*>(X), static_cast<float*>(Y));
  } else if (is_type<float16>(dtype)) {
    caffe_gpu_if_nonzero(count, static_cast<const float16*>(X), static_cast<float16*>(Y));
  } else if (is_type<double>(dtype)) {
    caffe_gpu_if_nonzero(count, static_cast<const double*>(X), static_cast<double*>(Y));
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
  }
}
#endif

void Blob::SetSparseMode(const SparseMode mode) {
    CHECK(mode != SPARSE_NONE);
	
    const shared_ptr<SyncedMemory>& data_mem = data_tensor_->synced_mem();
    shared_ptr<SyncedMemory>& connectivity_mem = data_tensor_->mutable_synced_mem();
		
	if (!data_mem) {
		return;
	}
		
    if(!connectivity_) {
      connectivity_ = make_shared<Tensor>(data_type());
    }
    connectivity_->Reshape(count_); 
    
	initialize_connectivity();  
  
	if(mode == SPARSE_UPDATE){
	    switch (data_mem->head()) {
	    case SyncedMemory::HEAD_AT_CPU:
		{
		    cpu_if_nonzero(count_, data_type(), data_mem->cpu_data(), connectivity_mem->mutable_cpu_data());
		    break;
		}
	    case SyncedMemory::HEAD_AT_GPU:
	    case SyncedMemory::SYNCED:
		{
#ifndef CPU_ONLY
		    gpu_if_nonzero(count_, data_type(), data_mem->cpu_data(), connectivity_mem->mutable_gpu_data());
#else
		    NO_GPU;
#endif
		    break;
		}
		default:
			  LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		 }
	}
	sparse_mode_ = mode;
}

int Blob::cpu_count_zero(int count, Type dtype, const void* X, float threshold) const {
  if (is_type<float>(dtype)) {
    return caffe_cpu_count_zero(count, static_cast<const float*>(X), (float)threshold);
#ifndef CPU_ONLY
  } else if (is_type<float16>(dtype)) {
    return caffe_cpu_count_zero(count, static_cast<const float16*>(X), (float16)threshold);
#endif
  } else if (is_type<double>(dtype)) {
    return caffe_cpu_count_zero(count, static_cast<const double*>(X),  (double)threshold);
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
	return 0;
  }
}

#ifndef CPU_ONLY
int Blob::gpu_count_zero(int count, Type dtype, const void* X, float threshold) const {
  if (is_type<float>(dtype)) {
    return caffe_gpu_count_zero(count, static_cast<const float*>(X), (float)threshold);
  } else if (is_type<float16>(dtype)) {
    return caffe_gpu_count_zero(count, static_cast<const float16*>(X), (float16)threshold);
  } else if (is_type<double>(dtype)) {
    return caffe_gpu_count_zero(count, static_cast<const double*>(X), (double)threshold);
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
	return 0;
  }
}
#endif

int Blob::count_zero(float threshold) const {
    const shared_ptr<SyncedMemory>& data_mem = data_tensor_->synced_mem();
	if (!data_mem) {
		return 0;
	}

	// We will perform update based on where the data is located.
	switch (data_mem->head()) {
	case SyncedMemory::HEAD_AT_CPU:
	{
		// perform computation on CPU
		int zero_num = cpu_count_zero(this->count(), data_type(), data_mem->cpu_data(), threshold);
		return zero_num;
	}
	case SyncedMemory::HEAD_AT_GPU:
	case SyncedMemory::SYNCED:
	{
#ifndef CPU_ONLY
		// perform computation on GPU
		int zero_num = gpu_count_zero(this->count(), data_type(), data_mem->gpu_data(), threshold);
		return zero_num;
#else
		NO_GPU;
#endif
		return 0;
	}
	default:
		LOG(WARNING)<< "Syncedmem not initialized.";
		return 0;
	}
}

void Blob::cpu_set(int count, Type dtype, void* X, float val) {
  if (is_type<float>(dtype)) {
    caffe::caffe_set(count, (float)val, static_cast<float*>(X));
#ifndef CPU_ONLY
  } else if (is_type<float16>(dtype)) {
    caffe::caffe_set(count, (float16)val, static_cast<float16*>(X));
#endif
  } else if (is_type<double>(dtype)) {
    caffe::caffe_set(count, (double)val, static_cast<double*>(X));
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
  }
}

#ifndef CPU_ONLY
void Blob::gpu_set(int count, Type dtype, void* X, float val) {
  if (is_type<float>(dtype)) {
    caffe_gpu_set(count, static_cast<float*>(X), (float)val);
  } else if (is_type<float16>(dtype)) {
    caffe_gpu_set(count, static_cast<float16*>(X), (float16)val);
  } else if (is_type<double>(dtype)) {
    caffe_gpu_set(count, static_cast<double*>(X), (double)val);
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
  }
}
#endif


void Blob::initialize_connectivity(float val) {
    const shared_ptr<SyncedMemory>& connectivity_mem = connectivity_->synced_mem();
	if (!connectivity_mem) {
		return;
	}

	// We will perform update based on where the data is located.
	switch (connectivity_mem->head()) {
	case SyncedMemory::HEAD_AT_CPU:
	{
		// perform computation on CPU
		cpu_set(this->count(), data_type(), connectivity_mem->mutable_cpu_data(), val);
		break;
	}
	case SyncedMemory::HEAD_AT_GPU:
	case SyncedMemory::SYNCED:
	{
#ifndef CPU_ONLY
		// perform computation on GPU
		gpu_set(this->count(), data_type(), connectivity_mem->mutable_gpu_data(), val);
#else
		NO_GPU;
#endif
		break;
	}
	default:
		LOG(WARNING)<< "Syncedmem not initialized.";
		return;
	}
}

void Blob::cpu_zerout(int count, Type dtype, const void* X, void* Y, float threshold) {
  if (is_type<float>(dtype)) {
    caffe_cpu_zerout(count, static_cast<const float*>(X), static_cast<float*>(Y), (float)threshold);
#ifndef CPU_ONLY
  } else if (is_type<float16>(dtype)) {
    caffe_cpu_zerout(count, static_cast<const float16*>(X), static_cast<float16*>(Y), (float16)threshold);
#endif
  } else if (is_type<double>(dtype)) {
    caffe_cpu_zerout(count, static_cast<const double*>(X), static_cast<double*>(Y), (double)threshold);
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
  }
}

#ifndef CPU_ONLY
void Blob::gpu_zerout(int count, Type dtype, const void* X, void* Y, float threshold) {
  if (is_type<float>(dtype)) {
    caffe_gpu_zerout(count, static_cast<const float*>(X), static_cast<float*>(Y), (float)threshold);
  } else if (is_type<float16>(dtype)) {
    caffe_gpu_zerout(count, static_cast<const float16*>(X), static_cast<float16*>(Y), (float16)threshold);
  } else if (is_type<double>(dtype)) {
    caffe_gpu_zerout(count, static_cast<const double*>(X), static_cast<double*>(Y), (double)threshold);
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
  }
}
#endif

void Blob::zerout(float threshold) {
    const shared_ptr<SyncedMemory>& connectivity_mem = connectivity_->synced_mem();
	if (!connectivity_mem) {
		return;
	}

	// We will perform update based on where the data is located.
	switch (connectivity_mem->head()) {
	case SyncedMemory::HEAD_AT_CPU:
	{
		// perform computation on CPU
		cpu_zerout(this->count(), data_type(), connectivity_mem->cpu_data(), connectivity_mem->mutable_cpu_data(), threshold);
		break;
	}
	case SyncedMemory::HEAD_AT_GPU:
	case SyncedMemory::SYNCED:
	{
#ifndef CPU_ONLY
		// perform computation on GPU
		gpu_zerout(this->count(), data_type(), connectivity_mem->gpu_data(), connectivity_mem->mutable_gpu_data(), threshold);
#else
		NO_GPU;
#endif
		break;
	}
	default:
		LOG(WARNING)<< "Syncedmem not initialized.";
		return;
	}
}


INSTANTIATE_CLASS(TBlob);

// we need full matrix of instantiations for blob
template class TBlob<int>;
template class TBlob<unsigned int>;

}  // namespace caffe
