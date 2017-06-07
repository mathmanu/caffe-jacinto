#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/image_label_list_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include "caffe/util/parallel.hpp"

namespace {

  cv::Mat PadImage(cv::Mat &image, int min_size, double value = -1) {
    if (image.rows >= min_size && image.cols >= min_size) {
      return image;
    }
    int top, bottom, left, right;
    top = bottom = left = right = 0;
    if (image.rows < min_size) {
      top = (min_size - image.rows) / 2;
      bottom = min_size - image.rows - top;
    }

    if (image.cols < min_size) {
      left = (min_size - image.cols) / 2;
      right = min_size - image.cols - left;
    }
    cv::Mat big_image;
    if (value < 0) {
      cv::copyMakeBorder(image, big_image, top, bottom, right, left,
          cv::BORDER_REFLECT_101);
    } else {
      cv::copyMakeBorder(image, big_image, top, bottom, right, left,
          cv::BORDER_CONSTANT, cv::Scalar(value));
    }
    return big_image;
  }

  cv::Mat ExtendLabelMargin(cv::Mat &image, int margin_w, int margin_h,
      double value = -1) {
    cv::Mat big_image;
    if (value < 0) {
      cv::copyMakeBorder(image, big_image, margin_h, margin_h, margin_w, margin_w,
          cv::BORDER_REFLECT_101);
    } else {
      cv::copyMakeBorder(image, big_image, margin_h, margin_h, margin_w, margin_w,
          cv::BORDER_CONSTANT, cv::Scalar(value));
    }
    return big_image;
  }

  template<typename Dtype>
  void GetLabelSlice(const Dtype *labels, int rows, int cols,
      const caffe::Slice &label_slice, Dtype *slice_data) {
    // for (int c = 0; c < channels; ++c) {
    labels += label_slice.offset(0) * cols;
    for (int h = 0; h < label_slice.dim(0); ++h) {
      labels += label_slice.offset(1);
      for (int w = 0; w < label_slice.dim(1); ++w) {
        slice_data[w] = labels[w * label_slice.stride(1)];
      }
      labels += cols * label_slice.stride(0) - label_slice.offset(1);
      slice_data += label_slice.dim(1);
    }
    //t_label_data += this->label_margin_h_ * (label_width + this->label_margin_w_ * 2);
    // }
  }

}

namespace caffe {

  template<typename Ftype, typename Btype>
  ImageLabelListDataLayer<Ftype, Btype>::ImageLabelListDataLayer(
      const LayerParameter &param) : BasePrefetchingDataLayer<Ftype, Btype>(param) {
    std::random_device rand_dev;
    rng_ = new std::mt19937(rand_dev());

    //Hang is observed when using default number of threads. Limit the number
    bool has_threads = this->layer_param_.image_label_data_param().has_threads();
    int input_threads = this->layer_param_.image_label_data_param().threads();
    int threads = has_threads? std::min<int>(std::max<int>(input_threads, 1), 4) : 2;
    this->layer_param_.mutable_data_param()->set_threads(threads);
    this->layer_param_.mutable_data_param()->set_parser_threads(threads);
  }

  template<typename Ftype, typename Btype>
  ImageLabelListDataLayer<Ftype, Btype>::~ImageLabelListDataLayer() {
    this->StopInternalThread();
    delete rng_;
  }

  template<typename Ftype, typename Btype>
  void ImageLabelListDataLayer<Ftype, Btype>::InternalThreadEntryN(size_t thread_id) {
#ifndef CPU_ONLY
    const bool use_gpu_transform = this->is_gpu_transform();
#endif
    static thread_local bool iter0 = this->phase_ == TRAIN;
    if (iter0 && this->net_inititialized_flag_ != nullptr) {
      this->net_inititialized_flag_->wait();
    } else {  // nothing to wait -> initialize and start pumping
      std::lock_guard<std::mutex> lock(this->mutex_in_);
      this->InitializePrefetch();
      start_reading();
      iter0 = false;
    }
    try {
      while (!this->must_stop(thread_id)) {
        const size_t qid = this->queue_id(thread_id);
#ifndef CPU_ONLY
        shared_ptr<Batch<Ftype>> batch = this->prefetches_free_[qid]->pop();

        CHECK_EQ((size_t) -1, batch->id());
        load_batch(batch.get(), thread_id, qid);
        if (Caffe::mode() == Caffe::GPU) {
          if (!use_gpu_transform) {
            batch->data_.async_gpu_push();
          }
          if (this->output_labels_) {
            batch->label_.async_gpu_push();
          }
          CUDA_CHECK(cudaStreamSynchronize(Caffe::th_stream_aux(Caffe::STREAM_ID_ASYNC_PUSH)));
        }

        this->prefetches_full_[qid]->push(batch);
#else
        shared_ptr<Batch<Ftype>> batch = this->prefetches_free_[qid]->pop();
        load_batch(batch.get(), thread_id, qid);
        this->prefetches_full_[qid]->push(batch);
#endif

        if (iter0) {
          if (this->net_iteration0_flag_ != nullptr) {
            this->net_iteration0_flag_->wait();
          }
          std::lock_guard<std::mutex> lock(this->mutex_out_);
          if (this->net_inititialized_flag_ != nullptr) {
            this->net_inititialized_flag_ = nullptr;  // no wait on the second round
            this->InitializePrefetch();
            start_reading();
          }
          //if (this->auto_mode_) {
          //  break;
          //}  // manual otherwise, thus keep rolling
          iter0 = false;
        }
      }
    } catch (boost::thread_interrupted&) {
      LOG(INFO) << "InternalThreadEntryN was interrupted" << std::endl;
    }
  }

  bool static inline file_exists(const std::string filename) {
    std::ifstream ifile(filename);
    return ifile.good();
  }

  template<typename Ftype, typename Btype>
  void ImageLabelListDataLayer<Ftype, Btype>::DataLayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
    epoch_ = 0;

    auto &data_param = this->layer_param_.image_label_data_param();
    string data_dir = data_param.data_dir();
    string image_dir = data_param.image_dir();
    string label_dir = data_param.label_dir();

    if (image_dir == "" && label_dir == "" && data_dir != "") {
      image_dir = data_dir;
      label_dir = data_dir;
    }

    // Read the file with filenames and labels
    const string& image_list_path =
    this->layer_param_.image_label_data_param().image_list_path();
    LOG(INFO) << "Opening image list " << image_list_path;
    std::ifstream infile(image_list_path.c_str());
    CHECK(infile.good() == true);

    image_lines_.clear();
    string filename;
    while (infile >> filename) {
      image_lines_.push_back(filename);
    }

    const string& label_list_path =
    this->layer_param_.image_label_data_param().label_list_path();
    LOG(INFO) << "Opening label list " << label_list_path;
    std::ifstream in_label(label_list_path.c_str());
    CHECK(in_label.good() == true);

    label_lines_.clear();
    while (in_label >> filename) {
      label_lines_.push_back(filename);
    }

    vector<bool> ignore_file(image_lines_.size(), false);
    auto check_image_file_func = [&](int i) {
      cv::Mat check_img = cv::imread(image_lines_[i]);
      cv::Mat check_lbl = cv::imread(label_lines_[i]);
      if(!check_img.data) {
        LOG(INFO) << "Could not open file " << image_lines_[i] << " - ignoring it";
        ignore_file[i] = true;
      }
      if(!check_lbl.data) {
        LOG(INFO) << "Could not open file " << label_lines_[i] << " - ignoring it";
        ignore_file[i] = true;
      }
    };

    if(this->layer_param_.image_label_data_param().check_image_files()) {
      LOG(INFO) << "Checking image files for errors";
      if(this->layer_param_.image_label_data_param().threads() != 1) {
        ParallelFor(0, image_lines_.size(), check_image_file_func);
      } else {
        for(int i=image_lines_.size()-1; i>=0; i--) {
          check_image_file_func(i);
        }
      }

      for(int i=image_lines_.size()-1; i>=0; i--) {
        if(ignore_file[i]) {
          image_lines_.erase(image_lines_.begin()+i);
          label_lines_.erase(label_lines_.begin()+i);
        }
      }
    }

    if (this->layer_param_.image_label_data_param().shuffle()) {
      // randomly shuffle data
      LOG(INFO) << "Shuffling data";
      const unsigned int prefetch_rng_seed = caffe_rng_rand();
      prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
      ShuffleImages();
    }
    LOG(INFO) << "A total of " << image_lines_.size() << " images.";
    LOG(INFO) << "A total of " << label_lines_.size() << " label.";
    CHECK_EQ(image_lines_.size(), label_lines_.size());

    lines_id_ = 0;
    // Check if we would need to randomly skip a few data points
    if (this->layer_param_.image_label_data_param().rand_skip()) {
      unsigned int skip = caffe_rng_rand() %
      this->layer_param_.image_label_data_param().rand_skip();
      LOG(INFO) << "Skipping first " << skip << " data points.";
      CHECK_GT(image_lines_.size(), skip) << "Not enough points to skip";
      lines_id_ = skip;
    }
    // Read an image, and use it to initialize the top blob.
    CHECK(file_exists(image_dir + image_lines_[lines_id_])) << "Could not load " << image_lines_[lines_id_];
    cv::Mat cv_img = ReadImageToCVMat(image_dir + image_lines_[lines_id_]);
    CHECK(cv_img.data) << "Could not load " << image_lines_[lines_id_];
    int crop_size = -1;
    auto transform_param = this->layer_param_.transform_param();
    if (transform_param.has_crop_size()) {
      crop_size = transform_param.crop_size();
    }
    cv_img = PadImage(cv_img, crop_size);

    // Use data_transformer to infer the expected blob shape from a cv_image.
    vector<int> data_shape = this->data_transformers_[0]->InferBlobShape(cv_img);

    // Reshape prefetch_data and top[0] according to the batch_size.
    const int batch_size = data_param.batch_size();
    CHECK_GT(batch_size, 0) << "Positive batch size required";
    data_shape[0] = batch_size;
    top[0]->Reshape(data_shape);

    /*
     * label
     */
    label_margin_h_ = data_param.has_label_slice()? data_param.label_slice().offset(0) : 0;
    label_margin_w_ = data_param.has_label_slice()? data_param.label_slice().offset(1) : 0;
    LOG(INFO) << "Assuming image and label map sizes are the same";
    vector<int> label_shape(4);
    label_shape[0] = batch_size;
    label_shape[1] = 1;
    label_shape[2] = data_param.has_label_slice()? data_param.label_slice().dim(0) : data_shape[2];
    label_shape[3] = data_param.has_label_slice()? data_param.label_slice().dim(1) : data_shape[3];

    top[1]->Reshape(label_shape);

    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->data_.Reshape(data_shape);
      this->prefetch_[i]->label_.Reshape(label_shape);
    }

    transformed_data_.resize(this->data_transformers_.size());
    transformed_label_.resize(this->data_transformers_.size());
    for(int i=0; i<this->data_transformers_.size(); i++) {
      transformed_data_[i].reset(new TBlob<Ftype>);
      transformed_data_[i]->Reshape(data_shape);

      transformed_label_[i].reset(new TBlob<Ftype>);
      transformed_label_[i]->Reshape(label_shape);
    }

    LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();

    LOG(INFO) << "output label size: " << top[1]->num() << ","
    << top[1]->channels() << "," << top[1]->height() << ","
    << top[1]->width();
  }

  template<typename Ftype, typename Btype>
  void ImageLabelListDataLayer<Ftype, Btype>::ShuffleImages() {
    caffe::rng_t* prefetch_rng =
    static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    vector<int> order(image_lines_.size());
    for (int i = 0; i < order.size(); ++i) {
      order[i] = i;
    }
    shuffle(order.begin(), order.end(), prefetch_rng);
    vector<std::string> new_image_lines(image_lines_.size());
    vector<std::string> new_label_lines(label_lines_.size());
    for (int i = 0; i < order.size(); ++i) {
      new_image_lines[i] = image_lines_[order[i]];
      new_label_lines[i] = label_lines_[order[i]];
    }
    swap(image_lines_, new_image_lines);
    swap(label_lines_, new_label_lines);
  }

  template<typename Ftype, typename Btype>
  void ImageLabelListDataLayer<Ftype, Btype>::SampleScale(cv::Mat *image, cv::Mat *label) {
    ImageLabelDataParameter data_param = this->layer_param_.image_label_data_param();
    if(data_param.size_min() != 0 || data_param.size_max() != 0) {
      int size_min = data_param.size_min();
      int size_max = data_param.size_max();

      bool adjust_size = (size_min != 0 && (std::min<int>(image->cols, image->rows) < size_min)) ||
      (size_max != 0 && (std::max<int>(image->cols, image->rows) > size_max));
      if(adjust_size) {
        auto clip_size = [&](int size) {
          if(size_min && size_max) {
            return std::max<int>(std::min<int>(size, size_max), size_min);
          } else if(size_min) {
            return std::max<int>(size, size_min);
          } else if(size_max) {
            return std::min<int>(size, size_max);
          } else {
            return size;
          }
        };

        cv::Size scaleSize(image->cols, image->rows);
        if(size_min) {
          if(image->cols < image->rows) {
            scaleSize.width = clip_size(scaleSize.width);
            scaleSize.height = clip_size(round(scaleSize.width * image->rows / (double)image->cols));
          } else {
            scaleSize.height = clip_size(scaleSize.height);
            scaleSize.width = clip_size(round(scaleSize.height * image->cols / (double)image->rows));
          }
        } else if(size_max) {
          if(image->cols > image->rows) {
            scaleSize.width = clip_size(scaleSize.width);
            scaleSize.height = clip_size(round(scaleSize.width * image->rows / (double)image->cols));
          } else {
            scaleSize.height = clip_size(scaleSize.height);
            scaleSize.width = clip_size(round(scaleSize.height * image->cols / (double)image->rows));
          }
        }

        ResizeTo(*image, image, *label, label, scaleSize);
      }
    }

    if(data_param.scale_prob()) {
      double scale_prob_rand_value = std::uniform_real_distribution<double>(0, 1.0)(*rng_);
      bool doScale = (scale_prob_rand_value <= data_param.scale_prob());
      //LOG(INFO) << "Random Resizing" << doScale;
      if(doScale) {
        double scale = std::uniform_real_distribution<double>(data_param.scale_min(), data_param.scale_max())(*rng_);
        cv::Size zero_size(0, 0);
        cv::resize(*label, *label, cv::Size(0, 0), scale, scale, cv::INTER_NEAREST);

        if (scale > 1) {
          cv::resize(*image, *image, zero_size, scale, scale, cv::INTER_CUBIC);
        } else {
          cv::resize(*image, *image, zero_size, scale, scale, cv::INTER_AREA);
        }
      }
    }
  }

  template<typename Ftype, typename Btype>
  void ImageLabelListDataLayer<Ftype, Btype>::ResizeTo(
      const cv::Mat& img,
      cv::Mat* img_temp,
      const cv::Mat& label,
      cv::Mat* label_temp,
      const cv::Size& size
  ) {
    // perform scaling if desired size and image size are non-equal:
    if (size.height != img.rows || size.width != img.cols) {
      int new_height = size.height;
      int new_width = size.width;

      //To preserve aspect ratio, even if both new_width and new_height are specified.
      //This is currently taken care outside.
      //int orig_height = img.rows;
      //int orig_width = img.cols;
      //if(orig_width > orig_height) {
      //    new_width = new_height*orig_width/orig_height;
      //} else {
      //    new_height = new_width*orig_height/orig_width;
      //}

      //LOG(INFO) << "Resizing " << img.cols << "x" << img.rows << " to " << new_width << "x" << new_height;
      double scale_value = ((new_width / (double)img.cols) + (new_height / (double)img.rows)) / 2;
      if(scale_value > 1) {
        cv::resize(img, *img_temp, cv::Size(new_width, new_height), cv::INTER_CUBIC);
      } else {
        cv::resize(img, *img_temp, cv::Size(new_width, new_height), cv::INTER_AREA);
      }
      cv::resize(label, *label_temp, cv::Size(new_width, new_height), cv::INTER_NEAREST);

      int h_off = (new_height - size.height) / 2;
      int w_off = (new_width - size.width) / 2;
      cv::Rect roi(w_off, h_off, size.width, size.height);
      *img_temp = (*img_temp)(roi);
      *label_temp = (*label_temp)(roi);

    } else {
      *img_temp = img.clone();
      *label_temp = label.clone();
    }
  }

  template<typename Ftype, typename Btype>
  void ImageLabelListDataLayer<Ftype, Btype>::load_batch(Batch<Ftype>* batch, int thread_id, size_t queue_id) {
    CPUTimer batch_timer;
    batch_timer.Start();
    CHECK(batch->data_.count());
    ImageLabelDataParameter data_param = this->layer_param_.image_label_data_param();
    const int batch_size = data_param.batch_size();
    string data_dir = data_param.data_dir();
    string image_dir = this->layer_param_.image_label_data_param().image_dir();
    string label_dir = this->layer_param_.image_label_data_param().label_dir();
    if (image_dir == "" && label_dir == "" && data_dir != "") {
      image_dir = data_dir;
      label_dir = data_dir;
    }
    int crop_size = -1;
    auto transform_param = this->layer_param_.transform_param();
    if (transform_param.has_crop_size()) {
      crop_size = transform_param.crop_size();
    }

    // Reshape according to the first image of each batch
    // on single input batches allows for inputs of varying dimension.
    CHECK(file_exists(image_dir + image_lines_[lines_id_])) << "Could not load " << image_lines_[lines_id_];
    cv::Mat cv_img = ReadImageToCVMat(image_dir + image_lines_[lines_id_], true);
    cv_img = PadImage(cv_img, crop_size);

    CHECK(cv_img.data) << "Could not load " << image_lines_[lines_id_];
    // Use data_transformer to infer the expected blob shape from a cv_img.
    vector<int> top_shape = this->data_transformers_[thread_id]->InferBlobShape(cv_img);

    // Reshape prefetch_data according to the batch_size.
    top_shape[0] = batch_size;
    batch->data_.Reshape(top_shape);

    CHECK(file_exists(label_dir + label_lines_[lines_id_])) << "Could not load " << label_lines_[lines_id_];
    cv::Mat cv_label = ReadImageToCVMat(label_dir + label_lines_[lines_id_], false);
    cv_label = PadImage(cv_label, crop_size);

    CHECK(cv_label.data) << "Could not load " << label_lines_[lines_id_];
    vector<int> label_shape = this->data_transformers_[thread_id]->InferBlobShape(cv_label);

    label_shape[0] = batch_size;
    label_shape[2] = data_param.has_label_slice()? data_param.label_slice().dim(0) : top_shape[2];
    label_shape[3] = data_param.has_label_slice()? data_param.label_slice().dim(1) : top_shape[3];
    batch->label_.Reshape(label_shape);

    Ftype* prefetch_data = batch->data_.mutable_cpu_data();
    Ftype* prefetch_label = batch->label_.mutable_cpu_data();

    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;

    auto lines_size = image_lines_.size();

    auto load_batch_parallel_func = [&](int item_id) {
      timer.Start();
      int line_d = (lines_id_ + item_id) % lines_size;
      CHECK_GT(lines_size, line_d);

      CHECK(file_exists(image_dir + image_lines_[lines_id_])) << "Could not load " << image_lines_[lines_id_];
      CHECK(file_exists(label_dir + label_lines_[lines_id_])) << "Could not load " << label_lines_[lines_id_];

      cv::Mat cv_img = ReadImageToCVMat(image_dir + image_lines_[line_d]);
      cv::Mat cv_label = ReadImageToCVMat(label_dir + label_lines_[line_d],
          false);
      SampleScale(&cv_img, &cv_label);

      switch (data_param.padding()) {
        case ImageLabelDataParameter_Padding_ZERO:
        cv_img = ExtendLabelMargin(cv_img, label_margin_w_, label_margin_h_, 0);
        cv_img = PadImage(cv_img, crop_size, 0);
        break;
        case ImageLabelDataParameter_Padding_REFLECT:
        cv_img = ExtendLabelMargin(cv_img, label_margin_w_, label_margin_h_, -1);
        cv_img = PadImage(cv_img, crop_size, -1);
        break;
        default:
        LOG(FATAL) << "Unknown Padding";
      }
      cv_label = ExtendLabelMargin(cv_label, label_margin_w_, label_margin_h_, 255);
      cv_label = PadImage(cv_label, crop_size, 255);

      CHECK(cv_img.data) << "Could not load " << image_lines_[line_d];
      CHECK(cv_label.data) << "Could not load " << label_lines_[line_d];
      read_time += timer.MicroSeconds();

      timer.Start();
      // Apply transformations (mirror, crop...) to the image
      int image_offset = batch->data_.offset(item_id);
      int label_offset = batch->label_.offset(item_id);

      transformed_data_[thread_id]->set_cpu_data(prefetch_data + image_offset);
      if(!data_param.has_label_slice()) {	  
        transformed_label_[thread_id]->set_cpu_data(prefetch_label + label_offset);
	  }
      this->data_transformers_[thread_id]->Transform(cv_img, cv_label, &(*transformed_data_[thread_id]), &(*transformed_label_[thread_id]));

      if(data_param.has_label_slice()) {
        Ftype *label_data = prefetch_label + label_offset;
        const Ftype *t_label_data = transformed_label_[thread_id]->cpu_data();
        GetLabelSlice(t_label_data, crop_size, crop_size, data_param.label_slice(), label_data);
	  }
      trans_time += timer.MicroSeconds();
    };

    int num_threads = std::min(batch_size, 2);
    ParallelFor(0, batch_size, load_batch_parallel_func, num_threads);

    // go to the next iter
    lines_id_ += batch_size;

    //LOG(INFO) << "Doing epoch " << epoch_ << "(" << lines_id_ << "/" << lines_size << ")";
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      LOG(INFO) << "Starting prefetch of epoch " << ++epoch_;
      lines_id_ = 0;
      if (this->layer_param_.image_label_data_param().shuffle()) {
        ShuffleImages();
      }
    }

    batch_timer.Stop();
    LOG_EVERY_N(INFO,10000) << "          Read batch time: " << read_time / 1000 << " ms.";
    LOG_EVERY_N(INFO,10000) << "     Transform batch time: " << trans_time / 1000 << " ms.";
    LOG_EVERY_N(INFO,10000) << "Total Prefetch batch time: " << batch_timer.MilliSeconds() << " ms.";

    batch->set_id(this->batch_id(thread_id));
  }

  INSTANTIATE_CLASS_FB(ImageLabelListDataLayer);
  REGISTER_LAYER_CLASS(ImageLabelListData);

}  // namespace caffe
#endif  // USE_OPENCV
