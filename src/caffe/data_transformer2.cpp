#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>

#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

namespace caffe {

#ifndef CPU_ONLY
template<typename Dtype>
void DataTransformer<Dtype>::TransformGPU(const TBlob<Dtype>* input_blob, TBlob<Dtype>* transformed_blob,
    const std::array<unsigned int, 3>& rand, bool use_mean) {
  unsigned int *randoms = reinterpret_cast<unsigned int *>(GPUMemory::thread_pinned_buffer(sizeof(unsigned int) * 3));
  std::memcpy(randoms, &rand.front(), sizeof(unsigned int) * 3);  // NOLINT(caffe/alt_fn)
  const vector<int> input_shape = input_blob->shape();

  TransformGPU(input_shape[0], input_shape[1], input_shape[2], input_shape[3], sizeof(Dtype), input_blob->gpu_data(),
      transformed_blob->mutable_gpu_data(false), randoms, use_mean);
}
#endif

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const TBlob<Dtype>* input_blob, TBlob<Dtype>* transformed_blob,
    const std::array<unsigned int, 3>& rand, bool use_mean) {
  bool use_gpu_transform = param_.use_gpu_transform() && Caffe::mode() == Caffe::GPU;

  const int crop_size = param_.crop_size();
  int transformed_height = crop_size ? crop_size : input_blob->height();
  int transformed_width = crop_size ? crop_size : input_blob->width();
  vector<int> transformed_shape = { input_blob->num(), input_blob->channels(), transformed_height, transformed_width };
  transformed_blob->Reshape(transformed_shape);

  if (use_gpu_transform) {
#ifndef CPU_ONLY
    TransformGPU(input_blob, transformed_blob, rand, use_mean);
    transformed_blob->cpu_data();
#else
    NO_GPU;
#endif
  } else {
    TransformCPU(input_blob, transformed_blob, rand, use_mean);
  }
}

//Used in ImageLabelDataLayer
template<typename Dtype>
void DataTransformer<Dtype>::TransformCPU(const TBlob<Dtype>* input_blob, TBlob<Dtype>* transformed_blob,
    const std::array<unsigned int, 3>& rand, bool use_mean) {
  const Dtype* data = input_blob->cpu_data();
  const int datum_channels = input_blob->channels();
  const int datum_height = input_blob->height();
  const int datum_width = input_blob->width();

  Dtype *transformed_data = transformed_blob->mutable_cpu_data();

  const int crop_size = param_.crop_size();
  const float scale = param_.scale();
  const bool do_mirror = param_.mirror() && (rand[0] % 2);
  const bool has_mean_file = use_mean ? param_.has_mean_file() : false;
  const bool has_mean_values = use_mean ? mean_values_.size() > 0 : false;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  const float* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels)
                                                                                << "Specify either 1 mean_value or as many as channels: "
                                                                                << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = rand[1] % (datum_height - crop_size + 1);
      w_off = rand[2] % (datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  int top_index, data_index, ch, cdho;
  const int m = do_mirror ? -1 : 1;

  Dtype datum_element;
  for (int c = 0; c < datum_channels; ++c) {
    cdho = c * datum_height + h_off;
    ch = c * height;
    for (int h = 0; h < height; ++h) {
      top_index = do_mirror ? (ch + h + 1) * width - 1 : (ch + h) * width;
      data_index = (cdho + h) * datum_width + w_off;
      for (int w = 0; w < width; ++w) {
        datum_element = data[data_index];
        if (has_mean_file) {
          transformed_data[top_index] = (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] = (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
        ++data_index;
        top_index += m;
      }
    }
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
    const cv::Mat& cv_label,
    TBlob<Dtype>* transformed_image,
    TBlob<Dtype>* transformed_label) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_image->channels();
  const int height = transformed_image->height();
  const int width = transformed_image->width();
  const int num = transformed_image->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U || cv_img.depth() == CV_32F)
  << "Image data type must be unsigned byte or float";
  CHECK(cv_label.type() == CV_8U)
  << "Label data type must be unsigned byte with single channel";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  float* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
    "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  cv::Mat cv_cropped_label = cv_label;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
    cv_cropped_label = cv_label(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);
  CHECK(cv_cropped_label.data);

  //TODO: This is currently disabled as there is a seperate layer to do transformations
  //Additional transformations
  //if (phase_ == TRAIN) {
  //	TransformInPlace2(cv_cropped_img, cv_cropped_label);
  //}

  if (param_.display() && phase_ == TRAIN) {
    //cv::imshow("Final Image", cv_cropped_img);
    //cv::imshow("Final Label", cv_cropped_label);
    //cv::waitKey(1000);
    static int counter = 0;
    std::stringstream ss; ss << counter++;
    std::string counter_str = ss.str();
    cv::imwrite("images/"+counter_str+".png", cv_cropped_img);
    cv::imwrite("labels/"+counter_str+".png", cv_cropped_label);
  }

  Dtype* transformed_data = transformed_image->mutable_cpu_data();
  // Dtype *transformed_data = (*t_img)[0];

  if (cv_cropped_img.depth() == CV_8U) {
    cv_cropped_img.convertTo(cv_cropped_img, CV_32F);
  }

  int top_index;
  for (int h = 0; h < height; ++h) {
    const float* ptr = cv_cropped_img.ptr<float>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
          (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
            (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }

  auto checkLabel = [](Dtype value, const Dtype min_val, const Dtype max_val) {
    return (value<min_val? Dtype(UINT8_MAX) : (value>max_val? Dtype(UINT8_MAX): value));
  };

  Dtype* transformed_label_data = transformed_label->mutable_cpu_data();
  // Dtype* transformed_label_data = (*t_label)[0];

  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_label.ptr<uchar>(h);
    int label_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < 1; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[label_index++]);
        if(param_.has_num_labels() && param_.num_labels() > 0) {
          Dtype max_label = param_.num_labels() - 1;
          pixel = checkLabel(pixel, 0, max_label);
        }
        transformed_label_data[top_index] = pixel;
      }
    }
  }
}
#endif  // USE_OPENCV

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0)<< "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template void DataTransformer<float>::Transform(const TBlob<float>* input_blob, TBlob<float>* transformed_blob,
    const std::array<unsigned int, 3>& rand, bool use_mean);
template void DataTransformer<float16>::Transform(const TBlob<float16>* input_blob, TBlob<float16>* transformed_blob,
    const std::array<unsigned int, 3>& rand, bool use_mean);
template void DataTransformer<double>::Transform(const TBlob<double>* input_blob, TBlob<double>* transformed_blob,
    const std::array<unsigned int, 3>& rand, bool use_mean);

template void DataTransformer<float>::Transform(const cv::Mat& cv_img, const cv::Mat& cv_label,
    TBlob<float>* transformed_image, TBlob<float>* transformed_label);
template void DataTransformer<float16>::Transform(const cv::Mat& cv_img, const cv::Mat& cv_label,
    TBlob<float16>* transformed_image, TBlob<float16>* transformed_label);
template void DataTransformer<double>::Transform(const cv::Mat& cv_img, const cv::Mat& cv_label,
    TBlob<double>* transformed_image, TBlob<double>* transformed_label);

}  // namespace caffe
