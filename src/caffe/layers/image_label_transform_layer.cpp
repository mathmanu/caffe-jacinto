#ifdef USE_OPENCV

#include "caffe/layers/image_label_transform_layer.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <boost/array.hpp>
#include <boost/foreach.hpp>
#include <boost/static_assert.hpp>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/parallel.hpp"

using namespace cv;  // NOLINT(build/namespaces)
using boost::array;
#define foreach_ BOOST_FOREACH


namespace caffe {

template<typename Dtype>
struct ImageLabelTransformationAugmentSelection {
    bool flip;
    float degree;
    Point crop;
    float scale;
    float hue_rotation;
    float saturation;
};


template<typename Dtype>
ImageLabelTransformationLayer<Dtype>::ImageLabelTransformationLayer(
    const LayerParameter& param) :
    Layer<Dtype>(param),
    a_param_(param.image_label_transform_param()),
    t_param_(param.transform_param()),
    phase_(param.phase()),
    rng_(new Caffe::RNG(caffe_rng_rand())) {}


template <typename Dtype>
void ImageLabelTransformationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // retrieve mean image or values:
  if (t_param_.has_mean_file()) {
    size_t image_x = bottom[0]->width();
    size_t image_y = bottom[0]->height();

    retrieveMeanImage(Size(image_x, image_y));

  } else if (t_param_.mean_value_size() != 0) {
    retrieveMeanChannels();
  }
}


template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::retrieveMeanImage(Size dimensions) {
  CHECK(t_param_.has_mean_file());

  const string& mean_file = t_param_.mean_file();
  BlobProto blob_proto;
  Blob<Dtype> data_mean;

  ReadProtoFromBinaryFileOrDie(mean_file, &blob_proto);
  data_mean.FromProto(blob_proto);
  data_mean_ = blobToMats(data_mean).at(0);

  // resize, if dimensions were defined:
  if (dimensions.area() > 0) {
    cv::resize(data_mean_, data_mean_, dimensions, 0, 0, cv::INTER_CUBIC);
  }
  // scale from 0..255 to 0..1:
  data_mean_ /= Dtype(UINT8_MAX);
}


template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::retrieveMeanChannels() {
  switch (t_param_.mean_value_size()) {
    case 1:
      mean_values_.fill(t_param_.mean_value(0) / Dtype(UINT8_MAX));
      break;
    case 3:
      for (size_t iChannel = 0; iChannel != 3; ++iChannel) {
        mean_values_[iChannel] =
            t_param_.mean_value(iChannel) / Dtype(UINT8_MAX);
      }
      break;
    case 0:
    default:
      break;
  }
}


template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top
) {
  // accept only three channel images:
  CHECK_EQ(bottom[0]->channels(), 3);
  // accept only equal numbers of labels and images:
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  // resize mean image if it exists:
  if (t_param_.has_mean_file()) {
    size_t image_x = bottom[0]->width();
    size_t image_y = bottom[0]->height();
    retrieveMeanImage(Size(image_x, image_y));
  }
  // resize image output layer: (never changes /wrt input blobs)
  top[0]->Reshape(
      bottom[0]->num(),
      bottom[0]->channels(),
      a_param_.crop_height(),
      a_param_.crop_width());
  // resize tensor output layer: (never changes /wrt input blobs)

  top[1]->Reshape(
      bottom[1]->num(),
      bottom[1]->channels(),
      a_param_.crop_height(),
      a_param_.crop_width());
}


template<typename Dtype>
typename ImageLabelTransformationLayer<Dtype>::Mat3v
ImageLabelTransformationLayer<Dtype>::dataToMat(
    const Dtype* _data,
    Size dimensions
) const {
  // NOLINT_NEXT_LINE(whitespace/line_length)
  // CODE FROM https://github.com/BVLC/caffe/blob/4874c01487/examples/cpp_classification/classification.cpp#L125-137
  // The format of the mean file is planar 32-bit float BGR or grayscale.
  vector<Mat> channels; channels.reserve(3);

  Dtype* data = const_cast<Dtype*>(_data);
  for (size_t iChannel = 0; iChannel != 3; ++iChannel) {
    // Extract an individual channel. This does not perform a datacopy, so doing
    //  this in a performance critical location should be fine.
    Mat channel(dimensions, cv::DataType<Dtype>::type, data);
    channels.push_back(channel);
    data += dimensions.area();
  }

  // Merge the separate channels into a single image.
  Mat3v result;
  merge(channels, result);

  return result;
}

template<typename Dtype>
typename ImageLabelTransformationLayer<Dtype>::Mat3v
ImageLabelTransformationLayer<Dtype>::dataToMat1(
    const Dtype* _data,
    Size dimensions
) const {
  // NOLINT_NEXT_LINE(whitespace/line_length)
  // CODE FROM https://github.com/BVLC/caffe/blob/4874c01487/examples/cpp_classification/classification.cpp#L125-137
  // The format of the mean file is planar 32-bit float BGR or grayscale.
  vector<Mat> channels; channels.reserve(3);

  Dtype* data = const_cast<Dtype*>(_data);
  for (size_t iChannel = 0; iChannel != 1; ++iChannel) {
    // Extract an individual channel. This does not perform a datacopy, so doing
    //  this in a performance critical location should be fine.
    Mat channel(dimensions, cv::DataType<Dtype>::type, data);
    channels.push_back(channel);
    channels.push_back(channel);
    channels.push_back(channel);
    data += dimensions.area();
  }

  // Merge the separate channels into a single image.
  Mat3v result;
  merge(channels, result);

  return result;
}

template<typename Dtype>
vector<typename ImageLabelTransformationLayer<Dtype>::Mat3v>
ImageLabelTransformationLayer<Dtype>::blobToMats(
    const Blob<Dtype>& images
) const {
  CHECK_EQ(images.channels(), 3);
  vector<Mat3v> result; result.reserve(images.num());

  for (size_t iImage = 0; iImage != images.num(); ++iImage) {
    const Dtype* image_data = &images.cpu_data()[
        images.offset(iImage, 0, 0, 0)];

    result.push_back(dataToMat(
        image_data,
        Size(images.width(), images.height())));
  }

  return result;
}

template<typename Dtype>
vector<typename ImageLabelTransformationLayer<Dtype>::Mat3v>
ImageLabelTransformationLayer<Dtype>::blobToMats1(
    const Blob<Dtype>& images
) const {
  CHECK_EQ(images.channels(), 1);
  vector<Mat3v> result; result.reserve(images.num());

  for (size_t iImage = 0; iImage != images.num(); ++iImage) {
    const Dtype* image_data = &images.cpu_data()[
        images.offset(iImage, 0, 0, 0)];

    result.push_back(dataToMat1(
        image_data,
        Size(images.width(), images.height())));
  }

  return result;
}


template<typename Dtype>
struct toDtype : public std::unary_function<float, Dtype> {
  Dtype operator() (const Vec<Dtype, 1>& value) { return value(0); }
};
template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::matToBlob(
    const Mat3v& source,
    Dtype* destination
) const {
  std::vector<Mat1v> channels;
  split(source, channels);

  size_t offset = 0;
  for (size_t iChannel = 0; iChannel != channels.size(); ++iChannel) {
    const Mat1v& channel = channels[iChannel];

    std::transform(
        channel.begin(),
        channel.end(),
        &destination[offset],
        toDtype<Dtype>());

    offset += channel.total();
  }
}

template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::matsToBlob(
    const vector<Mat3v>& _source,
    Blob<Dtype>* _dest
) const {
  for (size_t iImage = 0; iImage != _source.size(); ++iImage) {
    Dtype* destination = &_dest->mutable_cpu_data()[
        _dest->offset(iImage, 0, 0, 0)
    ];
    const Mat3v& source = _source[iImage];
    matToBlob(source, destination);
  }
}

template<typename Dtype>
struct roundType : public std::unary_function<float, Dtype> {
  Dtype operator() (const Vec<Dtype, 1>& value) { return round(value(0)); }
};

template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::matToRoundBlob1(
    const Mat3v& source,
    Dtype* destination
) const {
  std::vector<Mat1v> channels;
  split(source, channels);

  size_t offset = 0;
  for (size_t iChannel = 0; iChannel < 1; ++iChannel) {
    const Mat1v& channel = channels[iChannel];

    std::transform(
        channel.begin(),
        channel.end(),
        &destination[offset],
        roundType<Dtype>());

    offset += channel.total();
  }
}

template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::matsToRoundBlob1(
    const vector<Mat3v>& _source,
    Blob<Dtype>* _dest
) const {
  for (size_t iImage = 0; iImage != _source.size(); ++iImage) {
    Dtype* destination = &_dest->mutable_cpu_data()[
        _dest->offset(iImage, 0, 0, 0)
    ];
    const Mat3v& source = _source[iImage];
    matToRoundBlob1(source, destination);
  }
}

template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::pixelMeanSubtraction(
    Mat3v* source
) const {
  *source -= data_mean_;
}


template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::channelMeanSubtraction(
    Mat3v* source
) const {
  vector<Mat1f> channels;
  split(*source, channels);
  for (size_t iChannel = 0; iChannel != channels.size(); ++iChannel) {
    channels[iChannel] -= mean_values_[iChannel];
  }
  merge(channels, *source);
}


template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::meanSubtract(Mat3v* source) const {
  if (t_param_.has_mean_file()) {
    pixelMeanSubtraction(source);
  } else if (t_param_.mean_value_size() != 0) {
    channelMeanSubtraction(source);
  }
}


template<typename Dtype>
Dtype ImageLabelTransformationLayer<Dtype>::randDouble() {
  rng_t* rng =
      static_cast<rng_t*>(rng_->generator());
  uint64_t randval = (*rng)();

  return (Dtype(randval) / Dtype(rng_t::max()));
}


template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top
) {
  // verify image parameters:
  const int image_count = bottom[0]->num();
  // verify label parameters:
  const int label_count = bottom[1]->num();
  CHECK_EQ(image_count, label_count);

  const vector<Mat3v> inputImages = blobToMats(*bottom[0]);
  const vector<Mat3v> inputLabels = blobToMats1(*bottom[1]);

  vector<Mat3v> outputImages(inputImages.size());
  vector<Mat3v> outputLabels(inputLabels.size());

  if(a_param_.threads() != 1) {
    auto transform_image_label_func = [&](int iImage) {
      const Mat3v& inputImage = inputImages[iImage];
      const Mat3v& inputLabel = inputLabels[iImage];
      Mat3v& outputImage = outputImages[iImage];
      Mat3v& outputLabel = outputLabels[iImage];

      transform(inputImage, inputLabel, &outputImage, &outputLabel);

      //cv::imshow("Source", inputImage/255.0);
      //cv::imshow("Transormed", outputImage/255.0);
      //cv::waitKey(1000);
    };
    ParallelFor(0, inputImages.size(), transform_image_label_func);
  } else {
    for (size_t iImage = 0; iImage != inputImages.size(); ++iImage) {
      const Mat3v& inputImage = inputImages[iImage];
      const Mat3v& inputLabel = inputLabels[iImage];
      Mat3v& outputImage = outputImages[iImage];
      Mat3v& outputLabel = outputLabels[iImage];

      transform(inputImage, inputLabel, &outputImage, &outputLabel);

      //cv::namedWindow("Transormed", CV_WINDOW_AUTOSIZE);
      //cv::imshow("Source", inputImage/255.0);
      //cv::imshow("Transormed", outputImage/255.0);
      //cv::waitKey(1000);
    }
  }

  // emplace images in output image blob:
  matsToBlob(outputImages, top[0]);
  matsToRoundBlob1(outputLabels, top[1]);
}

template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::transform(
    const Mat3v& img,
    const Mat3v& label,
    Mat3v* img_aug,
    Mat3v* label_aug
) {
  uint32_t image_x = a_param_.crop_width();
  uint32_t image_y = a_param_.crop_height();

  // Perform mean subtraction on un-augmented image:
  Mat3v img_temp = img.clone();  // size determined by scale
  Mat3v label_temp = label.clone();  // size determined by scale
  // incoming float images have values from 0..255, but OpenCV expects these
  //  values to be from 0..1:
  img_temp /= Dtype(UINT8_MAX);
  label_temp /= Dtype(UINT8_MAX);

  *img_aug = Mat::zeros(image_y, image_x, img.type());
  *label_aug = Mat::zeros(image_y, image_x, label.type());

  ImageLabelTransformationAugmentSelection<Dtype> as;
  // We only do random transform as augmentation when training.
  if (this->phase_ == TRAIN) {
    // TODO: combine hueRotation and desaturation if performance is a concern,
    //  as we must redundantly convert to and from HSV colorspace for each
    //  operation
    as.hue_rotation = augmentation_hueRotation(img_temp, &img_temp);
    as.saturation = augmentation_desaturation(img_temp, &img_temp);
    augmentation_contrast(img_temp, &img_temp);
    clip_pixels(img_temp, 0.0, +1.0);

    // images now bounded from [-1,1] (range is dependent on mean subtraction)
    as.scale = augmentation_scale(img_temp, img_aug, label_temp, label_aug);
    as.degree =
        augmentation_rotate(*img_aug, &img_temp, *label_aug, &label_temp);
    as.flip = augmentation_flip(img_temp, img_aug, label_temp, label_aug);
    as.crop = augmentation_crop(
        *img_aug,
        &img_temp,
        *label_aug,
		&label_temp);

    clip_pixels(img_temp, 0.0, +1.0);

    // mean subtraction must occur after color augmentations as colorshift
    //  outside of 0..1 invalidates scale
    meanSubtract(&img_temp);

    *img_aug = img_temp;
    *label_aug = label_temp;

  } else if(a_param_.scale_width()!=0 && a_param_.scale_height()!=0) {
    Size scaleSize(a_param_.scale_width(), a_param_.scale_height());

    // deterministically scale the image and ground-truth, if requested:
    transform_scale_adaptive(
        img_temp,
        img_aug,
        label_temp,
        label_aug,
        scaleSize);

    // and then take a single random crop
    // see resnet paper for the motivation for this
    as.crop = augmentation_crop(
        *img_aug,
        &img_temp,
        *label_aug,
		&label_temp);

    // perform mean subtraction:
    meanSubtract(&img_temp);

    *img_aug = img_temp;
    *label_aug = label_temp;
  } else {
    // deterministically scale the image and ground-truth, if requested:
    transform_scale(
        img_temp,
        img_aug,
        label_temp,
        label_aug,
        Size(image_x, image_y));

    // perform mean subtraction:
    meanSubtract(img_aug);
  }

  CHECK_EQ(img_aug->cols, image_x);
  CHECK_EQ(img_aug->rows, image_y);

  // networks expect floats bounded from -255..255, so rescale to this range:
  *img_aug *= Dtype(UINT8_MAX);
  *label_aug *= Dtype(UINT8_MAX);

  if(a_param_.has_num_labels() &&  a_param_.num_labels() > 0) {
    int max_label = a_param_.num_labels() - 1;
    check_labels(*label_aug, 0.0, max_label);
  }
}


template<typename Dtype>
float ImageLabelTransformationLayer<Dtype>::augmentation_scale(
    const Mat3v& img_src,
    Mat3v* img_dst,
    const Mat3v& label_src,
    Mat3v* label_dst
) {
  Mat3v img_temp;
  Mat3v label_temp;

  if(a_param_.scale_width()!=0 && a_param_.scale_height()!=0) {
	 Size scaleSizeFixed(a_param_.scale_width(), a_param_.scale_height());
	 transform_scale_adaptive(
	        img_src,
	        img_dst,
	        label_src,
	        label_dst,
	        scaleSizeFixed);
	 img_temp = (*img_dst).clone();
	 label_temp = (*label_dst).clone();
  } else {
	 img_temp = (img_src).clone();
	 label_temp = (label_src).clone();
  }

  bool doScale = randDouble() <= a_param_.scale_prob();
  float scale = 1;
  if(doScale) {
	// linear shear into [scale_min, scale_max]
	scale =
	    a_param_.scale_min() +
	    (a_param_.scale_max() - a_param_.scale_min()) *
	    randDouble();

    // scale uniformly across both axes by some random value:
    Size scaleSize(round(img_temp.cols * scale), round(img_temp.rows * scale));
    transform_scale(
        img_temp,
        img_dst,
        label_temp,
        label_dst,
        scaleSize);
  } else {
    *img_dst = img_temp.clone();
    *label_dst = label_temp.clone();
    scale = 1;
  }

  return scale;
}


template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::transform_scale(
    const Mat3v& img,
    Mat3v* img_temp,
    const Mat3v& label,
    Mat3v* label_temp,
    const Size& size
) {
  // perform scaling if desired size and image size are non-equal:
  if (size.height!=0 && size.width!=0 && (size.height != img.rows || size.width != img.cols)) {
    //Dtype scale_x = (Dtype)size.width / img.cols;
    //Dtype scale_y = (Dtype)size.height / img.rows;
    cv::resize(img, *img_temp, size, 0, 0, cv::INTER_CUBIC);
    cv::resize(label, *label_temp, size, 0, 0, cv::INTER_NEAREST);
  } else {
    *img_temp = img.clone();
    *label_temp = label.clone();
  }
}

template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::transform_scale_adaptive(
    const Mat3v& img,
    Mat3v* img_temp,
    const Mat3v& label,
    Mat3v* label_temp,
    const Size& size
) {
  // perform scaling if desired size and image size are non-equal:
  if (size.height != img.rows || size.width != img.cols) {
	int orig_height = img.rows;
	int orig_width = img.cols;
    int new_height = size.height;
	int new_width = size.width;
    if (orig_width > orig_height) {
	    new_width = new_height*orig_width/orig_height;
	} else {
	    new_height = new_width*orig_height/orig_width;
	}
    cv::resize(img, *img_temp, cv::Size(new_width, new_height), cv::INTER_CUBIC);
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

template<typename Dtype>
Point ImageLabelTransformationLayer<Dtype>::augmentation_crop(
    const Mat3v& img_src,
    Mat3v* img_dst,
    const Mat3v& label_src,
    Mat3v* label_dst
) {
  Size2i crop(a_param_.crop_width(), a_param_.crop_height());
  Size2i shift(a_param_.shift_x(), a_param_.shift_y());
  Size2i imgSize(img_src.cols, img_src.rows);
  Point2i offset, inner, outer;
  
  if(crop.height == 0 || crop.width == 0) {
     *img_dst = img_src.clone();
     *label_dst = label_src.clone();
    offset = Point2i(
        (imgSize.width  - crop.width) / 2,
        (imgSize.height - crop.height) / 2);     
     return offset;
  }
  
  bool doCrop = (randDouble() <= a_param_.crop_prob());  
  if (doCrop) {
    // perform a random crop, bounded by the difference between the network's
    //  input and the size of the incoming image. Add a user-defined shift to
    //  the max range of the crop offset.
    offset = Point2i(
        round(randDouble() * (imgSize.width  - crop.width  + shift.width)),
        round(randDouble() * (imgSize.height - crop.height + shift.height)));
    offset -= Point2i(shift.width / 2, shift.height / 2);
  } else {
    // perform a deterministic crop, placing the image in the middle of the
    //  network's input region:
    offset = Point2i(
        (imgSize.width  - crop.width) / 2,
        (imgSize.height - crop.height) / 2);
  }
  inner = Point2i(std::max(0,  1 * offset.x), std::max(0,  1 * offset.y));
  outer = Point2i(std::max(0, -1 * offset.x), std::max(0, -1 * offset.y));

  // crop / grow to size:
  transform_crop(
      img_src,
      img_dst,
      label_src,
      label_dst,
      Rect(inner, crop),
      crop,
      outer);
  return offset;
}


template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::transform_crop(
    const Mat3v& img_src,
    Mat3v* img_dst,
    const Mat3v& label_src,
    Mat3v* label_dst,
    Rect inner,
    Size2i dst_size,
    Point2i outer
) const {
  // ensure src_rect fits within img_src:
  Rect src_rect = inner & Rect(Point(0, 0), Size2i(img_src.cols, img_src.rows));
  // dst_rect has a size the same as src_rect:
  Rect dst_rect(outer, src_rect.size());
  // ensure dst_rect fits within img_dst:
  dst_rect &= Rect(Point(0, 0), dst_size);
  // assert src_rect and dst_rect have the same size:
  src_rect = Rect(src_rect.tl(), dst_rect.size());
  // Fail with an Opencv exception if any of these are negative

  // no operation is needed in the case of zero transformations:
  if (src_rect == dst_rect
    && src_rect.tl() == Point2i(0, 0)
    && src_rect.size() == dst_size
    && dst_size == Size2i(img_src.cols, img_src.rows)) {
    // no crop is needed:
    *img_dst = img_src.clone();
    *label_dst = label_src.clone();
  } else {
    // construct a destination matrix:
    *img_dst = Mat3v(dst_size);
    // and fill with black:
    img_dst->setTo(Scalar(0, 0, 0));

    // define destinationROI inside of destination mat:
    Mat3v destinationROI = (*img_dst)(dst_rect);
    // sourceROI inside of source mat:
    Mat3v sourceROI = img_src(src_rect);
    // copy sourceROI into destinationROI:
    sourceROI.copyTo(destinationROI);

    // construct a destination matrix:
    *label_dst = Mat3v(dst_size);
    // and fill with black:
    label_dst->setTo(Scalar(0, 0, 0));

    // define destinationROI inside of destination mat:
    Mat3v labelDestinationROI = (*label_dst)(dst_rect);
    // sourceROI inside of source mat:
    Mat3v labelSourceROI = label_src(src_rect);
    // copy sourceROI into destinationROI:
    labelSourceROI.copyTo(labelDestinationROI);
  }
}


template<typename Dtype>
bool ImageLabelTransformationLayer<Dtype>::augmentation_flip(
    const Mat3v& img_src,
    Mat3v* img_aug,
    const Mat3v& label_src,
    Mat3v* label_aug) {
  bool doflip = randDouble() <= a_param_.flip_prob();
  if (doflip) {
    flip(img_src, *img_aug, 1);
    //float w = img_src.cols;

    flip(label_src, *label_aug, 1);
    //float wl = label_src.cols;
  } else {
    *img_aug = img_src.clone();
    *label_aug = label_src.clone();
  }
  return doflip;
}


template<typename Dtype>
typename ImageLabelTransformationLayer<Dtype>::Mat1v
ImageLabelTransformationLayer<Dtype>::getTransformationMatrix(
    Rect region,
    Dtype rotation
) const {
  Size2v size(region.width, region.height);
  Point2v center = size * (Dtype)0.5;
  array<Point2f, 4> srcTri, dstTri;

  // Define a rotated rectangle for our initial position, retrieving points
  //  for each of our 4 corners:
  RotatedRect initialRect(center, size, 0.0);
  initialRect.points(srcTri.c_array());
  // Another rotated rectangle for our eventual position.
  RotatedRect rotatedRect(center, size, rotation);
  // retrieve boundingRect, whose top-left will be below zero:
  Rectv boundingRect = rotatedRect.boundingRect();
  // push all points up by the the topleft boundingRect's delta from
  //  the origin:
  rotatedRect = RotatedRect(center - boundingRect.tl(), size, rotation);
  // retrieve points for each of the rotated rectangle's 4 corners:
  rotatedRect.points(dstTri.c_array());

  // compute the affine transformation of this operation:
  Mat1v result(2, 3);
  result = getAffineTransform(srcTri.c_array(), dstTri.c_array());
  // return the transformation matrix
  return result;
}


template<typename Dtype>
Rect ImageLabelTransformationLayer<Dtype>::getBoundingRect(
    Rect region,
    Dtype rotation
) const {
  Size2v size(region.width, region.height);
  Point2v center = size * (Dtype)0.5;

  return RotatedRect(center, size, rotation).boundingRect();
}


template<typename Dtype>
float ImageLabelTransformationLayer<Dtype>::augmentation_rotate(
    const Mat3v& img_src,
    Mat3v* img_aug,
    const Mat3v& label_src,
    Mat3v* label_aug) {
  bool doRotate = randDouble() <= a_param_.rotation_prob();
  float degree = (randDouble() - 0.5) * 2 * a_param_.max_rotate_degree();

  if (doRotate && std::abs(degree) > FLT_EPSILON) {
    Rect roi(0, 0, img_src.cols, img_src.rows);
    // determine new bounding rect:
    Size2i boundingSize = getBoundingRect(roi, degree).size();
    // determine rotation matrix:
    Mat1v transformationMatrix = getTransformationMatrix(roi, degree);

    // construct a destination matrix large enough to contain the rotated image:
    *img_aug = Mat3v(boundingSize);
    // and fill with black:
    img_aug->setTo(Scalar(0, 0, 0));
    // warp old image into new buffer, maintaining the background:
    warpAffine(
        img_src,
        *img_aug,
        transformationMatrix,
        boundingSize,
        INTER_LINEAR,
        BORDER_TRANSPARENT);

    // construct a destination matrix large enough to contain the rotated image:
    *label_aug = Mat3v(boundingSize);
    // and fill with black:
    label_aug->setTo(Scalar(0, 0, 0));
    // warp old image into new buffer, maintaining the background:
    warpAffine(
        label_src,
        *label_aug,
        transformationMatrix,
        boundingSize,
        INTER_LINEAR,
        BORDER_TRANSPARENT);
  } else {
    *img_aug = img_src.clone();
    *label_aug = label_src.clone();
    degree = 0.0f;
  }

  return degree;
}


template<typename Dtype>
float ImageLabelTransformationLayer<Dtype>::augmentation_hueRotation(
    const Mat3v& img,
    Mat3v* result) {
  bool doHueRotate = randDouble() <= a_param_.hue_rotation_prob();
  // rotate hue by this amount in degrees
  float rotation =
      (randDouble()                       // range: 0..1
      * (2.0 * a_param_.hue_rotation()))  // range: 0..2*rot
      - a_param_.hue_rotation();          // range: -rot..rot
  // clamp to -180d..180d
  rotation = std::max(std::min(rotation, 180.0f), -180.0f);

  // if we're actually rotating:
  if (doHueRotate && std::abs(rotation) > FLT_EPSILON) {
    // convert to HSV colorspace
    cvtColor(img, *result, COLOR_BGR2HSV);

    // retrieve the hue channel:
    static const array<int, 2> from_mix = {{0, 0}};
    Mat1v hueChannel(result->rows, result->cols);
    mixChannels(
        result, 1,
        &hueChannel, 1,
        from_mix.data(), from_mix.size()/2);

    // shift the hue's value by some amount:
    hueChannel += rotation;

    // place hue-rotated channel back in result matrix:
    // NOLINT_NEXT_LINE(whitespace/comma)
    static const array<int, 6> to_mix = {{3,0, 1,1, 2,2}};
    const array<Mat, 2> to_channels = {{*result, hueChannel}};
    mixChannels(
        to_channels.data(), 2,
        result, 1,
        to_mix.data(), to_mix.size()/2);

    // back to BGR colorspace
    cvtColor(*result, *result, COLOR_HSV2BGR);
  } else {
    *result = img;
    rotation = 0.0f;
  }

  return rotation;
}


template<typename Dtype>
float ImageLabelTransformationLayer<Dtype>::augmentation_desaturation(
    const Mat3v& img,
    Mat3v* result) {
  bool doDesaturate = randDouble() <= a_param_.desaturation_prob();
  // scale saturation by this amount:
  float saturation =
      1.0                           // inverse
    - randDouble()                  // range: 0..1
    * a_param_.desaturation_max();  // range: 0..max

  // if our random value is large enough to produce noticeable desaturation:
  if (doDesaturate && (saturation < 1.0 - 1.0/UINT8_MAX)) {
    // convert to HSV colorspace
    cvtColor(img, *result, COLOR_BGR2HSV);

    // retrieve the saturation channel:
    static const array<int, 2> from_mix = {{1, 0}};
    Mat1v saturationChannel(result->rows, result->cols);
    mixChannels(
        result, 1,
        &saturationChannel, 1,
        from_mix.data(), from_mix.size()/2);
    // de-saturate the channel by an amount:
    saturationChannel *= saturation;

    // place de-saturated channel back in result matrix:
    // NOLINT_NEXT_LINE(whitespace/comma)
    static const array<int, 6> to_mix = {{0,0, 3,1, 2,2}};
    const array<Mat, 2> to_channels = {{*result, saturationChannel}};
    mixChannels(
        to_channels.data(), 2,
        result, 1,
        to_mix.data(), to_mix.size()/2);

    // convert back to BGR colorspace:
    cvtColor(*result, *result, COLOR_HSV2BGR);
  } else {
    *result = img;
    saturation = 1.0;
  }

  return saturation;
}





template<typename Dtype>
float ImageLabelTransformationLayer<Dtype>::augmentation_contrast(
    const Mat3v& img, Mat3v* result) {
  bool doContrast = randDouble() <= a_param_.contrast_adjust_prob();
  // scale by this amount:
  float contrast = a_param_.contrast_adjust_min() +
      (a_param_.contrast_adjust_max() - a_param_.contrast_adjust_min()) * randDouble();
  // if our random value is large enough to produce noticeable contrast:
  if (doContrast) {
    //LOG(WARNING) << "doContrast=" << doContrast << "  contrast factor=" << contrast;
    addWeighted(img, contrast, img, 0.0, 0.0, *result);
  } else {
    *result = img;
    contrast = 1.0;
  }

  return contrast;
}

template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::clip_pixels(Mat3v& img, const Dtype min_val, const Dtype max_val) {
  auto saturateImageZeroToOne = [](Dtype value, const Dtype min_val, const Dtype max_val) {
    return (value<min_val? min_val : (value>max_val? max_val: value));
  };
  //saturate
  const int cn = img.channels();
  for(int r=0; r<img.rows; r++) {
    for(int c=0; c<img.cols; c++) {
      Vec3v& img_data = (img)(r,c);
      for(int c=0; c<cn; c++) {
        img_data[c] = saturateImageZeroToOne(img_data[c], min_val, max_val);
      }
    }
  }

}

template<typename Dtype>
void ImageLabelTransformationLayer<Dtype>::check_labels(Mat3v& label, const Dtype min_val, const Dtype max_val) {
  auto checkLabel = [](Dtype value, const Dtype min_val, const Dtype max_val) {
    return (value<min_val? UINT8_MAX : (value>max_val? UINT8_MAX: value));
  };
  const int cn = label.channels();
  for(int r=0; r<label.rows; r++) {
    for(int c=0; c<label.cols; c++) {
      Vec3v& label_data = (label)(r,c);
      for(int c=0; c<cn; c++) {
        label_data[c] = checkLabel(label_data[c], min_val, max_val);
      }
    }
  }

}

INSTANTIATE_CLASS(ImageLabelTransformationLayer);
REGISTER_LAYER_CLASS(ImageLabelTransformation);

}  // namespace caffe

#endif  // USE_OPENCV
