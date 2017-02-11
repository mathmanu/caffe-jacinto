#ifndef IMAGE_LABEL_TRANSFORMATION_HPP
#define IMAGE_LABEL_TRANSFORMATION_HPP

#include <boost/array.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>




struct ImageLabelTransformationAugmentSelection;

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class ImageLabelTransformationLayer : public Layer<Dtype> {
 public:
  typedef cv::Size2i Size2i;
  typedef cv::Size_<Dtype> Size2v;
  typedef cv::Point2i Point2i;
  typedef cv::Point_<Dtype> Point2v;
  typedef cv::Rect Rect;
  typedef cv::Rect_<Dtype> Rectv;
  typedef cv::Vec<Dtype, 3> Vec3v;
  typedef cv::Mat_<cv::Vec<Dtype, 1> > Mat1v;
  typedef cv::Mat_<Vec3v> Mat3v;

  explicit ImageLabelTransformationLayer(const LayerParameter& param);

  virtual ~ImageLabelTransformationLayer() {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SegmentationTransformation"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  virtual void Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {}

  void transform(
    const Mat3v& img,
    const Mat3v& label,
    Mat3v* img_aug,
    Mat3v* label_aug
);


  /**
   * @return a Dtype from [0..1].
   */
  Dtype randDouble();

  bool augmentation_flip(
    const Mat3v& img_src,
    Mat3v* img_aug,
    const Mat3v& label_src,
    Mat3v* label_aug
  );
  float augmentation_rotate(
      const Mat3v& img_src,
      Mat3v* img_aug,
      const Mat3v& label_src,
      Mat3v* label_aug);
  float augmentation_scale(
    const Mat3v& img_src,
    Mat3v* img_dst,
    const Mat3v& label_src,
    Mat3v* label_dst
);
  void transform_scale(
    const Mat3v& img,
    Mat3v* img_temp,
    const Mat3v& label,
    Mat3v* label_temp,
    const cv::Size& size
  );
  void transform_scale_adaptive(
    const Mat3v& img,
    Mat3v* img_temp,
    const Mat3v& label,
    Mat3v* label_temp,
    const cv::Size& size
  );
  Point2i augmentation_crop(
    const Mat3v& img_src,
    Mat3v* img_dst,
    const Mat3v& label_src,
    Mat3v* label_dst
);

  void transform_crop(
    const Mat3v& img_src,
    Mat3v* img_dst,
    const Mat3v& label_src,
    Mat3v* label_dst,
    Rect inner,
    Size2i dst_size,
    Point2i outer
) const;

  float augmentation_hueRotation(
      const Mat3v& img,
      Mat3v* result);

  float augmentation_desaturation(
      const Mat3v& img,
      Mat3v* result);

  float augmentation_contrast(
      const Mat3v& img,
      Mat3v* result);

  void clip_pixels(Mat3v& img, const Dtype min_val, const Dtype max_val);
  void check_labels(Mat3v& label, const Dtype min_val, const Dtype max_val);

  Mat1v getTransformationMatrix(Rect region, Dtype rotation) const;
  Rect getBoundingRect(Rect region, Dtype rotation) const;
  void matToBlob(const Mat3v& source, Dtype* destination) const;
  void matsToBlob(const vector<Mat3v>& source, Blob<Dtype>* destination) const;

  void matToRoundBlob1(const Mat3v& source, Dtype* destination) const;
  void matsToRoundBlob1(const vector<Mat3v>& source, Blob<Dtype>* destination) const;

  vector<Mat3v> blobToMats(const Blob<Dtype>& image) const;

  Mat3v dataToMat(
      const Dtype* _data,
      Size2i dimensions) const;

  vector<Mat3v> blobToMats1(const Blob<Dtype>& image) const;

  Mat3v dataToMat1(
      const Dtype* _data,
      Size2i dimensions) const;

  void retrieveMeanImage(Size2i dimensions = Size2i());
  void retrieveMeanChannels();

  void meanSubtract(Mat3v* source) const;
  void pixelMeanSubtraction(Mat3v* source) const;
  void channelMeanSubtraction(Mat3v* source) const;

  ImageLabelTransformationParameter a_param_;
  TransformationParameter t_param_;

  Phase phase_;

  Mat3v data_mean_;
  boost::array<Dtype, 3> mean_values_;
  shared_ptr<Caffe::RNG> rng_;
};

}  // namespace caffe

#endif /* IMAGE_LABEL_TRANSFORMATION_HPP */
