#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <turbojpeg.h>

#include "caffe/blob.hpp"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using namespace boost::property_tree;  // NOLINT(build/namespaces)
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output)) << "Possible reasons: no disk space, "
      "no write permissions, the destination folder doesn't exist";
}

bool ReadFileToDatum(const string& filename, const int label, Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

/**
 * Decode Datum to cv::Mat
 * @param datum
 * @param color_mode -1 enforce gray, 0 deduce from datum, +1 enforce color
 * @param out
 */
vector<int> DecodeDatumToCVMat(const Datum& datum, int color_mode, cv::Mat& cv_img,
    bool shape_only, bool accurate_jpeg) {
  CHECK(datum.encoded()) << "Datum not encoded";
  const std::string& content = datum.data();
  const size_t content_size = content.size();
  return Decode(reinterpret_cast<const unsigned char*>(content.data()), content_size,
                color_mode, &cv_img, nullptr, 0, shape_only, accurate_jpeg);
}

void DecodeDatumToSignedBuf(const Datum& datum, int color_mode,
    char* buf, size_t buf_len, bool accurate_jpeg) {
  CHECK(datum.encoded()) << "Datum not encoded";
  const std::string& content = datum.data();
  const size_t content_size = content.size();
  Decode(reinterpret_cast<const unsigned char*>(content.data()), content_size,
         color_mode, nullptr, buf, buf_len, false, accurate_jpeg);
}

// decodes to either cv_img or buf
vector<int> Decode(const unsigned char* content, size_t content_size, int color_mode,
    cv::Mat* cv_img, char* buf, size_t buf_len, bool shape_only, bool accurate_jpeg) {
  if (content_size > 1
      && content[0] == 255
      && content[1] == 216) {  // probably jpeg
    int width, height, subsamp;
    auto *content_data = const_cast<unsigned char *>(content);

    tjhandle jpeg_decoder = tjInitDecompress();
    tjDecompressHeader2(jpeg_decoder, content_data, content_size, &width, &height, &subsamp);

    int ch = subsamp == TJSAMP_GRAY ? 1 : 3;
    if (color_mode < 0) {
      ch = 1;
    }
    if (!shape_only) {
      if (cv_img != nullptr) {
        cv_img->create(height, width, ch == 3 ? CV_8UC3 : CV_8UC1);
      } else {
        CHECK_EQ(ch * height * width, buf_len);
      }
      if (0 > tjDecompress2(jpeg_decoder, content_data, content_size,
                            cv_img != nullptr ? cv_img->ptr<unsigned char>()
                                              : reinterpret_cast<unsigned char*>(buf),
                            width,
                            0,
                            height,
                            ch == 3 ? TJPF_BGR : TJPF_GRAY,  // TODO RGB?
                            (accurate_jpeg ? TJFLAG_ACCURATEDCT : TJFLAG_FASTDCT) |
                            TJFLAG_NOREALLOC)) {
        return vector<int>{};
      }
    }
    tjDestroy(jpeg_decoder);
    return vector<int>{1, ch, height, width};
  }
  // probably not jpeg...
  std::vector<char> vec_data(content, content + content_size);
  const int flag = color_mode < 0 ? cv::IMREAD_GRAYSCALE :
                   (color_mode > 0 ? cv::IMREAD_COLOR : cv::IMREAD_ANYCOLOR);
  if (cv_img != nullptr && !shape_only) {
    *cv_img = cv::imdecode(vec_data, flag);
    return vector<int>{1, cv_img->channels(), cv_img->rows, cv_img->cols};
  }
  cv::Mat img = cv::imdecode(vec_data, flag);
  if (!shape_only) {
    CHECK_EQ(img.channels() * img.rows * img.cols, buf_len);
    std::memcpy(buf, img.data, buf_len);  // NOLINT(caffe/alt_fn)
  }
  return vector<int>{1, img.channels(), img.rows, img.cols};
}

cv::Mat ReadImageToCVMat(const string& filename,
    int height, int width, bool is_color, int short_side) {
  cv::Mat cv_img_origin;
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (!ifs) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  std::string content;
  ifs.seekg(0, std::ios::end);
  content.resize(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  ifs.read(&content.front(), content.size());
  ifs.close();

  cv::Mat cv_img;
  vector<int> shape = Decode(reinterpret_cast<const unsigned char*>(content.data()), content.size(),
                             is_color ? 1 : -1, &cv_img_origin, nullptr, 0, false, true);
  if (shape.size() == 0) {
    int cv_read_flag = is_color ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
    // Trying this one. It's slow but might decode better
    cv_img_origin = cv::imread(filename, cv_read_flag);
  }
  if (cv_img_origin.data) {
    if (is_color && cv_img_origin.channels() < 3) {
      cv::cvtColor(cv_img_origin, cv_img, CV_GRAY2RGB);
    }
    if (short_side > 0) {
      if (cv_img_origin.rows > cv_img_origin.cols) {
        width = short_side;
        height = cv_img_origin.rows * short_side / cv_img_origin.cols;
      } else {
        height = short_side;
        width = cv_img_origin.cols * short_side / cv_img_origin.rows;
      }
    }
    if (height <= 0 || width <= 0) {
      return cv_img.data ? cv_img : cv_img_origin;
    }
    cv::Size sz(width, height);
    if (cv_img.data) {
      cv::resize(cv_img, cv_img_origin, sz, 0., 0., CV_INTER_LINEAR);
      return cv_img_origin;
    }
    cv::resize(cv_img_origin, cv_img, sz, 0., 0., CV_INTER_LINEAR);
  } else {
    LOG(ERROR) << "Could not decode file " << filename;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename, const int height,
    const int width, const int min_dim, const int max_dim,
    const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (min_dim > 0 || max_dim > 0) {
    int num_rows = cv_img_origin.rows;
    int num_cols = cv_img_origin.cols;
    int min_num = std::min(num_rows, num_cols);
    int max_num = std::max(num_rows, num_cols);
    float scale_factor = 1;
    if (min_dim > 0 && min_num < min_dim) {
      scale_factor = static_cast<float>(min_dim) / min_num;
    }
    if (max_dim > 0 && static_cast<int>(scale_factor * max_num) > max_dim) {
      // Make sure the maximum dimension is less than max_dim.
      scale_factor = static_cast<float>(max_dim) / max_num;
    }
    if (scale_factor == 1) {
      cv_img = cv_img_origin;
    } else {
      cv::resize(cv_img_origin, cv_img, cv::Size(0, 0),
                 scale_factor, scale_factor);
    }
  } else if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename, const int height,
    const int width, const int min_dim, const int max_dim) {
  return ReadImageToCVMat(filename, height, width, min_dim, max_dim, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) ) {
        return ReadFileToDatum(filename, label, datum);
      }
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, *datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const int min_dim, const int max_dim,
    const bool is_color, const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, min_dim, max_dim,
                                    is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          !min_dim && !max_dim && matchExt(filename, encoding) ) {
        datum->set_channels(cv_img.channels());
        datum->set_height(cv_img.rows);
        datum->set_width(cv_img.cols);
        return ReadFileToDatum(filename, label, datum);
      }
      EncodeCVMatToDatum(cv_img, encoding, datum);
      datum->set_label(label);
      return true;
    }
    CVMatToDatum(cv_img, *datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}

void GetImageSize(const string& filename, int* height, int* width) {
  cv::Mat cv_img = cv::imread(filename);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return;
  }
  *height = cv_img.rows;
  *width = cv_img.cols;
}

bool ReadRichImageToAnnotatedDatum(const string& filename,
    const string& labelfile, const int height, const int width,
    const int min_dim, const int max_dim, const bool is_color,
    const string& encoding, const AnnotatedDatum_AnnotationType type,
    const string& labeltype, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum) {
  // Read image to datum.
  bool status = ReadImageToDatum(filename, -1, height, width,
                                 min_dim, max_dim, is_color, encoding,
                                 anno_datum->mutable_datum());
  if (status == false) {
    return status;
  }
  anno_datum->clear_annotation_group();
  if (!boost::filesystem::exists(labelfile)) {
    return true;
  }
  switch (type) {
    case AnnotatedDatum_AnnotationType_BBOX:
      int ori_height, ori_width;
      GetImageSize(filename, &ori_height, &ori_width);
      if (labeltype == "xml") {
        return ReadXMLToAnnotatedDatum(labelfile, ori_height, ori_width,
                                       name_to_label, anno_datum);
      } else if (labeltype == "json") {
        return ReadJSONToAnnotatedDatum(labelfile, ori_height, ori_width,
                                        name_to_label, anno_datum);
      } else if (labeltype == "txt") {
        return ReadTxtToAnnotatedDatum(labelfile, ori_height, ori_width,
                                       anno_datum);
      } else {
        LOG(FATAL) << "Unknown label file type.";
        return false;
      }
      break;
    default:
      LOG(FATAL) << "Unknown annotation type.";
      return false;
  }
}

//bool ReadFileToDatum(const string& filename, const int label,
//    Datum* datum) {
//  std::streampos size;
//
//  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
//  if (file.is_open()) {
//    size = file.tellg();
//    std::string buffer(size, ' ');
//    file.seekg(0, ios::beg);
//    file.read(&buffer[0], size);
//    file.close();
//    datum->set_data(buffer);
//    datum->set_label(label);
//    datum->set_encoded(true);
//    return true;
//  } else {
//    return false;
//  }
//}

// Parse VOC/ILSVRC detection annotation.
bool ReadXMLToAnnotatedDatum(const string& labelfile, const int img_height,
    const int img_width, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum) {
  ptree pt;
  read_xml(labelfile, pt);

  // Parse annotation.
  int width = 0, height = 0;
  try {
    height = pt.get<int>("annotation.size.height");
    width = pt.get<int>("annotation.size.width");
  } catch (const ptree_error &e) {
    LOG(WARNING) << "When parsing " << labelfile << ": " << e.what();
    height = img_height;
    width = img_width;
  }
  LOG_IF(WARNING, height != img_height) << labelfile <<
      " inconsistent image height.";
  LOG_IF(WARNING, width != img_width) << labelfile <<
      " inconsistent image width.";
  CHECK(width != 0 && height != 0) << labelfile <<
      " no valid image width/height.";
  int instance_id = 0;
  BOOST_FOREACH(ptree::value_type &v1, pt.get_child("annotation")) {
    ptree pt1 = v1.second;
    if (v1.first == "object") {
      Annotation* anno = NULL;
      bool difficult = false;
      ptree object = v1.second;
      BOOST_FOREACH(ptree::value_type &v2, object.get_child("")) {
        ptree pt2 = v2.second;
        if (v2.first == "name") {
          string name = pt2.data();
          if (name_to_label.find(name) == name_to_label.end()) {
            LOG(FATAL) << "Unknown name: " << name;
          }
          int label = name_to_label.find(name)->second;
          bool found_group = false;
          for (int g = 0; g < anno_datum->annotation_group_size(); ++g) {
            AnnotationGroup* anno_group =
                anno_datum->mutable_annotation_group(g);
            if (label == anno_group->group_label()) {
              if (anno_group->annotation_size() == 0) {
                instance_id = 0;
              } else {
                instance_id = anno_group->annotation(
                    anno_group->annotation_size() - 1).instance_id() + 1;
              }
              anno = anno_group->add_annotation();
              found_group = true;
            }
          }
          if (!found_group) {
            // If there is no such annotation_group, create a new one.
            AnnotationGroup* anno_group = anno_datum->add_annotation_group();
            anno_group->set_group_label(label);
            anno = anno_group->add_annotation();
            instance_id = 0;
          }
          anno->set_instance_id(instance_id++);
        } else if (v2.first == "difficult") {
          difficult = pt2.data() == "1";
          NormalizedBBox* bbox = anno->mutable_bbox();
          bbox->set_difficult(difficult);
        } else if (v2.first == "bndbox") {
          int xmin = pt2.get("xmin", 0);
          int ymin = pt2.get("ymin", 0);
          int xmax = pt2.get("xmax", 0);
          int ymax = pt2.get("ymax", 0);
          CHECK_NOTNULL(anno);
          LOG_IF(WARNING, xmin > width) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymin > height) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmax > width) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymax > height) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmin < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymin < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmax < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymax < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmin > xmax) << labelfile <<
              " bounding box irregular.";
          LOG_IF(WARNING, ymin > ymax) << labelfile <<
              " bounding box irregular.";
          // Store the normalized bounding box.
          NormalizedBBox* bbox = anno->mutable_bbox();
          bbox->set_xmin(static_cast<float>(xmin) / width);
          bbox->set_ymin(static_cast<float>(ymin) / height);
          bbox->set_xmax(static_cast<float>(xmax) / width);
          bbox->set_ymax(static_cast<float>(ymax) / height);
          bbox->set_difficult(difficult);
        }
      }
    }
  }
  return true;
}

// Parse MSCOCO detection annotation.
bool ReadJSONToAnnotatedDatum(const string& labelfile, const int img_height,
    const int img_width, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum) {
  ptree pt;
  read_json(labelfile, pt);

  // Get image info.
  int width = 0, height = 0;
  try {
    height = pt.get<int>("image.height");
    width = pt.get<int>("image.width");
  } catch (const ptree_error &e) {
    LOG(WARNING) << "When parsing " << labelfile << ": " << e.what();
    height = img_height;
    width = img_width;
  }
  LOG_IF(WARNING, height != img_height) << labelfile <<
      " inconsistent image height.";
  LOG_IF(WARNING, width != img_width) << labelfile <<
      " inconsistent image width.";
  CHECK(width != 0 && height != 0) << labelfile <<
      " no valid image width/height.";

  // Get annotation info.
  int instance_id = 0;
  BOOST_FOREACH(ptree::value_type& v1, pt.get_child("annotation")) {
    Annotation* anno = NULL;
    bool iscrowd = false;
    ptree object = v1.second;
    // Get category_id.
    string name = object.get<string>("category_id");
    if (name_to_label.find(name) == name_to_label.end()) {
      LOG(FATAL) << "Unknown name: " << name;
    }
    int label = name_to_label.find(name)->second;
    bool found_group = false;
    for (int g = 0; g < anno_datum->annotation_group_size(); ++g) {
      AnnotationGroup* anno_group =
          anno_datum->mutable_annotation_group(g);
      if (label == anno_group->group_label()) {
        if (anno_group->annotation_size() == 0) {
          instance_id = 0;
        } else {
          instance_id = anno_group->annotation(
              anno_group->annotation_size() - 1).instance_id() + 1;
        }
        anno = anno_group->add_annotation();
        found_group = true;
      }
    }
    if (!found_group) {
      // If there is no such annotation_group, create a new one.
      AnnotationGroup* anno_group = anno_datum->add_annotation_group();
      anno_group->set_group_label(label);
      anno = anno_group->add_annotation();
      instance_id = 0;
    }
    anno->set_instance_id(instance_id++);

    // Get iscrowd.
    iscrowd = object.get<int>("iscrowd", 0);

    // Get bbox.
    vector<float> bbox_items;
    BOOST_FOREACH(ptree::value_type& v2, object.get_child("bbox")) {
      bbox_items.push_back(v2.second.get_value<float>());
    }
    CHECK_EQ(bbox_items.size(), 4);
    float xmin = bbox_items[0];
    float ymin = bbox_items[1];
    float xmax = bbox_items[0] + bbox_items[2];
    float ymax = bbox_items[1] + bbox_items[3];
    CHECK_NOTNULL(anno);
    LOG_IF(WARNING, xmin > width) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin > height) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax > width) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax > height) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin > xmax) << labelfile <<
        " bounding box irregular.";
    LOG_IF(WARNING, ymin > ymax) << labelfile <<
        " bounding box irregular.";
    // Store the normalized bounding box.
    NormalizedBBox* bbox = anno->mutable_bbox();
    bbox->set_xmin(xmin / width);
    bbox->set_ymin(ymin / height);
    bbox->set_xmax(xmax / width);
    bbox->set_ymax(ymax / height);
    bbox->set_difficult(iscrowd);
  }
  return true;
}

// Parse plain txt detection annotation: label_id, xmin, ymin, xmax, ymax.
bool ReadTxtToAnnotatedDatum(const string& labelfile, const int height,
    const int width, AnnotatedDatum* anno_datum) {
  std::ifstream infile(labelfile.c_str());
  if (!infile.good()) {
    LOG(INFO) << "Cannot open " << labelfile;
    return false;
  }
  int label;
  float xmin, ymin, xmax, ymax;
  while (infile >> label >> xmin >> ymin >> xmax >> ymax) {
    Annotation* anno = NULL;
    int instance_id = 0;
    bool found_group = false;
    for (int g = 0; g < anno_datum->annotation_group_size(); ++g) {
      AnnotationGroup* anno_group = anno_datum->mutable_annotation_group(g);
      if (label == anno_group->group_label()) {
        if (anno_group->annotation_size() == 0) {
          instance_id = 0;
        } else {
          instance_id = anno_group->annotation(
              anno_group->annotation_size() - 1).instance_id() + 1;
        }
        anno = anno_group->add_annotation();
        found_group = true;
      }
    }
    if (!found_group) {
      // If there is no such annotation_group, create a new one.
      AnnotationGroup* anno_group = anno_datum->add_annotation_group();
      anno_group->set_group_label(label);
      anno = anno_group->add_annotation();
      instance_id = 0;
    }
    anno->set_instance_id(instance_id++);
    LOG_IF(WARNING, xmin > width) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin > height) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax > width) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax > height) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin > xmax) << labelfile <<
      " bounding box irregular.";
    LOG_IF(WARNING, ymin > ymax) << labelfile <<
      " bounding box irregular.";
    // Store the normalized bounding box.
    NormalizedBBox* bbox = anno->mutable_bbox();
    bbox->set_xmin(xmin / width);
    bbox->set_ymin(ymin / height);
    bbox->set_xmax(xmax / width);
    bbox->set_ymax(ymax / height);
    bbox->set_difficult(false);
  }
  return true;
}

bool ReadLabelFileToLabelMap(const string& filename, bool include_background,
    const string& delimiter, LabelMap* map) {
  // cleanup
  map->Clear();

  std::ifstream file(filename.c_str());
  string line;
  // Every line can have [1, 3] number of fields.
  // The delimiter between fields can be one of " :;".
  // The order of the fields are:
  //  name [label] [display_name]
  //  ...
  int field_size = -1;
  int label = 0;
  LabelMapItem* map_item;
  // Add background (none_of_the_above) class.
  if (include_background) {
    map_item = map->add_item();
    map_item->set_name("none_of_the_above");
    map_item->set_label(label++);
    map_item->set_display_name("background");
  }
  while (std::getline(file, line)) {
    vector<string> fields;
    fields.clear();
    boost::split(fields, line, boost::is_any_of(delimiter));
    if (field_size == -1) {
      field_size = fields.size();
    } else {
      CHECK_EQ(field_size, fields.size())
          << "Inconsistent number of fields per line.";
    }
    map_item = map->add_item();
    map_item->set_name(fields[0]);
    switch (field_size) {
      case 1:
        map_item->set_label(label++);
        map_item->set_display_name(fields[0]);
        break;
      case 2:
        label = std::atoi(fields[1].c_str());
        map_item->set_label(label);
        map_item->set_display_name(fields[0]);
        break;
      case 3:
        label = std::atoi(fields[1].c_str());
        map_item->set_label(label);
        map_item->set_display_name(fields[2]);
        break;
      default:
        LOG(FATAL) << "The number of fields should be [1, 3].";
        break;
    }
  }
  return true;
}

bool MapNameToLabel(const LabelMap& map, const bool strict_check,
    std::map<string, int>* name_to_label) {
  // cleanup
  name_to_label->clear();

  for (int i = 0; i < map.item_size(); ++i) {
    const string& name = map.item(i).name();
    const int label = map.item(i).label();
    if (strict_check) {
      if (!name_to_label->insert(std::make_pair(name, label)).second) {
        LOG(FATAL) << "There are many duplicates of name: " << name;
        return false;
      }
    } else {
      (*name_to_label)[name] = label;
    }
  }
  return true;
}

bool MapLabelToName(const LabelMap& map, const bool strict_check,
    std::map<int, string>* label_to_name) {
  // cleanup
  label_to_name->clear();

  for (int i = 0; i < map.item_size(); ++i) {
    const string& name = map.item(i).name();
    const int label = map.item(i).label();
    if (strict_check) {
      if (!label_to_name->insert(std::make_pair(label, name)).second) {
        LOG(FATAL) << "There are many duplicates of label: " << label;
        return false;
      }
    } else {
      (*label_to_name)[label] = name;
    }
  }
  return true;
}

bool MapLabelToDisplayName(const LabelMap& map, const bool strict_check,
    std::map<int, string>* label_to_display_name) {
  // cleanup
  label_to_display_name->clear();

  for (int i = 0; i < map.item_size(); ++i) {
    const string& display_name = map.item(i).display_name();
    const int label = map.item(i).label();
    if (strict_check) {
      if (!label_to_display_name->insert(
              std::make_pair(label, display_name)).second) {
        LOG(FATAL) << "There are many duplicates of label: " << label;
        return false;
      }
    } else {
      (*label_to_display_name)[label] = display_name;
    }
  }
  return true;
}
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  DecodeDatumToCVMat(datum, 0, cv_img, false);
  return cv_img;
}

// tests only, TODO: clean
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  DecodeDatumToCVMat(datum, is_color ? 1 : -1, cv_img, false);
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img;
    DecodeDatumToCVMat(*datum, 0, cv_img, false);
    CVMatToDatum(cv_img, *datum);
    return true;
  } else {
    return false;
  }
}

bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img;
    DecodeDatumToCVMat(*datum, is_color ? 1 : 0, cv_img, false);
    CVMatToDatum(cv_img, *datum);
    return true;
  } else {
    return false;
  }
}

void EncodeCVMatToDatum(const cv::Mat& cv_img, const string& encoding,
                        Datum* datum) {
  std::vector<uchar> buf;
  cv::imencode("."+encoding, cv_img, buf);
  datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                              buf.size()));
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_encoded(true);
}

vector<int> DatumToCVMat(const Datum& datum, cv::Mat& img, bool shape_only) {
  if (datum.encoded()) {
    LOG(FATAL) << "Datum encoded";
  }
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  if (shape_only) {
    return vector<int>{1, datum_channels, datum_height, datum_width};
  }
  const int datum_size = datum_channels * datum_height * datum_width;
  CHECK_GT(datum_channels, 0);
  CHECK_GT(datum_height, 0);
  CHECK_GT(datum_width, 0);
  img.create(datum_height, datum_width, CVFC<float>(datum_channels));
  CHECK_EQ(img.channels(), datum_channels);
  CHECK_EQ(img.rows, datum_height);
  CHECK_EQ(img.cols, datum_width);
  const std::string& datum_buf = datum.data();
  CHECK_EQ(datum_buf.size(), datum_size);
  // CHW -> HWC
  chw2hwc(datum_channels, datum_width, datum_height,
      reinterpret_cast<const unsigned char*>(&datum_buf.front()), img.ptr<float>(0));
  return vector<int>{1, datum_channels, datum_height, datum_width};
}


void CVMatToDatum(const cv::Mat& cv_img, Datum& datum) {
  const unsigned int img_channels = cv_img.channels();
  const unsigned int img_height = cv_img.rows;
  const unsigned int img_width = cv_img.cols;
  const unsigned int img_size = img_channels * img_height * img_width;
  CHECK_GT(img_channels, 0);
  CHECK_GT(img_height, 0);
  CHECK_GT(img_width, 0);
  string* buf = datum.release_data();
  if (buf == nullptr || buf->size() != img_size) {
    delete buf;
    buf = new string(img_size, 0);
  }
  unsigned char* buf_front = reinterpret_cast<unsigned char*>(&buf->front());
  // HWC -> CHW
  if (cv_img.depth() == CV_8U) {
    hwc2chw(img_channels, img_width, img_height, cv_img.ptr<unsigned char>(0), buf_front);
  } else if (cv_img.depth() == CV_32F) {
    hwc2chw(img_channels, img_width, img_height, cv_img.ptr<float>(0), buf_front);
  } else if (cv_img.depth() == CV_64F) {
    hwc2chw(img_channels, img_width, img_height, cv_img.ptr<double>(0), buf_front);
  }
  datum.set_allocated_data(buf);
  datum.set_channels(img_channels);
  datum.set_height(img_height);
  datum.set_width(img_width);
  datum.set_encoded(false);
}

}  // namespace caffe
