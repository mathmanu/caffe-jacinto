/*
 * base_quantization_layer.hpp
 *
 *  Created on: Oct 12, 2016
 *      Author: a0393608
 */

#ifndef BASE_QUANTIZATION_LAYER_HPP_
#define BASE_QUANTIZATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
class QuantizedLayer : public Layer<Ftype, Btype> {
public:
  QuantizedLayer(const LayerParameter& layer_param) : Layer<Ftype, Btype>(layer_param) { }

  void Quantize_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  void Quantize_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);

  void QuantizeLayerOutputs_cpu(Ftype* data, const int blob_id, const int count);
  void QuantizeLayerInputs_cpu(Ftype* data, const int blob_id, const int count);
  void QuantizeLayerOutputs_gpu(Ftype* data, const int blob_id, const int count);
  void QuantizeLayerInputs_gpu(Ftype* data, const int blob_id, const int count);
  void QuantizeWeights_cpu(Ftype* data, const int blob_id, const int count, bool clip);
  void QuantizeWeights_gpu(Ftype* data, const int blob_id, const int count, bool clip);

  /**
   * @brief Trim data to fixed point.
   * @param fl The number of bits in the fractional part.
   */
  void Trim2FixedPoint_cpu(Ftype* data, const int cnt, bool power2_range, const int bitwidth,
      const int rounding, int fracbits, float scale, float offset, bool unsigned_quant, bool clip);
  void Trim2FixedPoint_gpu(Ftype* data, const int cnt, bool power2_range, const int bitwidth,
      const int rounding, int fracbits, float scale, float offset, bool unsigned_quant, bool clip);
	  
  /**
   * @brief Generate random number in [0,1) range.
   */
  inline double RandUniform_cpu();	  
};

}

#endif /* BASE_QUANTIZATION_LAYER_HPP_ */
