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

//If BN shares the blob with conv output, the clipping may have to be implemented differently.
//Disable it for the time being.
#define CLIP_QUANT 1


template <typename Dtype>
class QuantizedLayer : public Layer<Dtype> {
public:
  QuantizedLayer(const LayerParameter& layer_param) : Layer<Dtype>(layer_param) { }

  void Quantize_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void Quantize_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  void QuantizeLayerOutputs_cpu(Dtype* data, const int count);
  void QuantizeLayerInputs_cpu(Dtype* data, const int blob_id, const int count);
  void QuantizeLayerOutputs_gpu(Dtype* data, const int count);
  void QuantizeLayerInputs_gpu(Dtype* data, const int blob_id, const int count);
  void QuantizeWeights_cpu(Dtype* data, const int count, const int rounding, bool clip);
  void QuantizeWeights_gpu(Dtype* data, const int count, const int rounding, bool clip);
  /**
   * @brief Trim data to fixed point.
   * @param fl The number of bits in the fractional part.
   */
  void Trim2FixedPoint_cpu(Dtype* data, const int cnt, const int bit_width,
      const int rounding, int fl, Dtype sparsity_threshold, bool unsigned_data, bool clip);
  void Trim2FixedPoint_gpu(Dtype* data, const int cnt, const int bit_width,
      const int rounding, int fl, Dtype sparsity_threshold, bool unsigned_data, bool clip);
  /**
   * @brief Trim data to minifloat.
   * @param bw_mant The number of bits used to represent the mantissa.
   * @param bw_exp The number of bits used to represent the exponent.
   */
  void Trim2MiniFloat_cpu(Dtype* data, const int cnt, const int bw_mant,
      const int bw_exp, const int rounding);
  void Trim2MiniFloat_gpu(Dtype* data, const int cnt, const int bw_mant,
      const int bw_exp, const int rounding);
  /**
   * @brief Trim data to integer-power-of-two numbers.
   * @param min_exp The smallest quantized value is 2^min_exp.
   * @param min_exp The largest quantized value is 2^max_exp.
   */
  void Trim2IntegerPowerOf2_cpu(Dtype* data, const int cnt, const int min_exp,
      const int max_exp, const int rounding);
  void Trim2IntegerPowerOf2_gpu(Dtype* data, const int cnt, const int min_exp,
      const int max_exp, const int rounding);
	  
  /**
   * @brief Generate random number in [0,1) range.
   */
  inline double RandUniform_cpu();	  
};

}

#endif /* BASE_QUANTIZATION_LAYER_HPP_ */
