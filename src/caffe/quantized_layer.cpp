/*
 * base_quantization_layer.hpp
 *
 *  Created on: Oct 12, 2016
 *      Author: a0393608
 */

#include "caffe/quantized_layer.hpp"

namespace caffe {


template<typename Dtype>
void QuantizedLayer<Dtype>::Quantize_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.has_quantization_param()) {
    //LOG(INFO) << "Quantizing layer: " << this->layer_param_.name();
    const vector<shared_ptr<Blob<Dtype> > >& blobs = this->blobs();
    const QuantizationParameter& param = this->layer_param_.quantization_param();
    if (param.precision() != QuantizationParameter_Precision_FLOAT) {
      // Trim layer input
      if(param.quantize_layer_in()) {
        for (int i = 0; i < bottom.size(); ++i) {
          this->QuantizeLayerInputs_cpu(bottom[i]->mutable_cpu_data(), i, bottom[i]->count());
        }
      }

      // Trim weights
      if(param.quantize_layer_weights() && blobs.size() > 0) {
        this->QuantizeWeights_cpu(blobs[0]->mutable_cpu_data(), blobs[0]->count(), param.rounding_scheme(), true);
        if (blobs.size() > 1) { //if (this->bias_term_) {
          this->QuantizeWeights_cpu(blobs[1]->mutable_cpu_data(), blobs[1]->count(), param.rounding_scheme(), false);
        }
      }

      // Trim layer output
      if(param.quantize_layer_out()) {
        for (int i = 0; i < top.size(); ++i) {
          this->QuantizeLayerOutputs_cpu(top[i]->mutable_cpu_data(), top[i]->count());
        }
      }
    }
  }
}


template <typename Dtype>
void QuantizedLayer<Dtype>::QuantizeWeights_cpu(Dtype* data, const int count, const int rounding, bool clip) {
  const QuantizationParameter& param =  this->layer_param_.quantization_param();
  switch (param.precision()) {
  case QuantizationParameter_Precision_MINIFLOAT:
    Trim2MiniFloat_cpu(data, count, param.mant_bits(), param.exp_bits(), param.rounding_scheme());
    break;
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    Trim2FixedPoint_cpu(data, count, param.bw_weights(), param.rounding_scheme(), param.fl_weights(),
    		0, false, clip);
    break;
  case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
    Trim2IntegerPowerOf2_cpu(data, count, param.exp_min(), param.exp_max(),
    		param.rounding_scheme());
    break;
  case QuantizationParameter_Precision_FLOAT:
	break;
  default:
	 LOG(FATAL) << "Unknown trimming mode: " << param.precision() << " for layer:" << this->layer_param_.name();
    break;
  }
}

template <typename Dtype>
void QuantizedLayer<Dtype>::QuantizeLayerInputs_cpu(Dtype* data, const int blob_id,
      const int count) {
  const QuantizationParameter& param =  this->layer_param_.quantization_param();
  bool unsigned_layer_in = param.unsigned_layer_in_size()>0? param.unsigned_layer_in(blob_id): false;
  switch (param.precision()) {
    case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
      break;
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      if(param.fl_layer_in_size() > blob_id) {
        Trim2FixedPoint_cpu(data, count, param.bw_layer_in(), param.rounding_scheme(), param.fl_layer_in(blob_id),
    		  0, unsigned_layer_in, true);
      }
      break;
    case QuantizationParameter_Precision_MINIFLOAT:
      Trim2MiniFloat_cpu(data, count, param.mant_bits(), param.exp_bits(), param.rounding_scheme());
      break;
    case QuantizationParameter_Precision_FLOAT:
  	  break;
    default:
   	  LOG(FATAL) << "Unknown trimming mode: " << param.precision() << " for layer:" << this->layer_param_.name();
      break;
  }
}

template <typename Dtype>
void QuantizedLayer<Dtype>::QuantizeLayerOutputs_cpu(
      Dtype* data, const int count) {
  const QuantizationParameter& param =  this->layer_param_.quantization_param();
  switch (param.precision()) {
    case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
      break;
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_cpu(data, count, param.bw_layer_out(), param.rounding_scheme(), param.fl_layer_out(),
    		  0, param.unsigned_layer_out(), true);
      break;
    case QuantizationParameter_Precision_MINIFLOAT:
      Trim2MiniFloat_cpu(data, count, param.mant_bits(), param.exp_bits(), param.rounding_scheme());
      break;
    case QuantizationParameter_Precision_FLOAT:
  	  break;
    default:
	  LOG(FATAL) << "Unknown trimming mode: " << param.precision() << " for layer:" << this->layer_param_.name();
      break;
  }
}

template <typename Dtype>
void QuantizedLayer<Dtype>::Trim2FixedPoint_cpu(Dtype* data, const int cnt,
      const int bit_width, const int rounding, int fl, Dtype sparsity_threshold, bool unsigned_data, bool clip) {
  for (int index = 0; index < cnt; ++index) {

	data[index] = data[index] * powf(2, fl);

    // Saturate data
#if CLIP_QUANT
	  if(clip) {
	      int qrange = unsigned_data? bit_width :  (bit_width - 1);
	      Dtype max_data = +(powf(2, qrange) - 1);
	      Dtype min_data = unsigned_data? 0 : -(powf(2, qrange));
		  data[index] = std::max(std::min(data[index], max_data), min_data);
	  }
#endif
    // Round data
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = round(data[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      data[index] = floor(data[index] + RandUniform_cpu());
      break;
    default:
      break;
    }

    data[index] = data[index] * pow(2, -fl);

    if(sparsity_threshold != 0 && fabs(data[index]) < sparsity_threshold) {
		data[index] = 0;
    }
  }
}

typedef union {
  float d;
  struct {
    unsigned int mantisa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
} float_cast;

template <typename Dtype>
void QuantizedLayer<Dtype>::Trim2MiniFloat_cpu(Dtype* data, const int cnt,
      const int bw_mant, const int bw_exp, const int rounding) {
  for (int index = 0; index < cnt; ++index) {
    int bias_out = pow(2, bw_exp - 1) - 1;
    float_cast d2;
    // This casts the input to single precision
    d2.d = (float)data[index];
    int exponent=d2.parts.exponent - 127 + bias_out;
    double mantisa = d2.parts.mantisa;
    // Special case: input is zero or denormalized number
    if (d2.parts.exponent == 0) {
      data[index] = 0;
      return;
    }
    // Special case: denormalized number as output
    if (exponent < 0) {
      data[index] = 0;
      return;
    }
    // Saturation: input float is larger than maximum output float
    int max_exp = pow(2, bw_exp) - 1;
    int max_mant = pow(2, bw_mant) - 1;
    if (exponent > max_exp) {
      exponent = max_exp;
      mantisa = max_mant;
    } else {
      // Convert mantissa from long format to short one. Cut off LSBs.
      double tmp = mantisa / pow(2, 23 - bw_mant);
      switch (rounding) {
      case QuantizationParameter_Rounding_NEAREST:
        mantisa = round(tmp);
        break;
      case QuantizationParameter_Rounding_STOCHASTIC:
        mantisa = floor(tmp + RandUniform_cpu());
        break;
      default:
        break;
      }
    }
    // Assemble result
    data[index] = pow(-1, d2.parts.sign) * ((mantisa + pow(2, bw_mant)) /
        pow(2, bw_mant)) * pow(2, exponent - bias_out);
	}
}

template <typename Dtype>
void QuantizedLayer<Dtype>::Trim2IntegerPowerOf2_cpu(Dtype* data,
      const int cnt, const int min_exp, const int max_exp, const int rounding) {
	for (int index = 0; index < cnt; ++index) {
    float exponent = log2f((float)fabs(data[index]));
    int sign = data[index] >= 0 ? 1 : -1;
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      exponent = round(exponent);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      exponent = floorf(exponent + RandUniform_cpu());
      break;
    default:
      break;
    }
    exponent = std::max(std::min(exponent, (float)max_exp), (float)min_exp);
    data[index] = sign * pow(2, exponent);
	}
}


template <typename Dtype>
double QuantizedLayer<Dtype>::RandUniform_cpu(){
  return rand() / (RAND_MAX+1.0);
}


template void QuantizedLayer<float>::Quantize_cpu(const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void QuantizedLayer<double>::Quantize_cpu(const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);


}

