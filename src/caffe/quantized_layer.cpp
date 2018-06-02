/*
 * base_quantization_layer.hpp
 *
 *  Created on: Oct 12, 2016
 *      Author: a0393608
 */

#include "caffe/quantized_layer.hpp"

namespace caffe {


template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::Quantize_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  if (this->layer_param_.has_quantization_param()) {
    //LOG(INFO) << "Quantizing layer: " << this->layer_param_.name();
    const vector<shared_ptr<Blob > >& blobs = this->blobs();
    QuantizationParameter& param = *this->layer_param_.mutable_quantization_param();
    if (param.precision() != QuantizationParameter_Precision_FLOAT) {
      // Trim layer input
      for (int i = 0; i < std::min<int>(param.qparam_in_size(),bottom.size()); ++i) {
        if(param.qparam_in(i).quantize()) {
          this->QuantizeLayerInputs_cpu(bottom[i]->mutable_cpu_data<Ftype>(), i, bottom[i]->count());
        }
      }

      // Trim weights - do it only at the start of quantization
      for(int blob_id=0; blob_id<blobs.size(); blob_id++) {
    	if(param.qparam_w(blob_id).quantize() && param.quantized_infer_count() == 0) {
    	    bool clip = (blob_id == 0);
            this->QuantizeWeights_cpu(blobs[blob_id]->mutable_cpu_data<Ftype>(), blob_id, blobs[blob_id]->count(), clip);
        }
      }

      // Trim layer output
      for(int i=0; i<top.size(); i++) {
          if(param.qparam_out(i).quantize()) {
              this->QuantizeLayerOutputs_cpu(top[i]->mutable_cpu_data<Ftype>(), i, top[i]->count());
          }
      }
    }
  }
}


template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::QuantizeWeights_cpu(Ftype* data, const int blob_id, const int count, bool clip) {
  const QuantizationParameter& param =  this->layer_param_.quantization_param();
  const QuantizationParameter::QParams& qparam_w = param.qparam_w(blob_id);
  switch (param.precision()) {
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    Trim2FixedPoint_cpu(data, count, param.power2_scale_weights(), qparam_w.bitwidth(),
        param.rounding_scheme(), qparam_w.fracbits(), qparam_w.scale_target(),
        qparam_w.offset(), qparam_w.unsigned_quant(), clip);
    break;
  case QuantizationParameter_Precision_FLOAT:
	break;
  default:
	 LOG(FATAL) << "Unknown trimming mode: " << param.precision() << " for layer:" << this->layer_param_.name();
    break;
  }
}



template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::QuantizeLayerInputs_cpu(Ftype* data, const int blob_id,
      const int count) {
  const QuantizationParameter& param =  this->layer_param_.quantization_param();
  const QuantizationParameter::QParams& qparam_in = param.qparam_in(blob_id);
  switch (param.precision()) {
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_cpu(data, count, param.power2_scale_activations(), qparam_in.bitwidth(),
          param.rounding_scheme(), qparam_in.fracbits(), qparam_in.scale_target(),
          qparam_in.offset(), qparam_in.unsigned_quant(), true);
      break;
    case QuantizationParameter_Precision_FLOAT:
  	  break;
    default:
   	  LOG(FATAL) << "Unknown trimming mode: " << param.precision() << " for layer:" << this->layer_param_.name();
      break;
  }
}

template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::QuantizeLayerOutputs_cpu(
      Ftype* data, const int blob_id, const int count) {
  const QuantizationParameter& param =  this->layer_param_.quantization_param();
  const QuantizationParameter::QParams& qparam_out = param.qparam_out(blob_id);
  switch (param.precision()) {
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_cpu(data, count, param.power2_scale_activations(), qparam_out.bitwidth(),
          param.rounding_scheme(), qparam_out.fracbits(), qparam_out.scale_target(),
          qparam_out.offset(), qparam_out.unsigned_quant(), true);
      break;
    case QuantizationParameter_Precision_FLOAT:
  	  break;
    default:
	  LOG(FATAL) << "Unknown trimming mode: " << param.precision() << " for layer:" << this->layer_param_.name();
      break;
  }
}

template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::Trim2FixedPoint_cpu(Ftype* data, const int cnt, bool power2_range, const int bitwidth,
    const int rounding, int fracbits, float scale, float offset, bool unsigned_quant, bool clip) {
  float inv_scale = 1.0f/scale;

  int qrange = unsigned_quant? bitwidth :  (bitwidth - 1);
  Ftype max_data = +(powf(2, qrange) - 1);
  Ftype min_data = unsigned_quant? 0 : -(powf(2, qrange));

  for (int index = 0; index < cnt; ++index) {
    data[index] = (data[index] * scale) + offset;

    // Saturate data
    if(clip) {
      data[index] = std::max(std::min(data[index], max_data), min_data);
    }

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

    data[index] = (data[index] - offset) * inv_scale;
  }
}

template<typename Ftype, typename Btype>
double QuantizedLayer<Ftype, Btype>::RandUniform_cpu(){
  return rand() / (RAND_MAX+1.0);
}

template void QuantizedLayer<double, double>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<double, float>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);

template void QuantizedLayer<float, double>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float, float>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);

#ifndef CPU_ONLY
template void QuantizedLayer<double, float16>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float, float16>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);

template void QuantizedLayer<float16, double>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float16, float>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float16, float16>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
#endif

}

