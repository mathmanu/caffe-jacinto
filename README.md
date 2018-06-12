# Caffe-jacinto
###### Caffe-jacinto - embedded deep learning framework

Caffe-jacinto is a fork of [NVIDIA/caffe](https://github.com/NVIDIA/caffe), which in-turn is derived from [BVLC/Caffe](https://github.com/BVLC/caffe). The modifications in this fork enable training of sparse, quantized CNN models - resulting in low complexity models that can be used in embedded platforms. 

For example, the semantic segmentation example (see below) shows how to train a model that is nearly 80% sparse (only 20% non-zero coefficients) and 8-bit quantized. This reduces the complexity of convolution layers by <b>5x</b>. An inference engine designed to efficiently take advantage of sparsity can run <b>significantly faster</b> by using such a model. 

Care has to be taken to strike the right balance between quality and speedup. We have obtained more than 4x overall speedup for CNN inference on embedded device by applying sparsity. Since 8-bit multiplier is sufficient (instead of floating point), the speedup can be even higher on some platforms. See the section on quantization below for more details.

**Important note - Support for SSD Object detection has been added. The relevant SSD layers have been ported over from the [original Caffe SSD implementation](https://github.com/weiliu89/caffe/tree/ssd).** This is probably the first time that SSD object detection is added to a fork of [NVIDIA/caffe](https://github.com/NVIDIA/caffe). This enables fast training of SSD object detection with all the additional speedup benefits that [NVIDIA/caffe](https://github.com/NVIDIA/caffe) offers. 

**Examples for training and inference (image classification, semantic segmentation and SSD object detection) are in [tidsp/caffe-jacinto-models](https://github.com/tidsp/caffe-jacinto-models).**

### Installation
* After cloning the source code, switch to the branch caffe-0.17, if it is not checked out already.
-- *git checkout caffe-0.17*

* Please see the [installation instructions](INSTALL.md) for installing the dependencies and building the code. 

### Training procedure
**After cloning and building this source code, please visit [tidsp/caffe-jacinto-models](https://github.com/tidsp/caffe-jacinto-models) to do the training.**

### Additional Information (can be skipped)

**SSD Object detection is supported. The relevant SSD layers have been ported over from the [original Caffe SSD implementation](https://github.com/weiliu89/caffe/tree/ssd).** Note: caffe-0.16 branch allows us to set different types (float, float16 for forward, backward and math types). However for the SSD specific layers, forward, backward and math must use the same type - this limitation can probably be overcome by spending some more time in the porting - but it doesn't look like a serious limitation.

New layers and options have been added to support sparsity and quantization. A brief explanation is given in this section, but more details can be found by [clicking here](FEATURES.md). 

Note that Caffe-jacinto does not directly support any embedded/low-power device. But the models trained by it can be used for fast inference on such a device due to the sparsity and quantization.

###### Additional layers
* ImageLabelData and IOUAccuracy layers have been added to train for semantic segmentation.

###### Sparsity
* Sparse training methods: zeroing out of small coefficients during training, or fine tuning without updating the zero coefficients - similar to caffe-scnn [paper](https://arxiv.org/abs/1608.03665), [code](https://github.com/wenwei202/caffe/tree/scnn). It is possible to set a target sparsity and the training will try to achieve that.
* Measuring sparsity in convolution layers while training is in progress. 
* Thresholding tool to zero-out some convolution weights in each layer to attain certain sparsity in each layer.

###### Quantization
* **Estimate the accuracy drop by simulating quantization. Note that caffe-jacinto does not actually do quantization - it only simulates the accuracy loss due to quantization - by quantizing the coefficients and activations and then converting it back to float.** And embedded implementation can use the methods used here to achieve speedup by using only integer arithmetic.
* Variuos options are supported to control the quantization. Important features include: power of 2 quantization, non-power of 2 quantization, bitwidths, applying of offset to control bias around zero. See definition of NetQuantizationParameter for more details.
* Dynamic -8 bit fixed point quantization, improved from Ristretto [paper](https://arxiv.org/abs/1605.06402), [code](https://github.com/pmgysel/caffe).

###### Absorbing Batch Normalization into convolution weights
* A tool is provided to absorb batch norm values into convolution weights. This may help to speedup inference. This will also help if Batch Norm layers are not supported in an embedded implementation.

### Acknowledgements
This repository was forked from [NVIDIA/caffe](http://www.github.com/NVIDIA/caffe) and we have added several enhancements on top of it. We acknowledge use of code from other soruces as listed below, and sincerely thank their authors. See the [LICENSE](/LICENSE) file for the COPYRIGHT and LICENSE notices.
* [BVLC/caffe](https://github.com/BVLC/caffe) - base code.
* [NVIDIA/caffe](http://www.github.com/NVIDIA/caffe) - base code.
* [weiliu89/caffe/tree/ssd](https://github.com/weiliu89/caffe/tree/ssd) - Caffe SSD Object Detection source code and related scripts, which were later incorporated into [NVIDIA/caffe](http://www.github.com/NVIDIA/caffe).
* [Ristretto](https://github.com/pmgysel/caffe) - Quantization accuracy simulation
* [dilation](https://github.com/fyu/dilation) - Semantic Segmentation data loading layer ImageLabelListData layer (not used in the latest branch) and some parameters.
* [MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe) - Mobilenet scripts are inspired by Mobilenet-Caffe pre-trained models and scripts.
* TODO in the next release (not yet added): [sp2823/caffe](https://github.com/sp2823/caffe), [BVCL/caffe/pull/5665](https://github.com/BVLC/caffe/pull/5665) - ConvolutionDepthwise layer for faster depthwise separable convolutions.
* TODO in the next release (not yet added): [drnikolaev/caffe](https://github.com/drnikolaev/caffe) - experimental commits that are not yet integrated into [NVIDIA/caffe](http://www.github.com/NVIDIA/caffe).

<br>
The following sections are kept as it is from the original Caffe.
# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu))
and community contributors.

# NVCaffe

NVIDIA Caffe ([NVIDIA Corporation &copy;2017](http://nvidia.com)) is an NVIDIA-maintained fork
of BVLC Caffe tuned for NVIDIA GPUs, particularly in multi-GPU configurations.
Here are the major features:
* **16 bit (half) floating point train and inference support**.
* **Mixed-precision support**. It allows to store and/or compute data in either 
64, 32 or 16 bit formats. Precision can be defined for every layer (forward and 
backward passes might be different too), or it can be set for the whole Net.
* **Layer-wise Adaptive Rate Control (LARC) and adaptive global gradient scaler** for better
 accuracy, especially in 16-bit training.
* **Integration with  [cuDNN](https://developer.nvidia.com/cudnn) v7**.
* **Automatic selection of the best cuDNN convolution algorithm**.
* **Integration with v2.2 of [NCCL library](https://github.com/NVIDIA/nccl)**
 for improved multi-GPU scaling.
* **Optimized GPU memory management** for data and parameters storage, I/O buffers 
and workspace for convolutional layers.
* **Parallel data parser, transformer and image reader** for improved I/O performance.
* **Parallel back propagation and gradient reduction** on multi-GPU systems.
* **Fast solvers implementation with fused CUDA kernels for weights and history update**.
* **Multi-GPU test phase** for even memory load across multiple GPUs.
* **Backward compatibility with BVLC Caffe and NVCaffe 0.15 and higher**.
* **Extended set of optimized models** (including 16 bit floating point examples).


## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

## Useful notes

Libturbojpeg library is used since 0.16.5. It has a packaging bug. Please execute the following (required for Makefile, optional for CMake):
```
sudo apt-get install libturbojpeg
sudo ln -s /usr/lib/x86_64-linux-gnu/libturbojpeg.so.0.1.0 /usr/lib/x86_64-linux-gnu/libturbojpeg.so
```
