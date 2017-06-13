# Caffe-jacinto
###### Caffe-jacinto - embedded deep learning framework

Caffe-jacinto is a fork of [NVIDIA/caffe](https://github.com/NVIDIA/caffe), which in-turn is derived from [BVLC/Caffe](https://github.com/BVLC/caffe). The modifications in this fork enable training of sparse, quantized CNN models - resulting in low complexity models that can be used in embedded platforms. 

For example, the semantic segmentation example (see below) shows how to train a model that is nearly 80% sparse (only 20% non-zero coefficients) and 8-bit quantized. This reduces the complexity of convolution layers by <b>5x</b>. An inference engine designed to efficiently take advantage of sparsity can run <b>significantly faster</b> by using such a model. 

Care has to be taken to strike the right balance between quality and speedup. We have obtained more than 4x overall speedup for CNN inference on embedded device by applying sparsity. Since 8-bit multiplier is sufficient (instead of floating point), the speedup can be even higher on some platforms.

### Installation
* After cloning the source code, switch to the branch caffe-0.16, if it is not checked out already.
-- *git checkout caffe-0.16*

* Please see the [installation instructions](INSTALL.md) for installing the dependencies and building the code. 

### Features

New layers and options have been added to support sparsity and quantization. A brief explanation is given in this section, but more details can be found by [clicking here](FEATURES.md).

Note that Caffe-jacinto does not directly support any embedded/low-power device. But the models trained by it can be used for fast inference on such a device due to the sparsity and quantization.

###### Additional layers
* ImageLabelData and IOUAccuracy layers have been added to train for semantic segmentation.

###### Sparsity
* Measuring sparsity in convolution layers while training is in progress. 
* Thresholding tool to zero-out some convolution weights in each layer to attain certain sparsity in each layer.
* Sparse training methods: zeroing out of small coefficients during training, or fine tuning without updating the zero coefficients - similar to caffe-scnn [paper](https://arxiv.org/abs/1608.03665), [code](https://github.com/wenwei202/caffe/tree/scnn)

###### Quantization
* Collecting statistics (range of weights) to enable quantization
* Dynamic -8 bit fixed point quantization, improved from Ristretto [paper](https://arxiv.org/abs/1605.06402), [code](https://github.com/pmgysel/caffe)

###### Absorbing Batch Normalization into convolution weights
* A tool is provided to absorb batch norm values into convolution weights. This may help to speedup inference. This will also help if Batch Norm layers are not supported in an embedded implementation.

### Examples
###### Semantic segmentation:
* Note that ImageNet training (see below) is recommended before doing this segmentation training to create the pre-trained weights. The segmentation training will read the ImageNet trained caffemodel for doing the fine tuning on segmentation. However it is possible to directly do segmentation training without ImageNet training, but the quality might be inferior.
* [Train sparse, quantized CNN for semantic segmentation](examples/tidsp/docs/Cityscapes_Segmentation_README.md) on the cityscapes dataset. Inference script is also provided to test out the final model.

###### Classification:
* [Training on ILSVRC ImageNet dataset](examples/tidsp/docs/Imagenet_Classification_README.md). The 1000 class ImageNet trained weights is useful for fine tuning other tasks.
* [Train sparse, quantized CNN on cifar10 dataset](examples/tidsp/docs/Cifar10_Classification_README.md) for classification. Note that this is just a toy example and no inference script is provided to test the final model.

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
* **Integration with  [cuDNN](https://developer.nvidia.com/cudnn) v6**.
* **Automatic selection of the best cuDNN convolution algorithm**.
* **Integration with v1.3.4 of [NCCL library](https://github.com/NVIDIA/nccl)**
 for improved multi-GPU scaling.
* **Optimized GPU memory management** for data and parameters storage, I/O buffers 
and workspace for convolutional layers.
* **Parallel data parser and transformer** for improved I/O performance.
* **Parallel back propagation and gradient reduction** on multi-GPU systems.
* **Fast solvers implementation with fused CUDA kernels for weights and history update**.
* **Multi-GPU test phase** for even memory load across multiple GPUs.
* **Backward compatibility with BVLC Caffe and NVCaffe 0.15**.
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
