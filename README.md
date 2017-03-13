# Caffe-jacinto
###### Caffe-jacinto - embedded deep learning framework

Caffe-jacinto is a fork of [NVIDIA/caffe](https://github.com/NVIDIA/caffe), which in-turn is derived from [BVLC/Caffe](https://github.com/BVLC/caffe). The modifications in this fork enable training of sparse, quantized CNN models - resulting in low complexity models that can be used in embedded platforms.

For example, the semantic segmentation example (see below) shows how to train a model that is nearly 80% sparse (only 20% non-zero coefficients) and 8-bit quantized. An inference engine designed to efficiently take advantage of sparsity can run up to <b>5x faster</b> by using such a model. Since 8-bit multiplier is sufficient (instead of floating point), the speedup can be even higher on some platforms.

### Installation
* After cloning the source code, switch to the branch caffe-0.15, if it is not checked out already.
-- *git checkout caffe-0.15*

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
* [Train sparse, quantized CNN for semantic segmentation](examples/tidsp/models/sparse/cityscapes_segmentation/README.md) on the cityscapes dataset. Inference script is also provided to test out the final model.

###### Classification:
* [Training on ILSVRC ImageNet dataset](examples/tidsp/models/sparse/imagenet_classification/README.md). The 1000 class ImageNet trained weights is useful for fine tuning other tasks.
* [Train sparse, quantized CNN on cifar10 dataset](examples/tidsp/models/sparse/cifar10_classification/README.md) for classification. Note that this is just a toy example and no inference script is provided to test the final model.

<br>
The following sections are kept as it is from the original Caffe.

# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

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
