# Caffe-jacinto
###### Caffe-jacinto - embedded deep learning framework

Caffe-jacinto is a fork of [NVIDIA/caffe](https://github.com/NVIDIA/caffe), which in-turn is derived from [BVLC/Caffe](https://github.com/BVLC/caffe). The modifications in this fork enable training of sparse, quantized CNN models - resulting in low complexity models that can be used in embedded platforms. 

For example, the semantic segmentation example shows how to train a model that is nearly 80% sparse (only 20% non-zero coefficients) and 8-bit quantized. This reduces the complexity of convolution layers by <b>5x</b>. An inference engine designed to efficiently take advantage of sparsity can run <b>significantly faster</b> by using such a model. 

Care has to be taken to strike the right balance between quality and speedup. We have obtained more than 4x overall speedup for CNN inference on embedded device by applying sparsity. Since 8-bit multiplier is sufficient (instead of floating point), the speedup can be even higher on some platforms.

### Prerequisite
* The scripts for training and infering models using caffe-jacinto is placed in [caffe-jacinto-models](https://github.com/tidsp/caffe-jacinto-models). Please see it's documentation before proceeding further.

### Installation
* After cloning the source code, switch to the branch caffe-0.15, if it is not checked out already.
-- *git checkout caffe-0.15*

* Please see the [installation instructions](INSTALL.md) for installing the dependencies and building the code. 

### Features

New layers and options have been added to support sparsity and quantization. 

Note that Caffe-jacinto does not directly support any embedded/low-power device. But the models trained by it can be used for fast inference on such a device due to the sparsity and quantization.

###### Additional layers
* ImageLabelData and ImageLabelListData layers have been added to support training for semantic segmentation.

###### Sparsity
* One major feature is quickly and accurately introducing sparsity in the coefficients. These include  zeroing out of small coefficients during training, training without updating the zero coefficients (sparse update). 
* See another work that uses sparse update: caffe-scnn [paper](https://arxiv.org/abs/1608.03665), [code](https://github.com/wenwei202/caffe/tree/scnn)

###### Quantization
* Dynamic -8 bit fixed point quantization, improved from Ristretto [paper](https://arxiv.org/abs/1605.06402), [code](https://github.com/pmgysel/caffe)

### Examples
See [caffe-jacinto-models](https://github.com/tidsp/caffe-jacinto-models) for several examples.

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
