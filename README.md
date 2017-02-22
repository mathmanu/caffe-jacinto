# Caffe-jacinto
Caffe-jacinto is a fork of [NVIDIA/caffe](https://github.com/NVIDIA/caffe) that enables training of sparse, quantized CNN models.

After cloning the source code, switch to the branch caffe-0.15, if you are not on it already.

The procedure for installation and usage of Caffe-jacinto is quite similar to [NVIDIA/Caffe](https://github.com/NVIDIA/caffe), which in-turn is derived from [BVLC/Caffe](https://github.com/BVLC/caffe). Please see the tutorial documentations in their websites before proceeding further, if you are not familiar with Caffe.

In summary, if you have all the per-requisites installed, you can go to the caffe-jacinto folder and execute the following:
* make 
 * Instead, you can also do "make -j50" to speed up the compilaiton
* make pycaffe
 * To compile the python bindings

### Features

New layers have been added to help train for semantic segmentation. Several new options have also been added in solver and layer parameters to support sparsity and quantization. These options will be explained with examples for training.

###### Sparsity
* Measuring sparsity in convolution layers even while training is in progress. 
* Zeroing out of small coefficients during training, similar to caffe-scnn [paper](https://arxiv.org/abs/1608.03665), [code](https://github.com/wenwei202/caffe/tree/scnn)
* Thresholding tool to zero-out some convolution weights in each layer to attain a certain sparsity in each layer.

###### Quantization
* Collecting statistics (range of weights) to enable quantization
* Dynamic -8bit fixed point quantization, improved from Ristretto [paper](https://arxiv.org/abs/1605.06402), [code](https://github.com/pmgysel/caffe)

### Examples
###### Classification:<br>
* [Train sparse, quantized CNN on cifar10 dataset](examples/tidsp/models/sparse/cifar10_classification/README.md): Training scripts and example models that demonstrate training with sparsification and quantization for classification. Note that this is just a toy example and no inference script is provided to test teh final model.

* [Training on LSVRC ImageNet dataset](examples/tidsp/models/sparse/imagenet_classification/README.md).

###### Semantic segmentation:<br>
* New layers, training scripts and examples are provided for semanitc segmentaiton.
* [Train sparse, quantized CNN for segmentation](examples/tidsp/models/sparse/imagenet_segmentation/README.md): Train sparse, quantized CNN on the cityscapes dataset. 
* Note that ImageNet training is must be done before doing this segmentation training to create the pre-trained weights.

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
