# Caffe-jacinto
Caffe-jacinto is a fork of [NVIDIA/caffe](https://github.com/NVIDIA/caffe) that enables training of sparse, quantized CNN models.

### Features

Sparsity
- Measuring sparsity in convolution layers even while training is in progress. 
- Zeroing out of small coefficients during training, similar to caffe-scnn [paper](https://arxiv.org/abs/1608.03665), [code](https://github.com/wenwei202/caffe/tree/scnn)
- Thresholding tool to zero-out some convolution weights in each layer to attain a certain sparsity in each layer.

Quantization
- Collecting statistics (range of weights) to enable quantization
- Dynamic -8bit fixed point quantization, improved from Ristretto [paper](https://arxiv.org/abs/1605.06402), [code](https://github.com/pmgysel/caffe)

Classification
- Training scripts and example models that demonstrate sparsification and quantization for classification on cifar10 dataset

Semantic segmentation
- Training scripts and example models that demonstrate sparsification and quantization for semantic segmentation on [Cityscapes dataset](https://www.cityscapes-dataset.com/)
- Additional layers to help semantic segmentation training: [ImageLabelData Layer](https://github.com/fyu/caffe-dilation), IOUAccuracy Layer 

### Installation and usage
The procedure for installation and usage of Caffe-jacinto is quite similar to Caffe. Please see the Caffe section below to understand the details.

Several new optionas are added in solver and layer parameters to support sparsity and quantization. These options will be explained with examples for training with sparsity and quantization.

### Training procedure for sparsity and quantization

[Train sparse, quantized CNN for classification or segmentation](examples/tidsp/sparse_quant/README.md)


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
