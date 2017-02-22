# ILSVRC ImageNet training

Imagenet training is a pre-requisite before doing several other training tasks. This will create pre-trained weights that can be finetuned for a variety of other tasks.

### Dataset preparation

* First, open a bash prompt and set CAFFE_HOME to the location where Caffe-jacinto is placed. For example:
CAFFE_HOME=~/work/caffe-jacinto

* Change directory.
 * cd $CAFFE_HOME/examples/tidsp

* The following website gives details of how to obtain the ImageNet dataset and organize the data: 
https://github.com/amd/OpenCL-caffe/wiki/Instructions-to-create-ImageNet-2012-data

* The above webpage also explains how to create lmdb database. It can also be created by executing  ./tools/create_imagenet_classification_lmdb.sh. Before executing, open this file and modify the DATA field to point to the location where ImageNet train and val folders are placed.

* After creating the lmdb database, make sure that ilsvrc12_train_lmdb and ilsvrc12_val_lmdb folders in $CAFFE_HOME/examples/tidsp/data point to it. (If they are not there, you can either move them there or create soft links)

### Training 
* Open the file train_imagenet_classification.sh  and look at the gpu variable. If you have more than one NVIDIA CUDA supported GPUs modify this field to reflect it so that teh training will complete faster.

* Execute the script /tools/train_imagenet_classification.sh to do the ImageNEt training. This will take several hours or days, depending on your GPU configuration.

* At the end of the training, the file "jacintonet11_bn_iter_320000.caffemodel" will be created in the training folder. This is the final ImageNet trained model which can be used for classification or for further fine-tuning. 