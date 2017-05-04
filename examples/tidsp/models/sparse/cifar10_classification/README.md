# Sparse, Quantized CNN training for classification

In this section, cifar10 dataset is used as an example to explain the training procedure. This is  a toy example to do the flow flushing of the entire training procedure. No inference script is provided to test the resulting model.

First, open a bash prompt and set CAFFE_HOME to the location where Caffe-jacinto is placed. For example:
CAFFE_HOME=~/work/caffe-jacinto

### Dataset preparation
Change directory to your caffe-jacinto folder.
* cd $CAFFE_HOME/examples/tidsp

Fetch the cifar10 dataset by executing:
* ./tools/get_cifar10.sh

Create LMDB folders for the cifar10 dataset by executing:
* ./tools/create_cifar10.sh

### Execution

Execute the script:
* train_cifar10_classification.sh

This script will perform all the stages required to generate a sparse, quantized CNN model. The quantized model will be placed in $CAFFE_HOME/examples/tidsp/final.


