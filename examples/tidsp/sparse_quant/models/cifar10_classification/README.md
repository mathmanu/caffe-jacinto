# Sparse, Quantized CNN training for classification

In this section, cifar10 dataset is used as an example to explain the training porcedure.

First, open a bash prompt and set CAFFE_HOME to the location where Caffe-jacinto is placed. For example:
CAFFE_HOME=~/work/caffe-jacinto

### Dataset preparation
Change directory to your cafee-jacinto folder.
- cd $CAFFE_HOME

Fetch the cifar10 dataset by executing:
- $CAFFE_HOME/data/cifar10/get_cifar10.sh

Create LMDB folders for the cifar10 dataset by executing:
- $CAFFE_HOME/examples/cifar10/create_cifar10.sh

### Execution
In the folder "$CAFFE_HOME/examples/tidsp/sparse_quant/classification" there aer several prototxt files. These files are used to performa various stages of sparse adn quantized training. Most of these prototxt files uses the lmdb folders that we created using the create_cifar10.sh script.

Open these prototxt fiels and change teh train and test lmdb paths to the ones that we cerated.

In bash change the folder:
- cd $CAFFE_HOME/examples/tidsp/sparse_quant/classification

Execute the script:
- train_cifar10_jacintonet11.sh

This script will perform all the stages required to generate a sparse, quantized CNN model. The quantized model will be placed in $CAFFE_HOME/examples/tidsp/sparse_quant/classification/final.

## Additional details (Optional)

This section explaines some of the additional options added to caffe solver and layer parameters. It also explains the various stages of the entire training process.

### Stage 1: Pre-training
Pre-training is done using the prototxt cifar10/jacintonet11_bn_train_L2.prototxt. As the name indicates, this stage uses L2 regularized training.

### Stage 2: L1-Regularized training
Done using the prototxt cifar10/jacintonet11_bn_train_L1.prototxt. As the name indicates, this stage uses L1 regularized training. L1 regularization will indue several small coefficients (weights). The following options used in the corresponding prototxt are important:  

- display_sparsity: 1000
Display the fraction of sparsity achieved in each layer.

- threshold_weights: true
Whether to threshold the small weights to zero or not.

- sparsity_threshold: 1e-4
Value below which the weights will be set to zero.

- sparsity_target: 0.80
Once this much sparsity is reached, the learning rate of that layer will be reduced - so that the sprasity remains at that level.

- sparsity_lr_mult: 0.01
Once sparsity target is reached for a layer, the lr_mult of that layer will be set to this value (typically a very low value) So that further adjustment will be minimized.

### Stage 3: Thresholding
Thresholding is performed by specifying threshold option to the caffe executable. We can also specify the fraction fo sparsity that we need. The threshold step will look at the caffemodel provided layer by layer try to zero out as many coefficients as specified in the options.

### Stage 4: Sparse finetuning
Sparse finetuning stage takes the thresholded model and tries to recover the quality that was lost during thresholding. 

- weight_connect_mode: WEIGHT_DISCONNECTED_ELTWISE
The coefficients that are already zered out (in the previous step) will be retained as zeros during this finetuning  process.

### Stage 5: Quantization
The quantization stage uses the following additional options.

- snapshot_log: true
Write prototxts containing imporatn information used for quantization. For example, this output prototxt containes quantization bitwidths and shifts for each layer.

- insert_quantization_param: true
Automatically insert quantization related parameters into the layer parameters.

- quantization_start_iter: 4000
The iteration at which the quantization starts. The iterations before this will be used to collect statistics required for quantization.

### Stage 6: Batch Norm Optimization
Embedded implementations need not actually implement batch norm that comes next to a convolution layer. Batch Norm optimization stage merges the batch norm coefficeints into the convolution layers. Subsequent trainign or testing stages need not use batch norm layers. It is performed by specifying optimize option to the caffe executable.

### Stage 7: Final finetuning without batch norm
This final stage will do a re-quantization as the batch norm optimization stage will change in the ranges of convolution.