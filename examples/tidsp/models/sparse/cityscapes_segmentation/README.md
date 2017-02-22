# Sparse, Quantized CNN training for segmentation

In this section, cifar10 dataset is used as an example to explain the training procedure. An inference script is also provided to test the resultant model on your images or video.

First, open a bash prompt and set CAFFE_HOME to the location where Caffe-jacinto is placed. For example:
CAFFE_HOME=~/work/caffe-jacinto

### Dataset preparation
The details about how to obtain the [Cityscapes Dataset](https://www.cityscapes-dataset.com/) can be seen from their website.

Change directory.
* cd $CAFFE_HOME/examples/tidsp

Before training, ceate list files needed to train on cityscapes dataset.
* Open the file /tiolls/create_cityscapes_lists.sh and change the DATASETPATH to the location where you have downloaded the dataset. Under this folder, the gtFine and leftImg8bit folders of Cityscapes should be present.
* Then execute ./tools/create_cityscapes_lists.sh

### Execution
* Open the file train_cityscapes_segmentation.sh  and look at the gpu variable. If you have more than one NVIDIA CUDA supported GPUs modify this field to reflect it so that teh training will complete faster.

* Execute the script: train_cityscapes_segmentation.sh

* This script will perform all the stages required to generate a sparse, quantized CNN model. The following quantized prototxt and model files will be placed in $CAFFE_HOME/examples/tidsp/final:
jacintonet11+seg10_train_L1_nobn_quant_final_iter_4000.prototxt
jacintonet11+seg10_train_L1_nobn_quant_final_iter_4000.caffemodel

### Inference using the trained model
* Copy the file jacintonet11+seg10_train_L1_nobn_quant_final_iter_4000.prototxt into jacintonet11+seg10_train_L1_nobn_quant_final_iter_4000_deploy.prototxt

* Remove everything before the "data_bias" layer and add the following:
name: "ConvNet-11(8)"
input: "data"
input_shape {
  dim: 1
  dim: 3
  dim: 512
  dim: 1024
}

* Remove everthing after the layer "out_deconv_final_up8" and add the following:
layer {
  name: "prob"
  type: "Softmax"
  bottom: "out_deconv_final_up8"
  top: "prob"
}
layer {
  name: "argMaxOut"
  type: "ArgMax"
  bottom: "out_deconv_final_up8"
  top: "argMaxOut"
  argmax_param {
    axis: 1
  }
}

* Open the file infer_cityscapes_segmentation.sh and set the correct paths to model, weights, input and output

* Run the file infer_cityscapes_segmentation.sh
This will create the output images or video in the location corresponding to the output paramter mentioned in the script.

## Additional details (Optional)
In the folder "$CAFFE_HOME/examples/tidsp/models/sparse/cityscapes_segmentation" there are several prototxt files. These files are used to perform various stages of sparse and quantized training.

This section explaines some of the additional options added to caffe solver and layer parameters. It also explains the various stages of the entire training process.

### Stage 1: Pre-training
Pre-training is done using the prototxt jacintonet11_bn_train_L2.prototxt. As the name indicates, this stage uses L2 regularized training.

### Stage 2: L1-Regularized training
Done using the prototxt jacintonet11_bn_train_L1.prototxt. As the name indicates, this stage uses L1 regularized training. L1 regularization will indue several small coefficients (weights). The following options used in the corresponding prototxt are important:  

* display_sparsity: 1000
Display the fraction of sparsity achieved in each layer.

* threshold_weights: true
Whether to threshold the small weights to zero or not.

* sparsity_threshold: 1e-4
Value below which the weights will be set to zero.

* sparsity_target: 0.80
Once this much sparsity is reached, the learning rate of that layer will be reduced - so that the sprasity remains at that level.

* sparsity_lr_mult: 0.01
Once sparsity target is reached for a layer, the lr_mult of that layer will be set to this value (typically a very low value) So that further adjustment will be minimized.

### Stage 3: Thresholding
Thresholding is performed by specifying threshold option to the caffe executable. We can also specify the fraction fo sparsity that we need. The threshold step will look at the caffemodel provided layer by layer try to zero out as many coefficients as specified in the options.

### Stage 4: Sparse finetuning
Sparse finetuning stage takes the thresholded model and tries to recover the quality that was lost during thresholding. 

* weight_connect_mode: WEIGHT_DISCONNECTED_ELTWISE
The coefficients that are already zered out (in the previous step) will be retained as zeros during this finetuning  process.

### Stage 5: Batch Norm Optimization
Embedded implementations need not actually implement batch norm that comes next to a convolution layer. Batch Norm optimization stage merges the batch norm coefficeints into the convolution layers. Subsequent training or testing stages need not use batch norm layers. It is performed by specifying optimize option to the caffe executable.

### Stage 6: Quantization
The quantization stage uses the following additional options.

* snapshot_log: true
Write prototxts containing important information used for quantization. For example, this output prototxt containes quantization bitwidths and shifts for each layer.

* insert_quantization_param: true
Automatically insert quantization related parameters into the layer parameters.

* quantization_start_iter: 2000
The iteration at which the quantization starts. The iterations before this will be used to collect statistics required for quantization.

* weight_connect_mode: WEIGHT_DISCONNECTED_ELTWISE
The coefficients that are already zered out (in the previous step) will be retained as zeros during this finetuning  process.