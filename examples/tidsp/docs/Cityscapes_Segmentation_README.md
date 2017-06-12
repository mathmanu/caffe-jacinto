# Sparse, Quantized CNN training for segmentation

### Pre-requisites
It is assumed here, that all the pre-requisites required for running Caffe-jacinto are met.

Open a bash terminal and set CAFFE_HOME to the location where Caffe-jacinto is placed. For example:
CAFFE_HOME=~/work/caffe-jacinto

### Dataset preparation
The details about how to obtain the [Cityscapes Dataset](https://www.cityscapes-dataset.com/) can be seen from their website. Download and unzip gtFine and leftImg8bit as sub-directories into a suitable folder.

Change directory to the tidsp folder. All the remaining scripts are to be executed from this folder.
* cd $CAFFE_HOME/examples/tidsp

Before training, create list files needed to train on cityscapes dataset.
* Open the file ./tools/create_cityscapes_lists.sh (eg. vi ./tools/create_cityscapes_lists.sh) and change the DATASETPATH to the location where you have downloaded the dataset. Under this folder, the gtFine and leftImg8bit folders of Cityscapes should be present.
* Then execute ./tools/create_cityscapes_lists.sh. This script creates the image and label lists used for training. It also does label transformation. 
* We have chosen a smaller set of 5-classes for training. 32 classes of cityscapes are converted into 5-classes - so the trained model will learn to segment 5-classes (background, road, person, road signs, vehicle). 
* Note: The number of classes and class mappings can be easily changed in this script. The network model prototxt may also need to be changed if the number of classes chosen is more than the output channels in the model.
* Note: this 5-class training is different from the typical [19-class training done for cityscapes](https://github.com/mcordts/cityscapesScripts) and reported on the benchmark website. 


### Execution
* Open the file train_cityscapes_segmentation.sh  and look at the gpu variable. If you have more than one NVIDIA CUDA supported GPUs, modify this field to reflect it so that the training will complete faster.

* Execute the training by running the training script: ./train_cityscapes_segmentation.sh. 

* The training will perform all the stages required to generate a sparse, quantized CNN model. 

* The training takes around 22 hours, when using one NVIDIA GTX 1080 GPU.

* After the training, The following quantized prototxt and model files will be placed in $CAFFE_HOME/examples/tidsp/final:
jacintonet11+seg10_train_L1_nobn_quant_final_iter_4000.prototxt
jacintonet11+seg10_train_L1_nobn_quant_final_iter_4000.caffemodel

### Results

The validation accuracy is printed in the training log. Following is what we got for the 5-class (background, road, person, road signs, vehicle) training.


|Configuration                                    |Pixel Accuracy  |Mean IOU  |
|-------------------------------------------------|----------------|----------|
|Initial L2 regularized training                  |96.20           |83.23     |
|L1 regularized fine tuning                       |96.32           |<b>83.94  |
|Sparse fine tuned(nearly 80% zero coefficients)  |96.11           |82.85     |
|Sparse(80%), Quantized(8-bit dynamic fixed point)|95.91           |<b>82.15  |
|<b>Overall impact due to sparse+quant            |<b>-0.42        |<b>-1.79  |

* 80% sparsity (i.e. zero coefficients in convolution weights) implies that the complexity of inference can be potentially reduced by 5x - by using a suitable sparse convolution implementation.

* It is possible to change the value of sparsity applied - see the training script for more details.

### Inference using the trained model
* This section explains how the trained model can be used for inference on a PC using Caffe-jacinto.

* Copy the file jacintonet11+seg10_train_L1_nobn_quant_final_iter_4000.prototxt into jacintonet11+seg10_train_L1_nobn_quant_final_iter_4000_deploy.prototxt (we will call this as the "deploy  prototxt").  

* We will also call jacintonet11+seg10_train_L1_nobn_quant_final_iter_4000.caffemodel as the "deploy caffemodel".

* Remove everything before the "data_bias" layer and add the following, in the deploy  prototxt:  
name: "ConvNet-11(8)"  
input: "data"  
input_shape {  
  dim: 1  
  dim: 3  
  dim: 512  
  dim: 1024  
}  

* Remove everything after the layer "out_deconv_final_up8" and add the following, in the deploy  prototxt:  
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

* Open the file infer_cityscapes_segmentation.sh and set the correct paths to model (should point to deploy  prototxt), weights (should point to the deploy caffemodel).

* In the same file, correct the paths of input (an mp4 video file or a folder containing images) and output (an mp4 name or a folder name)

* Run the file infer_cityscapes_segmentation.sh
This will create the output images or video in the location corresponding to the output parameter mentioned in the script.
