## Training procedure to generate sparse and quantized model
The training procedure uses several prototxt files. These files are used to perform various stages in the training. An examples is the set of [prototxt files used for training on the Cityscapes dataset](examples/tidsp/models/sparse/cityscapes_segmentation)

This section explains the various stages of the entire training process.

##### Stage 1: Pre-training
Pre-training on a large dataset before doing the training on target dataset is widely used. For example many recognition tasks such as semantic segmentation and object detection uses models that are pre-trained on the ILSVRC ImageNet dataset. Although this is optional, it is highly recommended.

##### Stage 2: L2-Regularized training
It is desirable to first do a round of training with L2-regularization on the target dataset before attempting to induce sparsity by L1-regularization. This is also optional, but again, highly recommended.

##### Stage 3: L1-Regularized training
The remaining stages will use L1-regularization as it promotes sparsity. L1 regularization will induce several small coefficients (weights) and this makes it easy for the thresholding stage later. 

##### Stage 4: Thresholding
Thresholding is performed by specifying "threshold" option to the caffe executable. We can also specify the fraction of sparsity that we need. The threshold step will look at the caffemodel in a layer by layer fashion and try to zero out as many coefficients as specified in the options.

##### Stage 5: Sparse fine tuning
Sparse fine tuning stage takes the thresholded model and tries to recover the quality that was lost during thresholding. 

##### Stage 6: Batch Norm Optimization
Embedded implementations need not actually implement batch norm that comes next to a convolution layer. Batch Norm optimization stage merges the batch norm coefficients into the convolution layers. Subsequent training or testing stages need not use batch norm layers. It is performed by specifying optimize option to the caffe executable.

##### Stage 7: Quantization
The quantization stage collects statistics of layers weights and activations and then quantizes these accordingly. Quantization to 8-bit integer is recommended and is chosen by default when quantization is enabled.


### Additional tools

#### Thresholding
Thresholding is performed by specifying threshold option to the caffe executable. We can also specify the fraction of sparsity that we need. The threshold step will look at the caffemodel provided layer by layer try to zero out as many coefficients as specified in the options.
* Usage:
 * *caffe threshold --threshold_fraction_low value --threshold_fraction_mid value --threshold_fraction_high value --threshold_value_max value --threshold_value_maxratio value --threshold_step_factor value --model="deploy.prototxt" --gpu="gpu string" --weights="trained.caffemodel" --output="output.caffemodel"*
 * threshold_fraction_low: fraction of coefficients to zero out in layers having only few input/output channels
 * threshold_fraction_mid: fraction of coefficients to zero out in layers having a medium number of input/output channels
 * thershold_fraction_high: fraction of coefficients to zero out in layers having a high number of input/output channels
 * threshold_value_max: coefficients having absolute value above this will not be zeroed out
 * threshold_value_maxratio: check the ratio between the coefficient and the max coefficient value in the layer. If absolute value of the ratio is higher than the specified value, it won't be zeroed out.
* Example:
 * *$CAFFE_HOME/build/tools/caffe threshold --threshold_fraction_low 0.40 --threshold_fraction_mid 0.80 --threshold_fraction_high 0.80 --threshold_value_max 0.2 --threshold_value_maxratio 0.2 --threshold_step_factor 1e-6 --model="deploy.prototxt" --gpu="0" --weights="trained.caffemodel" --output="thresholded.caffemodel"*

#### Batch Norm Optimization
Embedded implementations need not actually implement batch norm that comes next to a convolution layer. Batch Norm optimization stage merges the batch norm coefficients into the convolution layers, if there is one nearby. Subsequent training, fine tuning or testing stages using the generated caffemodel need not use batch norm layers in the prototxt. It is performed by specifying optimize option to the caffe executable.
* Usage: 
 * *caffe optimize --model deploy.prototxt --weights=trained.caffemodel --output=output.caffemodel*
* Example
 * *$CAFFE_HOME/build/tools/caffe optimize --model="jacintonet11+seg10(8)_bn_deploy.prototxt"  --gpu=$gpu --weights="jacintonet11+seg10(8)_bn_iter_32000.caffemodel" --output="jacintonet11+seg10_train_bn_optimized_iter_32000.caffemodel"*


## Additional parameters

If a parameter (below) is not specified, the default behavior for Caffe will apply for that specific functionality. Note that the values given against these parameters are for example. 

As usual, [caffe.proto](src/proto/caffe.proto) has the definition of new parameters that can be used to configure the solver or the layers explained below.

### Solver parameters
SolverParameter has the list of all solver parameters, including the new ones added. Below are some of the important new fields in solver parameter. These flags are optional. If they are not specified, the default behavior will apply.  

* display_sparsity: *num_iter*  
Display the fraction of sparsity achieved in each layer. Example:  
display_sparsity: 1000

* threshold_weights: *true or false*  
Whether to threshold the small weights to zero or not. Example:  
threshold_weights: true

* sparsity_threshold: *fraction*  
Value below which the weights will be set to zero. Example:  
sparsity_threshold: 1e-4

* sparsity_target: *fraction*
Once this much sparsity is reached, the learning rate of that layer will be reduced - so that the sparsity remains at that level. Example:  
sparsity_target: 0.80

* sparsity_lr_mult: *fraction*
Once sparsity target is reached for a layer, the lr_mult of that layer will be set to this value (typically a very low value) So that further adjustment will be minimized. Example:  
sparsity_lr_mult: 0.01

* weight_connect_mode: *WEIGHT_DISCONNECTED_ELTWISE*  
The coefficients that are already zeroed out (in the previous step) will be retained as zeros during the fine tuning  stage. Useful to fine tune while retaining sparsity.

* snapshot_log: *true or false*  
Write prototxts containing important information used for quantization. For example, this output prototxt contains quantization bit-widths and shifts for each layer. Example:  
snapshot_log: true

* insert_quantization_param: *true or false*  
Automatically insert quantization related parameters into the layer parameters. Recommended to set to *true* in the quantization stage of training. If this flag is set to *false* in the quantization stage, layer quantization parameters should be present in the input prototxt. Example:  
insert_quantization_param: true

* quantization_start_iter: *iter_number*  
The iteration at which the quantization starts. The iterations before this will be used to collect statistics required for quantization. Example:  
quantization_start_iter: 2000

### Additional Layers
* ImageLabelData Layer
 * Can take list of image files and another list of corresponding label files to produce data and label images for training segmentation.
 * It also supports data augmentation via scale and crop (of the image and label simultaneously).
 * ImageLabelDataParameter is used to configure this layer.

* IOUAccuracy Layer
 * Measures intersection over union (IoU), which is a commonly used measure for computing accuracy for semantic segmentation. 
 * This layer uses AccuracyParameter for accepting the configurations.

