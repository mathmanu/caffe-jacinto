#!/bin/bash
function pause(){
  #read -p "$*"
  echo "$*"
}

#-------------------------------------------------------
rm training/*.caffemodel training/*.prototxt training/*.solverstate training/*.txt
rm final/*.caffemodel final/*.prototxt final/*.solverstate final/*.txt
#-------------------------------------------------------

#-------------------------------------------------------
LOG="training/train-log-`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
#-------------------------------------------------------

#-------------------------------------------------------
caffe=../../../../build/tools/caffe.bin
#-------------------------------------------------------

$caffe train --solver="jacintonet11/jacintonet11_bn_train_L2.prototxt" --gpu=0 
pause 'Finished L2 training. Press [Enter] to continue...'

$caffe train --solver="jacintonet11/jacintonet11_bn_train_L1.prototxt" --gpu=0 --weights="training/train_L2_jacintonet11_bn_iter_32000.caffemodel"
pause 'Finished L1 training. Press [Enter] to continue...'

#Optimize step (merge batch norm coefficients to convolution weights - batch norm coefficients will be set to identity after this in the caffemodel)
$caffe optimize --model="jacintonet11/jacintonet11_bn_deploy.prototxt" --gpu=0 --weights="training/train_L1_jacintonet11_bn_iter_32000.caffemodel" --output="training/optimized_train_L1_jacintonet11_bn_iter_32000.caffemodel"
pause 'Finished optimization. Press [Enter] to continue...'

#Threshold step
$caffe threshold --threshold_fraction_low 0.40 --threshold_fraction_mid 0.80 --threshold_fraction_high 0.80 --threshold_value_max 1.0 --threshold_value_maxratio 1.0 --threshold_step_factor 1e-6 --model="jacintonet11/jacintonet11_bn_deploy.prototxt" --gpu=0 --weights="training/optimized_train_L1_jacintonet11_bn_iter_32000.caffemodel" --output="training/threshold_jacintonet11_bn_iter_32000.caffemodel"
pause 'Finished thresholding. Press [Enter] to continue...'

#Sparse finetuning
$caffe train --solver="jacintonet11/jacintonet11_nobn_train_sparse.prototxt" --gpu=0 --weights="training/threshold_jacintonet11_bn_iter_32000.caffemodel"
pause 'Finished sparse finetuning. Press [Enter] to continue...'

#Quantization step
$caffe train --solver="jacintonet11/jacintonet11_nobn_quant.prototxt" --gpu=0 --weights="training/sparse_jacintonet11_nobn_iter_32000.caffemodel"
cp training/*.txt final/
cp training/sparse_quant_jacintonet11_nobn_iter_32000.* final/
pause 'Finished quantization. Press [Enter] to continue...'


