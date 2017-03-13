#!/bin/bash
function pause(){
  #read -p "$*"
  echo "$*"
}

#-------------------------------------------------------
#rm training/*.caffemodel training/*.prototxt training/*.solverstate training/*.txt
#rm final/*.caffemodel final/*.prototxt final/*.solverstate final/*.txt
#-------------------------------------------------------

#-------------------------------------------------------
LOG="training/train-log-`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
#-------------------------------------------------------

#-------------------------------------------------------
caffe=../../build/tools/caffe.bin
gpu="0" #"1,0" #"0"
#-------------------------------------------------------

#L2 regularized training
$caffe train --solver="models/sparse/cifar10_classification/jacintonet11_bn_train_L2.prototxt" --gpu=$gpu
pause 'Finished L2 training. Press [Enter] to continue...'

#L1 regularized finetuning - induce sparsity
$caffe train --solver="models/sparse/cifar10_classification/jacintonet11_bn_train_L1.prototxt" --gpu=$gpu --weights="training/train_L2_jacintonet11_bn_iter_32000.caffemodel"
pause 'Finished L1 training. Press [Enter] to continue...'

#Threshold step - force a fixed fraction of sparsity - OPTIONAL
$caffe threshold --threshold_fraction_low 0.40 --threshold_fraction_mid 0.80 --threshold_fraction_high 0.80 --threshold_value_max 0.2 --threshold_value_maxratio 0.2 --threshold_step_factor 1e-6 --model="models/sparse/cifar10_classification/jacintonet11_bn_deploy.prototxt" --gpu=$gpu --weights="training/train_L1_jacintonet11_bn_iter_32000.caffemodel" --output="training/threshold_jacintonet11_bn_iter_32000.caffemodel"
pause 'Finished thresholding. Press [Enter] to continue...'

#Sparse finetuning
$caffe train --solver="models/sparse/cifar10_classification/jacintonet11_bn_train_sparse.prototxt" --gpu=$gpu --weights="training/threshold_jacintonet11_bn_iter_32000.caffemodel"
pause 'Finished sparse finetuning. Press [Enter] to continue...'

#Optimize step (merge batch norm coefficients to convolution weights - batch norm coefficients will be set to identity after this in the caffemodel)
$caffe optimize --model="models/sparse/cifar10_classification/jacintonet11_bn_deploy.prototxt" --gpu=$gpu --weights="training/sparse_jacintonet11_bn_iter_32000.caffemodel" --output="training/optimized_sparse_quant_jacintonet11_bn_iter_32000.caffemodel"
pause 'Finished optimization. Press [Enter] to continue...'

#Final No BN Quantization step
$caffe train --solver="models/sparse/cifar10_classification/jacintonet11_nobn_train_quant_final.prototxt" --gpu=$gpu --weights="training/optimized_sparse_quant_jacintonet11_bn_iter_32000.caffemodel"
pause 'Finished quantization. Press [Enter] to continue...'

#Test the final model
$caffe train --solver="models/sparse/cifar10_classification/jacintonet11_nobn_test_quant.prototxt" --gpu=$gpu --weights="training/final_sparse_quant_jacintonet11_nobn_iter_4000.caffemodel"
pause 'Finished test. Press [Enter] to continue...'

#Save the final model
cp training/*.txt final/
cp training/final_sparse_quant_jacintonet11_nobn_iter_4000.* final/
pause 'Done.'
