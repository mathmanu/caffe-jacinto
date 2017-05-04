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
#-------------------------------------------------------

#GLOG_minloglevel=3 
#--v=5

nw_path="/data/mmcodec_video2_tier3/users/manu/experiments/object"
gpu="1,0" #'0'

#L2 training.
weights="$nw_path/segmentation/2017.04/2017.04.15.jsegnet21.maxpool.(sparse_bugfix).rc13/cityscapes/training/jacintonet11+seg10_train_L2_bn_iter_32000.caffemodel"
##weights="training/jacintonet11_bn_iter_320000.caffemodel"
$caffe train --solver="models/sparse/cityscapes_segmentation/jacintonet11+seg10_maxpool/jacintonet11+seg10(8)_bn_train_L2.prototxt" --gpu=$gpu --weights=$weights
pause 'Finished L2 training.'

##L1 training.
weights="training/jacintonet11+seg10_train_L2_bn_iter_32000.caffemodel"
$caffe train --solver="models/sparse/cityscapes_segmentation/jacintonet11+seg10_maxpool/jacintonet11+seg10(8)_bn_train_L1.prototxt" --gpu=$gpu --weights=$weights
pause 'Finished L1 training.'

#Threshold step - force a fixed fraction of sparsity - OPTIONAL
weights="training/jacintonet11+seg10_train_L1_bn_iter_32000.caffemodel"
$caffe threshold --threshold_fraction_low 0.40 --threshold_fraction_mid 0.80 --threshold_fraction_high 0.80 --threshold_value_max 0.2 --threshold_value_maxratio 0.2 --threshold_step_factor 1e-6 --model="models/sparse/cityscapes_segmentation/jacintonet11+seg10_maxpool/jacintonet11+seg10(8)_bn_deploy.prototxt" --gpu=$gpu --weights=$weights --output="training/jacintonet11+seg10_train_L1_bn_sparse_iter_32000.caffemodel"
pause 'Finished thresholding. Press [Enter] to continue...'

#Sparse finetuning
weights="training/jacintonet11+seg10_train_L1_bn_sparse_iter_32000.caffemodel"
$caffe train --solver="models/sparse/cityscapes_segmentation/jacintonet11+seg10_maxpool/jacintonet11+seg10(8)_bn_train_L1_finetune.prototxt"  --gpu=$gpu --weights=$weights
pause 'Finished sparse finetuning. Press [Enter] to continue...'

#Optimize step (merge batch norm coefficients to convolution weights - batch norm coefficients will be set to identity after this in the caffemodel)
weights="training/jacintonet11+seg10_train_L1_bn_finetune_iter_32000.caffemodel"
$caffe optimize --model="models/sparse/cityscapes_segmentation/jacintonet11+seg10_maxpool/jacintonet11+seg10(8)_bn_deploy.prototxt"  --gpu=$gpu --weights=$weights --output="training/jacintonet11+seg10_train_L1_bn_quant_optimized_iter_32000.caffemodel"
pause 'Finished optimization. Press [Enter] to continue...'

#Final NoBN Quantization step
weights="training/jacintonet11+seg10_train_L1_bn_quant_optimized_iter_32000.caffemodel"
$caffe train --solver="models/sparse/cityscapes_segmentation/jacintonet11+seg10_maxpool/jacintonet11+seg10(8)_nobn_train_L1_quant_final.prototxt"  --gpu=$gpu --weights=$weights
pause 'Finished final NoBN step. Press [Enter] to continue...'

#Save the final model
cp training/*.txt final/
cp training/jacintonet11+seg10_train_L1_nobn_quant_final_iter_4000.* final/
pause 'Done.'


