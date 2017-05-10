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

#Optimize step (merge batch norm coefficients to convolution weights - batch norm coefficients will be set to identity after this in the caffemodel)
weights="/data/mmcodec_video2_tier3/users/manu/experiments/object/classification/2017.05/imagenet_caffe-0.16/jacintonet11_(loaded-from-caffe-0.15-trained-model)_(old_bn_names)_maxpool(60.52%)_fullval(60.89%)/imagenet_jacintonet11_bn_maxpool_L2_iter_100.caffemodel"

model="models/sparse/imagenet_classification/jacintonet11_maxpool/jacintonet11(1000)_bn_maxpool_deploy_oldBNNames.prototxt"

$caffe optimize --model=$model  --gpu=$gpu --weights=$weights --output="training/imagenet_jacintonet11_bn_maxpool_L2_iter_100_optimized.caffemodel"
