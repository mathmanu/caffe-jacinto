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
LOG="train-log-`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
#-------------------------------------------------------

#-------------------------------------------------------
caffe=../../build/tools/caffe.bin
#-------------------------------------------------------

#GLOG_minloglevel=3 
#--v=5

#L2 regularized training

nw_path="/data/mmcodec_video2_tier3/users/manu/experiments/object"
gpu="0" #"1,0" #'0'


#Optimize step (merge batch norm coefficients to convolution weights - batch norm coefficients will be set to identity after this in the caffemodel)
#weights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/examples/imagenet/training_absorb_bn/EXNet_Gr4_NVIDIA_BN_iter_484000_bn_absorbed.caffemodel" 
#$caffe optimize --model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/examples/imagenet/training_absorb_bn/EXNet_Gr4_NVIDIA_BN_iter_484000_bn_absorbed_v5_deployed.prototxt"  --gpu=$gpu --weights=$weights --output="/user/a0875091/files/work/github/weiliu89/caffe-ssd/examples/imagenet/training_absorb_bn/EXNet_Gr4_NVIDIA_BN_iter_484000_bn_absorbed_itr2.caffemodel"

weights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/examples/imagenet/training_EXNET_nvidia_dil_0_Top1_64.56/absorb_bn/EXNet_Gr4_Dil_0_NVIDIA_BN_iter_640000.caffemodel"
$caffe optimize --model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/examples/imagenet/training_EXNET_nvidia_dil_0_Top1_64.56/absorb_bn/EXNet_gr4_dil_0_bvlc_bn_deploy.prototxt"  --gpu=$gpu --weights=$weights --output="/user/a0875091/files/work/github/weiliu89/caffe-ssd/examples/imagenet/training_EXNET_nvidia_dil_0_Top1_64.56/absorb_bn/op/EXNet_Gr4_Dil_0_NVIDIA_BN_iter_640000_bn_absorbed.caffemodel"
