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

#L2 regularized training

$caffe train --solver="models/sparse/imagenet_classification/jacintonet11_maxpool/jacintonet11(1000)_bn_maxpool_train_L2.prototxt" --gpu=0,1 --weights="/data/mmcodec_video2_tier3/users/manu/experiments/object/classification/2017.05/imagenet_caffe-0.16/jacintonet11_maxpool2x2_scale_bias(false)_acc(56.92%)/imagenet_jacintonet11_bn_maxpool_L2_iter_320000.caffemodel"

pause 'Finished L2 training. Press [Enter] to continue...'


#Save the final model
#cp training/*.txt final/
#cp training/jacintonet11_nobn_iter_.* final/
#pause 'Done.'
