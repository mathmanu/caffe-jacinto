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

#$caffe train --solver="models/sparse/imagenet_classification/jacintonet11(1000)_bn_train_L2.prototxt" --gpu=1,0
$caffe train --solver="models/sparse/imagenet_classification/jacintonet11(1000)_bn_maxpool_train_L2.prototxt" --gpu=0 --weights="/data/mmcodec_video2_tier3/users/manu/experiments/object/classification/2017.02/imagenet_jacintonet11(60.77%)/jacintonet11_bn_iter_320000.caffemodel"

pause 'Finished L2 training. Press [Enter] to continue...'


#Save the final model
#cp training/*.txt final/
#cp training/jacintonet11_nobn_iter_.* final/
#pause 'Done.'
