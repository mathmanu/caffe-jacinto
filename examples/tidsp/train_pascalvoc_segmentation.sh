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
gpu="0"
#-------------------------------------------------------

#GLOG_minloglevel=3 
#--v=5

#L2 regularized training
pause 'Starting L2 training.'
$caffe train --solver="models/sparse/pascalvoc_segmentation/jacintonet11+seg10(32)_bn_train_L2.prototxt" --gpu=$gpu
pause 'Finished L2 training.'


