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
caffe=../../../build/tools/caffe.bin
#-------------------------------------------------------

#GLOG_minloglevel=3 
#--v=5

#L2 regularized training
pause 'Starting L2 training.'

$caffe train --solver="models/cityscapes_segmentation/jsegnet21(8)_bn_train_L2.prototxt" --gpu=1 --weights="/user/a0393608/files/work/code/vision/ti/bitbucket/algoref/vision-dl-src/apps/classification/training/2016.12/convnet10x3c512(c3.1c4.1c5.1)(grp1-4)(lr-poly320k)(61.1%)/original/convnet10_iter_320000.caffemodel"

pause 'Finished L2 training.'


