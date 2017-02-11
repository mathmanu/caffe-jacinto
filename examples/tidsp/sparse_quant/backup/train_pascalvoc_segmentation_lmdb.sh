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
$caffe train --solver="models/pascalvoc_segmentation/jsegnet21_bn_train_L2_lmdb_augument.prototxt" --gpu=1
pause 'Finished L2 training. Press [Enter] to continue...'


