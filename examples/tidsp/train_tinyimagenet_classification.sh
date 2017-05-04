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
caffe=../../build/tools/caffe.bin
#-------------------------------------------------------

#L2 regularized training
$caffe train --solver="models/sparse/tinyimagenet_classification/jacintonet11_bn_maxpool_train_L2.prototxt" --gpu=0
pause 'Finished L2 training. Press [Enter] to continue...'

pause 'Done.'
