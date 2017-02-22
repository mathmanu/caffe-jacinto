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

$caffe train --solver="models/sparse/imagenet_classification/jacintonet11(1000)_bn_train_L2.prototxt" --gpu=1,0

pause 'Finished L2 training. Press [Enter] to continue...'


#Save the final model
#cp training/*.txt final/
#cp training/jacintonet11_nobn_iter_.* final/
#pause 'Done.'
