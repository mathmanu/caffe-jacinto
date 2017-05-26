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

#L2 regularized training

nw_path="/data/mmcodec_video2_tier3/users/manu/experiments/object"
gpu="1,0" #'0'

val_model="models/sparse/cityscapes_segmentation/jsegnet21_maxpool/jsegnet21_maxpool(8)_bn_deploy.prototxt"
val_crop=0 #"1024 512" #"512 512"
val_resize="1024 512"
val_input="./data/val-image-list.txt"
val_label="./data/val-label-list.txt"
val_classes=34 #19 #5
val_weights=0
num_images=100000
ignore_class_dict="{0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:0, 8:0, 9:1, 10:1, 11:0, 12:0, 13:0, 14:1, 15:1, 16:1, 17:0, 18:1, 19:0, 20:0, 21:0, 22:0, 23:0, 24:0, 25:0, 26:0, 27:0, 28:0, 29:1, 30:1, 31:0, 32:0, 33:0, 255:1}"


##------------------------------------------------
##L2 training.
#val_weights="training/jsegnet21_maxpool_L2_bn_iter_32000.caffemodel"
#python ./tools/infer_segmentation.py --crop $val_crop --resize $val_resize --model $val_model --weights $val_weights --input $val_input --label $val_label --num_classes=$val_classes --num_images=$num_images --resize_back
#pause 'Finished L2 eval.'

#------------------------------------------------
#L1 training.
val_weights="training/jsegnet21_maxpool_L1_bn_iter_32000.caffemodel"
python ./tools/infer_segmentation.py --crop $val_crop --resize $val_resize --model $val_model --weights $val_weights --input $val_input --label $val_label --num_classes=$val_classes --num_images=$num_images --resize_back --ignore_class_dict="$ignore_class_dict"
pause 'Finished L1 eval.'

##------------------------------------------------
#val_weights="training/jsegnet21_maxpool_L1_bn_finetune_iter_32000.caffemodel"
#python ./tools/infer_segmentation.py --crop $val_crop --resize $val_resize --model $val_model --weights $val_weights --input $val_input --label $val_label --num_classes=$val_classes --num_images=$num_images --resize_back
#pause 'Finished sparse finetuning eval. Press [Enter] to continue...'

##------------------------------------------------
##Final NoBN Quantization step
#val_model="training/jsegnet21_maxpool_L1_nobn_quant_final_iter_4000_deploy.prototxt"
#val_weights="training/jsegnet21_maxpool_L1_nobn_quant_final_iter_4000.caffemodel"
#python ./tools/infer_segmentation.py --crop $val_crop --resize $val_resize --model $val_model --weights $val_weights --input $val_input --label $val_label --num_classes=$val_classes --num_images=$num_images --resize_back
#pause 'Finished quantization eval. Press [Enter] to continue...'



