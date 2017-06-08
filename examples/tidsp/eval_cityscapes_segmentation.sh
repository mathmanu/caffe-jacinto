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
val_resize=0 #"1024 512"
val_input="./data/val-image-list.txt"
val_label="./data/val-label-list.txt"
val_classes=34 #20 #5
val_weights=0
num_images=10 #100000

#for 19 or 20 classes training of cityscapes, first convert to original labelIds and then apply the pallete
label_dict_20_to_34="{0:7, 1:8, 2:11, 3:12, 4:13, 5:17, 6:19, 7:20, 8:21, 9:22, 10:23, 11:24, 12:25, 13:26, 14:27, 15:28, 16:31, 17:32, 18:33, 19:0}"

#class grouping and ignored - 1:19 extra class that we added is also mapped to ignored (1:255)
class_dict="{0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255, 7:0, 8:1, 9:255, 10:255, 11:2, 12:3, 13:4, 14:255, 15:255, 16:255, 17:5, 18:255, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:255, 30:255, 31:16, 32:17, 33:18, 255:255}"

#7 categories
#class_dict="{0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255, 7:1, 8:1, 9:255, 10:255, 11:2, 12:2, 13:2, 14:255, 15:255, 16:255, 17:3, 18:255, 19:3, 20:3, 21:4, 22:4, 23:5, 24:6, 25:6, 26:7, 27:7, 28:7, 29:255, 30:255, 31:7, 32:7, 33:7, 255:255}"


##------------------------------------------------
##L2 training.
#val_weights="training/jsegnet21_maxpool_L2_bn_iter_32000.caffemodel"
#python ./tools/utils/infer_segmentation.py --crop $val_crop --resize $val_resize --model $val_model --weights $val_weights --input $val_input --label $val_label --num_classes=$val_classes --num_images=$num_images --resize_back --label_dict="$label_dict_20_to_34" --class_dict="$class_dict"
#pause 'Finished L2 eval.'

#------------------------------------------------
#L1 training.
val_weights="training/jsegnet21_maxpool_L1_bn_iter_32000.caffemodel"
python ./tools/utils/infer_segmentation.py --crop $val_crop --resize $val_resize --model $val_model --weights $val_weights --input $val_input --label $val_label --num_classes=$val_classes --num_images=$num_images --resize_back --label_dict="$label_dict_20_to_34" --class_dict="$class_dict"
pause 'Finished L1 eval.'

##------------------------------------------------
#val_weights="training/jsegnet21_maxpool_L1_bn_finetune_iter_32000.caffemodel"
#python ./tools/utils/infer_segmentation.py --crop $val_crop --resize $val_resize --model $val_model --weights $val_weights --input $val_input --label $val_label --num_classes=$val_classes --num_images=$num_images --resize_back --label_dict="$label_dict_20_to_34" --class_dict="$class_dict"
#pause 'Finished sparse finetuning eval. Press [Enter] to continue...'

##------------------------------------------------
##Final NoBN Quantization step
#val_model="training/jsegnet21_maxpool_L1_nobn_quant_final_iter_4000_deploy.prototxt"
#val_weights="training/jsegnet21_maxpool_L1_nobn_quant_final_iter_4000.caffemodel"
#python ./tools/utils/infer_segmentation.py --crop $val_crop --resize $val_resize --model $val_model --weights $val_weights --input $val_input --label $val_label --num_classes=$val_classes --num_images=$num_images --resize_back --label_dict="$label_dict_20_to_34" --class_dict="$class_dict"
#pause 'Finished quantization eval. Press [Enter] to continue...'



