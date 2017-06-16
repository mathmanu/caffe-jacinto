#!/bin/bash

#-------------------------------------------------------
DATE_TIME=`date +'%Y-%m-%d_%H-%M-%S'`
#-------------------------------------------------------

#-------------------------------------------------------
model_name=jacintonet11
dataset=imagenet
folder_name=training/"$model_name"_"$dataset"_"$DATE_TIME";mkdir $folder_name

#------------------------------------------------
LOG=$folder_name/train-log_"$DATE_TIME".txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#------------------------------------------------
caffe="../../build/tools/caffe.bin"

#-------------------------------------------------------
max_iter=320000
base_lr=0.1

#-------------------------------------------------------

stage="stage0"
config_name=$folder_name/$stage;mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','pretrain_model':None}" 
solver_param="{'type':'SGD','base_lr':$base_lr,'max_iter':$max_iter}"
python ./models/image_classification.py --config_param="$config_param" --solver_param="$solver_param"



