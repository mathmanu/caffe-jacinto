#!/bin/bash

#-------------------------------------------------------
model_name="jacintonet11"
folder_name=training/"$model_name"_imagenet_`date +'%Y-%m-%d_%H-%M-%S'`;mkdir $folder_name

max_iter=320000
base_lr=0.1
#-------------------------------------------------------

stage="stage0"
config_name=$folder_name/$stage;mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','pretrain_model':None}" 
solver_param="{'type':'SGD','base_lr':$base_lr,'max_iter':$max_iter}"
python ./models/image_classification.py --config_param="$config_param" --solver_param="$solver_param"



