#!/bin/bash

#-------------------------------------------------------
folder_name="training/jacintonet11_imagenet_2017.06.12";mkdir $folder_name
model_name="jacintonet11"
max_iter=320000
base_lr=0.1
#-------------------------------------------------------

stage="stage0"
config_name=$folder_name/$stage;mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','pretrain_model':None}" 
solver_param="{'type':'SGD','base_lr':$base_lr,'max_iter':$max_iter}"
python ./models/image_classification.py --config_param="$config_param" --solver_param="$solver_param"



