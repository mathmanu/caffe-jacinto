#!/bin/bash

#-------------------------------------------------------
folder_name="training/jacintonet11_cifar10_2017.06.12";mkdir $folder_name
model_name="jacintonet11"
max_iter=32000
base_lr=0.1
#-------------------------------------------------------

stage="stage0"
config_name=$folder_name/$stage;mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','pretrain_model':None,\
'num_output':10,'image_width':32,'image_height':32,'crop_size':32,'total_stride':2,\
'train_data':'./data/cifar10_train_lmdb','test_data':'./data/cifar10_test_lmdb',\
'test_interval':1000,'num_test_image':10000}" 
solver_param="{'type':'SGD','base_lr':$base_lr,'max_iter':$max_iter}"
python ./models/image_classification.py --config_param="$config_param" --solver_param="$solver_param"



