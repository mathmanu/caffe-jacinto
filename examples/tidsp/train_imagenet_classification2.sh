#!/bin/bash

#-------------------------------------------------------
folder_name="jacintonet11_cityscapes_2017.06.09"
model_name="jacintonet11"

mkdir $folder_name
max_iter=320000
#-------------------------------------------------------

stage="stage0"
job_dir="$folder_name"/$stage; echo $job_dir; mkdir $job_dir
python ./tools/train/train_imagenet_classification.py --model_name=$model_name --config_name="$job_dir" --solver_param="{'base_lr':0.1,'max_iter':$max_iter, 'test_initialization':0}"



