#!/bin/bash

#-------------------------------------------------------
config_name="2017.06.09_imagenet"
model_name="jacintonet11"

base_dir=training/"$config_name"_"$model_name";
job_dir="$base_dir"/stage0; echo $job_dir
mkdir $base_dir
mkdir $job_dir

#LOG="$job_dir/train-log-`date +'%Y-%m-%d_%H-%M-%S'`.txt"
#exec &> >(tee -a "$LOG")
#echo Logging output to "$LOG"
#-------------------------------------------------------


python ./tools/train/train_imagenet_classification.py --model_name=$model_name --config_name=$config_name --stage_name="stage0" 


echo 'Finished L2 training. Press [Enter] to continue...'

