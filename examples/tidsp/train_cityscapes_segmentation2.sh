#!/bin/bash

#-------------------------------------------------------
folder_name="training/jsegnet21_cityscapes_2017.06.09"
model_name="jsegnet21"

mkdir $folder_name
max_iter=32000
threshold_step_factor=1e-6
base_lr=1e-3

solver_param="{'type':'Adam','base_lr':$base_lr,'max_iter':$max_iter,'test_interval':2000,'test_initialization':0}"

sparse_solver_param="{'type':'Adam','base_lr':$base_lr,'max_iter':$max_iter,'test_interval':2000,'test_initialization':0,\
'sparse_mode':1,'display_sparsity':1000}"

quant_solver_param="{'type':'Adam','base_lr':$base_lr,'max_iter':$max_iter,'test_interval':2000,'test_initialization':0,\
'sparse_mode':1,'display_sparsity':1000,'insert_quantization_param':1,'quantization_start_iter':2000,'snapshot_log':1}"

caffe="../../build/tools/caffe.bin"

#------------------------------------------------
#Download the pretrained weights
weights_dst="training/imagenet_jacintonet11_v2_bn_iter_160000.caffemodel"
if [ -f $weights_dst ]; then
  echo "Using pretrained model $weights_dst"
else
  weights_src="https://github.com/tidsp/caffe-jacinto-models/blob/master/examples/tidsp/models/non_sparse/imagenet_classification/jacintonet11_v2/imagenet_jacintonet11_v2_bn_iter_160000.caffemodel?raw=true"
  wget $weights_src -O $weights_dst
fi
#-------------------------------------------------------

#Initial training
weights=$weights_dst
stage="stage0"
job_dir="$folder_name"/$stage; echo $job_dir; mkdir $job_dir
python ./tools/train/train_cityscapes_segmentation.py --model_name=$model_name --config_name="$job_dir" --pretrain_model="$weights" --solver_param=$solver_param
job_dir_prev=$job_dir

#Threshold step
stage="stage1"
weights="$job_dir_prev/jsegnet21_iter_$max_iter.caffemodel"
job_dir="$folder_name"/$stage; echo $job_dir; mkdir $job_dir
$caffe threshold --threshold_fraction_low 0.40 --threshold_fraction_mid 0.70 --threshold_fraction_high 0.70 --threshold_value_max 0.2 --threshold_value_maxratio 0.2 --threshold_step_factor $threshold_step_factor --model="$job_dir_prev/deploy.prototxt" --gpu="0" --weights=$weights --output="$job_dir/jsegnet21_iter_$max_iter.caffemodel"
job_dir_prev=$job_dir

#fine tuning
stage="stage2"
weights="$job_dir_prev/jsegnet21_iter_$max_iter.caffemodel"
job_dir="$folder_name"/$stage; echo $job_dir; mkdir $job_dir
python ./tools/train/train_cityscapes_segmentation.py --model_name=$model_name --config_name="$job_dir" --pretrain_model="$weights" --solver_param=$sparse_solver_param
job_dir_prev=$job_dir

#Threshold step
stage="stage3"
weights="$job_dir_prev/jsegnet21_iter_$max_iter.caffemodel"
job_dir="$folder_name"/$stage; echo $job_dir; mkdir $job_dir
$caffe threshold --threshold_fraction_low 0.40 --threshold_fraction_mid 0.80 --threshold_fraction_high 0.80 --threshold_value_max 0.2 --threshold_value_maxratio 0.2 --threshold_step_factor $threshold_step_factor --model="$job_dir_prev/deploy.prototxt" --gpu="0" --weights=$weights --output="$job_dir/jsegnet21_iter_$max_iter.caffemodel"
job_dir_prev=$job_dir

#fine tuning
stage="stage4"
weights="$job_dir_prev/jsegnet21_iter_$max_iter.caffemodel"
job_dir="$folder_name"/$stage; echo $job_dir; mkdir $job_dir
python ./tools/train/train_cityscapes_segmentation.py --model_name=$model_name --config_name="$job_dir" --pretrain_model="$weights" --solver_param=$sparse_solver_param
job_dir_prev=$job_dir

#Threshold step
stage="stage5"
weights="$job_dir_prev/jsegnet21_iter_$max_iter.caffemodel"
job_dir="$folder_name"/$stage; echo $job_dir; mkdir $job_dir
$caffe threshold --threshold_fraction_low 0.40 --threshold_fraction_mid 0.90 --threshold_fraction_high 0.90 --threshold_value_max 0.2 --threshold_value_maxratio 0.2 --threshold_step_factor $threshold_step_factor --model="$job_dir_prev/deploy.prototxt" --gpu="0" --weights=$weights --output="$job_dir/jsegnet21_iter_$max_iter.caffemodel"
job_dir_prev=$job_dir

#fine tuning
stage="stage6"
weights="$job_dir_prev/jsegnet21_iter_$max_iter.caffemodel"
job_dir="$folder_name"/$stage; echo $job_dir; mkdir $job_dir
python ./tools/train/train_cityscapes_segmentation.py --model_name=$model_name --config_name="$job_dir" --pretrain_model="$weights" --solver_param=$sparse_solver_param
job_dir_prev=$job_dir

#quantization
stage="stage7"
weights="$job_dir_prev/jsegnet21_iter_$max_iter.caffemodel"
job_dir="$folder_name"/$stage; echo $job_dir; mkdir $job_dir
python ./tools/train/train_cityscapes_segmentation.py --model_name=$model_name --config_name="$job_dir" --pretrain_model="$weights" --solver_param=$quant_solver_param
job_dir_prev=$job_dir
