#!/bin/bash

#-------------------------------------------------------
folder_name="training/jsegnet21_cityscapes20_2017.06.12";mkdir $folder_name
model_name="jsegnet21"

#------------------------------------------------
max_iter=32000
stepvalue=24000
threshold_step_factor=1e-6
base_lr=1e-4

#------------------------------------------------
solver_param="{'type':'Adam','base_lr':$base_lr,'max_iter':$max_iter}"

sparse_solver_param="{'type':'Adam','base_lr':$base_lr,'max_iter':$max_iter\
'lr_policy':'multistep','stepvalue':[$stepvalue],'sparse_mode':1,'display_sparsity':1000}"

quant_solver_param="{'type':'Adam','base_lr':$base_lr,'max_iter':$max_iter,\
'sparse_mode':1,'display_sparsity':1000,'insert_quantization_param':1,'quantization_start_iter':2000,'snapshot_log':1}"

#------------------------------------------------
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
stage="stage0"
weights=$weights_dst
config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','pretrain_model':'$weights'}" 
python ./models/image_segmentation.py --config_param="$config_param" --solver_param=$solver_param
config_name_prev=$config_name

#Threshold step
stage="stage1"
weights="$config_name_prev/jsegnet21_iter_$max_iter.caffemodel"
config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
$caffe threshold --threshold_fraction_low 0.40 --threshold_fraction_mid 0.70 --threshold_fraction_high 0.70 --threshold_value_max 0.2 --threshold_value_maxratio 0.2 --threshold_step_factor $threshold_step_factor --model="$config_name_prev/deploy.prototxt" --gpu="0" --weights=$weights --output="$config_name/jsegnet21_iter_$max_iter.caffemodel"
config_name_prev=$config_name

#fine tuning
stage="stage2"
weights="$config_name_prev/jsegnet21_iter_$max_iter.caffemodel"
config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','pretrain_model':'$weights'}" 
python ./models/image_segmentation.py --config_param="$config_param" --solver_param=$sparse_solver_param
config_name_prev=$config_name

#Threshold step
stage="stage3"
weights="$config_name_prev/jsegnet21_iter_$max_iter.caffemodel"
config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
$caffe threshold --threshold_fraction_low 0.40 --threshold_fraction_mid 0.80 --threshold_fraction_high 0.80 --threshold_value_max 0.2 --threshold_value_maxratio 0.2 --threshold_step_factor $threshold_step_factor --model="$config_name_prev/deploy.prototxt" --gpu="0" --weights=$weights --output="$config_name/jsegnet21_iter_$max_iter.caffemodel"
config_name_prev=$config_name

#fine tuning
stage="stage4"
weights="$config_name_prev/jsegnet21_iter_$max_iter.caffemodel"
config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','pretrain_model':'$weights'}" 
python ./models/image_segmentation.py --config_param="$config_param" --solver_param=$sparse_solver_param
config_name_prev=$config_name

#Threshold step
stage="stage5"
weights="$config_name_prev/jsegnet21_iter_$max_iter.caffemodel"
config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
$caffe threshold --threshold_fraction_low 0.40 --threshold_fraction_mid 0.90 --threshold_fraction_high 0.90 --threshold_value_max 0.2 --threshold_value_maxratio 0.2 --threshold_step_factor $threshold_step_factor --model="$config_name_prev/deploy.prototxt" --gpu="0" --weights=$weights --output="$config_name/jsegnet21_iter_$max_iter.caffemodel"
config_name_prev=$config_name

#fine tuning
stage="stage6"
weights="$config_name_prev/jsegnet21_iter_$max_iter.caffemodel"
config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','pretrain_model':'$weights'}" 
python ./models/image_segmentation.py --config_param="$config_param" --solver_param=$sparse_solver_param
config_name_prev=$config_name

#quantization
stage="stage7"
weights="$config_name_prev/jsegnet21_iter_$max_iter.caffemodel"
config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','pretrain_model':'$weights'}" 
python ./models/image_segmentation.py --config_param="$config_param" --solver_param=$quant_solver_param
config_name_prev=$config_name
