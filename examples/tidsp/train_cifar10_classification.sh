#!/bin/bash

#-------------------------------------------------------
DATE_TIME=`date +'%Y-%m-%d_%H-%M-%S'`
#-------------------------------------------------------

#-------------------------------------------------------
model_name=jacintonet11
dataset=cifar10
folder_name=training/"$model_name"_"$dataset"_"$DATE_TIME";mkdir $folder_name

#------------------------------------------------
LOG=$folder_name/train-log_"$DATE_TIME".txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#------------------------------------------------
caffe="../../build/tools/caffe.bin"

#-------------------------------------------------------
max_iter=64000
base_lr=0.1
threshold_step_factor=1e-6
type=SGD
batch_size=64
stride_list="[1,1,2,1,2]"
#-------------------------------------------------------
solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'test_interval':1000}"

weights=/user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/examples/tidsp/training/jacintonet11_cifar10_2017-06-19_13-30-37_91.89%/stage0/jacintonet11_cifar10_iter_64000.caffemodel

#-------------------------------------------------------
stage="stage0"
config_name=$folder_name/$stage;mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','stride_list':$stride_list,'pretrain_model':'$weights',\
'num_output':10,'image_width':32,'image_height':32,'crop_size':32,\
'accum_batch_size':$batch_size,'batch_size':$batch_size,\
'train_data':'./data/cifar10_train_lmdb','test_data':'./data/cifar10_test_lmdb',\
'num_test_image':10000,'test_batch_size':50}" 
python ./models/image_classification.py --config_param="$config_param" --solver_param="$solver_param"
config_name_prev=$config_name


#Threshold step
stage="stage1"
weights=$config_name_prev/"$model_name"_"$dataset"_iter_$max_iter.caffemodel
config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','stride_list':$stride_list:'pretrain_model':'$weights',\
'num_output':10,'image_width':32,'image_height':32,'crop_size':32,\
'accum_batch_size':$batch_size,'batch_size':$batch_size,\
'train_data':'./data/cifar10_train_lmdb','test_data':'./data/cifar10_test_lmdb',\
'num_test_image':10000,'test_batch_size':50}" 
$caffe threshold --threshold_fraction_low 0.40 --threshold_fraction_mid 0.80 --threshold_fraction_high 0.80 --threshold_value_max 0.2 --threshold_value_maxratio 0.2 --threshold_step_factor $threshold_step_factor --model="$config_name_prev/deploy.prototxt" --gpu="0" --weights=$weights --output=$config_name/"$model_name"_"$dataset"_iter_$max_iter.caffemodel
config_name_prev=$config_name


#-------------------------------------------------------
#fine tuning
stage="stage2"
weights=$config_name_prev/"$model_name"_"$dataset"_iter_$max_iter.caffemodel

max_iter=64000
stepvalue1=32000
stepvalue2=48000
type=SGD
base_lr=0.01

sparse_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'test_interval':1000,\
'lr_policy':'multistep','stepvalue':[$stepvalue1,$stepvalue2],\
'sparse_mode':1,'display_sparsity':1000}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','stride_list':$stride_list,'pretrain_model':'$weights',\
'num_output':10,'image_width':32,'image_height':32,'crop_size':32,\
'accum_batch_size':$batch_size,'batch_size':$batch_size,\
'train_data':'./data/cifar10_train_lmdb','test_data':'./data/cifar10_test_lmdb',\
'num_test_image':10000,'test_batch_size':50}" 

python ./models/image_classification.py --config_param="$config_param" --solver_param=$sparse_solver_param
config_name_prev=$config_name


#-------------------------------------------------------
#quantization
stage="stage3"
base_lr=0.001
quant_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'test_interval':1000,\
'lr_policy':'multistep','stepvalue':[$stepvalue1,$stepvalue2],\
'sparse_mode':1,'display_sparsity':1000,\
'insert_quantization_param':1,'quantization_start_iter':2000,'snapshot_log':1}"

weights=$config_name_prev/"$model_name"_"$dataset"_iter_$max_iter.caffemodel
config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','stride_list':$stride_list,'pretrain_model':'$weights',\
'num_output':10,'image_width':32,'image_height':32,'crop_size':32,\
'accum_batch_size':$batch_size,'batch_size':$batch_size,\
'train_data':'./data/cifar10_train_lmdb','test_data':'./data/cifar10_test_lmdb',\
'num_test_image':10000,'test_batch_size':50}" 
python ./models/image_classification.py --config_param="$config_param" --solver_param=$quant_solver_param
config_name_prev=$config_name


