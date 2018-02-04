from __future__ import print_function
import sys
sys.path.insert(0, '/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/python')
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess
from enum import IntEnum
###################################################################################
#params
#VOC0712, cityscape, cityscapes_wo_ignored, CITY_512, CITY_720x368,
#CITY_720x368_TI_CATEGORY, CITY_1024x512
#TI2016, TI2017, TI2017_Tiny,TI2017_Tiny_MulTiles_minSize12x32, 
#TI2017_Tiny_MulTiles_minSize24x56,TI2017_SingleImg,TI2017_Tiny_MulTiles_mulCat, TI2017_V105, TI2017_V7_106_MulTiles
#TI_201708, TI_201708_720x368
#KITTI, KITTI_Per_Car_Cyclist, KITTI_Per_Car_Cyclist_P1P2,TI_mulTS_1024X512_V007_008_106
dataset = "CITY_720x368_TI_CATEGORY"
wideAR = False
#JACINTO, JACINTO_V2, VGG, VGG_G4, RES101,RESTEN, "JINET", "ENET", "mobile"
baseNetwork = "JACINTO_V2"
#set True to match with ssd mobile n/w provided by Chuanqi
ssd_mobile_chuanqi = False
if ssd_mobile_chuanqi:
  postfix_char = '/'
else:  
  postfix_char = '_'

#'bvlc', 'nvidia', 'none'
bn_type = 'bvlc'
#'bvlc', 'nvidia'
caffe_fork= 'bvlc'
enable_dilation_enet=True
ex_net=False
bn_at_start=True
force_norm=False
use_shuffle=False

#'cifar10', 'IMGNET', 'SSD'
#for SSD there will not be downsample in conv5
training_type='SSD' 

#if True use dilation in base n/w in each layer after stride is removed
#Def(False) only supported for mobilenet
dil_when_stride_removed = False

#SCRATCH, IMGNET, IMGNET_VOC0712,IMGNET_VOC0712_CITY,KITTI,IMGNET_VOC0712_CITY512,IMGNET_VOC0712_CITY_TI_CAT,IMGNET_COCO,
#IMGNET_COCO_VOC0712Plus, CUSTOM, CITY_TI_CAT, IMG_VOC0712_CITY 
preTrainedModelType = "IMGNET"
#step, poly,multistep
LR_POLICY = "multistep"
#after how many steps (fraction of max_iter) lr rate will be dropped
#orig: 8/12,10/12,12/12
#stepvalue_fac = [0.9, 0.95, 1.0] 
#stepvalue_fac = [0.85, 0.95, 1.0] 
stepvalue_fac = [8.0/12.0, 10.0/12.0, 1.0] 
#stepvalue_fac = [6200.0/15100.0, 10000.0/15100.0, 1.0] 
#stepvalue_fac = [80000.0/125000.0, 110000.0/125000.0, 1.0] 
#stepvalue_fac = [0.7, 0.85, 1.0] 
test_interval = 2000
snapshot_interval = test_interval
#def 0.0005
weight_decay=0.0005
evaluate_difficult_gt = False
log_train_accu = False
# if while fine tuning few initial layers of base n/w need to be frozen
freezeInitFewLayers = True
dil_bf = True
train_on_diff_gt = False
#if small objects are more in the dataset like coco or TI automotive enable
small_objs = True
resize_width = 720
resize_height = 368
#'512x512', '300x300', '256x256'
ssd_size = '512x512' 
non_sq_dataset = False
#if set to true spatial res and ch at each head will be same as VGG16
heads_same_as_vgg = False

#if fine tune from earlier SSD trained model then dampen base_lr, def =1 (no dampen)
fine_tune_fac = 1.00
prefix_name = ""
depth_mul = 0.5
#This option is defined for only Jacinto_V2
fully_conv_at_end=True

# index of first additional layer after base network
first_idx = 6
# Set true if you want to start training right after generating all files.
run_soon = True
#############################################################################
#override params based on other params values
# if pre-trained model is not there then can't use freeze layers
if(preTrainedModelType =="SCRATCH"):
   freezeInitFewLayers = False

if('KITTI' in dataset):
  #use it only for Kitti eval to get close results as KITTI test script
  evaluate_difficult_gt = True

#VGG was not trained with batch norm at the begining
if baseNetwork == "VGG":
  bn_at_start=False

if baseNetwork == "JACINTO_V2":
  bn_at_start=False

if('VOC0712' in dataset):
  #VOC0712 has mostly big objects
  small_objs = False

#earlier training of Jacinto had force_norm True 
#if ('VGG' in baseNetwork) or ('JACINTO' in baseNetwork):
if 'VGG' in baseNetwork:
  force_norm = True 

if baseNetwork == "mobile":
  bn_at_start = False
  #enable disable based on exp. Has not given better results so far
  heads_same_as_vgg = True
  #to match with Manu's 0.5 mobilebnet pre-trained model
  postfix_char = '/'

  
if ssd_mobile_chuanqi and (baseNetwork == 'mobile'):
  #'cifar10', 'IMGNET', 'SSD'
  #for SSD there will not be downsample in conv5
  training_type='IMGNET' 
  first_idx = 14
  resize_width = 300
  resize_height = 300
  #'512x512', '300x300'
  ssd_size = '300x300' 

#############################################################################
# min_ratio
# VOC_300 20
# VOC_512 15
# coco_300 15
# coco_512 10

# in percent %
if ssd_size == '512x512':
  min_ratio = 15
elif (ssd_size == '300x300') or(ssd_size == '256x256'):
  min_ratio = 20

if small_objs: 
  min_ratio = min_ratio - 5
 
max_ratio = 90

if ssd_mobile_chuanqi:
  max_ratio = 95

if(dataset == "KITTI_MulTiles"):
   # for kitti training images are 368x368 as original height was 368
   resize_width = 368
   resize_height = 368
   # minimum dimension of input image
   min_dim = 368
elif(dataset == "KITTI") or ("KITTI_Per_Car_Cyclist" in dataset):
   resize_width = 1248
   resize_height = 384
   # minimum dimension of input image
   min_dim = 384
elif(dataset == "TI_mulTS_1024X512_V007_008_106") :
   resize_width = 1024
   resize_height = 512
   # minimum dimension of input image
   min_dim = 512
else:   
   resize_width = resize_width
   resize_height = resize_height
   # minimum dimension of input image
   min_dim = min(resize_width, resize_height)
###################################################################################
# Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
def AddExtraLayers(net, use_batchnorm=True, lr_mult=1, baseNetwork="VGG",
    ssd_size='512x512', training_type='SSD'):
    print("In AddExtraLayers()")
    print ("baseNetwork:", baseNetwork)

    #if (training_type == 'IMGNET') and (ssd_size == '512x512'):
    #  print("Not supported IMGNET trainign type with 512x512")
    #  sys.exit()

    if 'RES' in baseNetwork:
       use_relu = True

       # Add additional convolutional layers.
       # 19 x 19 (ssd300x300)
       last_layer = net.keys()[-1]

       # 10 x 10 (ssd300x300)
       from_layer = last_layer
       out_layer = "{}/conv1_1".format(last_layer)
       print ("out_layer: ", out_layer) 
       ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1)
       from_layer = out_layer

       out_layer = "{}/conv1_2".format(last_layer)
       print ("out_layer: ", out_layer) 
       ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2)
       from_layer = out_layer

       for i in xrange(2, 5):
         out_layer = "{}/conv{}_1".format(last_layer, i)
         print ("out_layer: ", out_layer) 
         ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1)
         from_layer = out_layer

         out_layer = "{}/conv{}_2".format(last_layer, i)
         print ("out_layer: ", out_layer) 
         ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2)
         from_layer = out_layer

       # Add global pooling layer.
       name = net.keys()[-1]
       print ("pool layer name: ", name) 
       net.pool6 = L.Pooling(net[name], pool=P.Pooling.AVE, global_pooling=True)
       return net
    else:
      use_relu = True

      bn_postfix='_bn'
      scale_postfix='_scale'

      if ssd_mobile_chuanqi: 
        bn_postfix='/bn'
        scale_postfix='/scale'

      if (ssd_size == '512x512') and (training_type == 'SSD'):
                        #32x32    #16x16    #8x8      #4x4      #2x2
        num_outputs =  [256, 512, 128, 256, 128, 256, 128, 256, 128, 256,]
        kernel_sizes = [  1,   3,   1,   3,   1,   3,   1,   3,   1,   4,]
        pads =         [  0,   1,   0,   1,   0,   1,   0,   1,   0,   1,]
        strides=       [  1,   2,   1,   2,   1,   2,   1,   2,   1,   1,]
      elif (ssd_size == '300x300') and (training_type == 'SSD'):
                       #19x19     #10x10    #5x5      #3x3
        num_outputs =  [256, 512, 128, 256, 128, 256, 128, 256,]
        kernel_sizes = [  1,   3,   1,   3,   1,   3,   1,   3,]
        pads =         [  0,   1,   0,   1,   0,   0,   0,   0,]
        strides=       [  1,   2,   1,   2,   1,   1,   1,   1,]
      elif (ssd_size == '256x256') and (training_type == 'SSD'):
                       #16x16     #8x8    #8x8      #4x4
        num_outputs =  [256, 512, 128, 256, 128, 256, 128, 256,]
        kernel_sizes = [  1,   3,   1,   3,   1,   3,   1,   3,]
        pads =         [  0,   1,   0,   1,   0,   0,   0,   0,]
        strides=       [  1,   2,   1,   1,   1,   1,   1,   1,]
      elif (ssd_size == '512x512') and (training_type == 'IMGNET'):
                        #16x16    #8x8      #4x4      #2x2
        num_outputs =  [256, 512, 128, 256, 128, 256, 128, 256,]
        kernel_sizes = [  1,   3,   1,   3,   1,   3,   1,   3,]
        pads =         [  0,   1,   0,   1,   0,   1,   0,   1,]
        strides=       [  1,   2,   1,   2,   1,   2,   1,   2,]
      elif (ssd_size == '300x300') and (training_type == 'IMGNET'):
                        #10x10    #5x5      #3x3       #2x2   
        num_outputs =  [256, 512, 128, 256, 128, 256,  64, 128,]
        kernel_sizes = [  1,   3,   1,   3,   1,   3,   1,   3,]
        pads =         [  0,   1,   0,   1,   0,   1,   0,   1,]
        strides=       [  1,   2,   1,   2,   1,   2,   1,   2,]


      from_layer = net.keys()[-1]
      blk_idx = first_idx
      for idx in range (0, len(num_outputs)):
        # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
        one_or_two = (idx%2) + 1
        out_layer = "conv{}_{}".format(blk_idx, one_or_two)
        ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_outputs[idx],
            kernel_sizes[idx], pads[idx], strides[idx], lr_mult=lr_mult, bn_postfix=bn_postfix,
            scale_postfix=scale_postfix)
        from_layer = out_layer
        if one_or_two == 2:
          blk_idx = blk_idx + 1

      return net

### Modify the following parameters accordingly ###
# The directory which contains the caffe code.
# We assume you are running the script at the CAFFE_ROOT.
caffe_root = os.getcwd()

# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
resume_training = True
# If true, Remove old model files.
remove_old_models = False

if (dataset=="VOC0712"):
    # The database file for training data. Created by data/VOC0712/create_data.sh
    train_data = "examples/VOC0712/VOC0712_trainval_lmdb"
    # The database file for testing data. Created by data/VOC0712/create_data.sh
    test_data = "examples/VOC0712/VOC0712_test_lmdb"
    # need separate LMDB for measuring train accuracy 
    train_accu_data = "examples/VOC0712_Temp/VOC0712_Temp_trainval_lmdb"
else:    
    train_data = "/user/a0875091/files/data/datasets/object-detect/ti/detection/xml/{}/lmdb/{}_train_lmdb".format(dataset,dataset)
    test_data = "/user/a0875091/files/data/datasets/object-detect/ti/detection/xml/{}/lmdb/{}_test_lmdb".format(dataset,dataset)
    # need separate LMDB for measuring train accuracy 
    train_accu_data = "/user/a0875091/files/data/datasets/object-detect/ti/detection/xml/{}/lmdb/{}_train_accu_lmdb".format(dataset,dataset)

resize = "{}x{}".format(resize_width, resize_height)

# Specify the batch sampler.
batch_sampler = [
        {
                'sampler': {
                        },
                'max_trials': 1,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.1,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.3,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.5,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.7,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.9,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'max_jaccard_overlap': 1.0,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        ]


if bn_at_start:
  mean_value = [0, 0, 0]
else:  
  if 'JACINTO' in baseNetwork:
    mean_value = [128, 128, 128]
  elif baseNetwork == "mobile":
    mean_value = [103.94,116.78,123.68]
    if ssd_mobile_chuanqi:
      mean_value = [127.5,127.5,127.5]
  else:
    #Imagenet mean vec
    mean_value = [104, 117, 123]

resize_param_train_transform = {
  'prob': 1,
  'height': resize_height,
  'width': resize_width,
  'interp_mode': [
          P.Resize.LINEAR,
          P.Resize.AREA,
          P.Resize.NEAREST,
          P.Resize.CUBIC,
          P.Resize.LANCZOS4,
          ],
  }

if non_sq_dataset:
  resize_param_train_transform['resize_mode'] = P.Resize.FIT_SMALL_SIZE
  resize_param_train_transform['height_scale'] = resize_height
  resize_param_train_transform['width_scale'] = resize_width
else:
  resize_param_train_transform['resize_mode'] = P.Resize.WARP

train_transform_param = {
        'mirror': True,
        'mean_value': mean_value,
        'resize_param': resize_param_train_transform,
        'distort_param': {
                'brightness_prob': 0.5,
                'brightness_delta': 32,
                'contrast_prob': 0.5,
                'contrast_lower': 0.5,
                'contrast_upper': 1.5,
                'hue_prob': 0.5,
                'hue_delta': 18,
                'saturation_prob': 0.5,
                'saturation_lower': 0.5,
                'saturation_upper': 1.5,
                'random_order_prob': 0.0,
                },
        'expand_param': {
                'prob': 0.5,
                'max_expand_ratio': 4.0,
                },
        'emit_constraint': {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
            }

        }
if baseNetwork == "mobile":
  train_transform_param['scale'] = 0.017
  if ssd_mobile_chuanqi: 
    train_transform_param['scale'] = 0.007843


resize_param_test_transform = {
                'prob': 1,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [P.Resize.LINEAR],
                }

if non_sq_dataset:
  resize_param_test_transform['resize_mode'] = P.Resize.FIT_SMALL_SIZE
  resize_param_test_transform['height_scale'] = resize_height
  resize_param_test_transform['width_scale'] = resize_width
else:
  resize_param_test_transform['resize_mode'] = P.Resize.WARP

test_transform_param = {
        'mean_value': mean_value,
        'resize_param': resize_param_test_transform,
        }
if baseNetwork == "mobile":
  test_transform_param['scale'] = 0.017
  if ssd_mobile_chuanqi: 
    test_transform_param['scale'] = 0.007843

if ('RES' in baseNetwork) or (baseNetwork=='ENET'): 
  # use batch norm for all newly added layers.
  # Currently only the non batch norm version has been tested.
  use_batchnorm = True
  # use batch norm for mbox layers 
  use_batchnorm_mbox = True
elif (baseNetwork == 'mobile'):  
  use_batchnorm = True
  use_batchnorm_mbox = False
else:
  use_batchnorm = False
  use_batchnorm_mbox = False

lr_mult = 1
    
# Use different initial learning rate.
if use_batchnorm:
    base_lr = 0.0004
    #set it same as non BN version
    base_lr *= 0.1
else:
    # A learning rate for batch_size = 1, num_gpus = 1.
    base_lr = 0.00004

#if already fine tune dampen base_lr 
base_lr *= fine_tune_fac   

#get first three chars from each model name
modelStrToPrint = '' 
for model in preTrainedModelType.split("_"):
  modelStrToPrint = modelStrToPrint + model[0:3] + "_"
#take last "_" out  
modelStrToPrint = modelStrToPrint[0:-1] 

# Modify the job name if you want.
job_name="SSD_{}_Pre_{}_frz_{:.1}_bn_{:.1}_bnStart_{:.1}_tr_{:.3}_norm_{:.1}_fc_{:.1}".format(resize,modelStrToPrint,str(freezeInitFewLayers),
        bn_type, str(bn_at_start),str(training_type), str(force_norm),
        str(fully_conv_at_end))

if non_sq_dataset:
  job_name="{}_nsq_{:.1}".format(job_name, str(non_sq_dataset))

if baseNetwork == 'ENET':
  job_name="{}_DilENet_{:.1}_exNet_{:.1}_dilBF_{:1}".format(job_name, str(enable_dilation_enet), str(ex_net), dil_bf)

if baseNetwork == 'JACINTO_V2':
  job_name="{}_fc_{:.1}".format(job_name, str(fully_conv_at_end))

if ssd_mobile_chuanqi:
  job_name="{}_chuanqi".format(job_name)

if use_shuffle:
  job_name="{}_shfl_{:.1}".format(job_name,str(use_shuffle))

if heads_same_as_vgg:
  job_name="{}_vggHead_{:.1}".format(job_name,str(heads_same_as_vgg))

if (depth_mul != 1.0) and ('mobile' in baseNetwork):
  job_name="{}_depth_{}".format(job_name,depth_mul)

job_name="{}{}".format(job_name, prefix_name)

# The name of the model. Modify it if you want.
model_name = "{}_{}_{}".format(baseNetwork, dataset,job_name)

# Directory which stores the model .prototxt file.
save_dir = "models/{}/{}/{}".format(baseNetwork, dataset,job_name)
# Directory which stores the snapshot of models.
snapshot_dir = "models/{}/{}/{}".format(baseNetwork, dataset,job_name)
# Directory which stores the job script and log file.
#job_dir = "jobs/{}/{}/{}".format(baseNetwork, dataset,job_name)
job_dir = save_dir+"/jobs"
# Directory which stores the detection results.
#output_result_dir = "{}/data/VOCdevkit/results/{}/{}/Main".format(os.environ['HOME'], dataset, job_name)
output_result_dir = save_dir+"/results/"
print ("output_result_dir :", output_result_dir) 
if not os.path.exists(output_result_dir):
    os.makedirs(output_result_dir)
    print ("result dir creatd")

# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
# snapshot prefix.
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
# job script path.
job_file = "{}/{}.sh".format(job_dir, model_name)

#Select Pre-train model 
def pretrain_jacinto():
  pretrain_model= ""
  if(preTrainedModelType == "IMGNET"):
     # Imagenet pre-trained model for Jacinto Net
     pretrain_model = "examples/imagenet/JacintoNet/training_bvlcBatchNorm_cudnn5_1_60.5/store/convnet10_iter_160000.caffemodel"
  elif(preTrainedModelType == "IMGNET_VOC0712"):
    # IMGNET-VOC0712 pre-trained model for SSD(Jacinto)
    #ssd-2016
    #pretrain_model = "models/JACINTO/VOC0712/SSD_BN_MyPreTrained_500x500/store/JACINTO_VOC0712_SSD_BN_MyPreTrained_500x500_iter_83000.caffemodel"
    #ssd-2017
    pretrain_model = "models/JACINTO/VOC0712/SSD_512x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_F/JACINTO_VOC0712_SSD_512x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_iter_118000.caffemodel"
  elif(preTrainedModelType == "IMGNET_VOC0712_CITY"):
    #IMGNET-VOC0712-CITY pre-trained model for SSD(Jacinto)
    pretrain_model = "models/JACINTO/CITY/SSD_BN_MyPreTrained_500x500/store/JACINTO_CITY_SSD_BN_MyPreTrained_500x500_iter_60000.caffemodel"
  
  return pretrain_model 


def pretrain_jacinto_v2():
  pretrain_model= ""
  if(preTrainedModelType == "IMGNET"):
    #Top1-60.88, NVIDIA BN converted to BVLC BN
    pretrain_model = "examples/imagenet/JacintoNetV2/Pre-Trained/test_cleanedUpNames_bn_b/imagenet_jacintonet11v2_iter_320000_cleanedUpNames_bn_b.caffemodel"
  elif(preTrainedModelType == "IMGNET_VOC0712"):
    #VOC0712 67.63%
    #pretrain_model = "models/JACINTO_V2/VOC0712/SSD_512x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_F/JACINTO_V2_VOC0712_SSD_512x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_iter_118000.caffemodel"
    
    #VOC0712-70.74 (padding corrected to 32x32 from 30x30)
    pretrain_model = "models/JACINTO_V2/VOC0712/SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_70.74/JACINTO_V2_VOC0712_SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_iter_64000.caffemodel"
  elif(preTrainedModelType == "IMGNET_VOC0712_CITY"):
    #pretrain_model = "models/JACINTO_V2/CITY_512/SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_26.61/JACINTO_V2_CITY_512_SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_26.61_iter_35000.caffemodel"
    pretrain_model = "models/JACINTO_V2/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_32.73/JACINTO_V2_CITY_720x368_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_32.73_iter_36000.caffemodel"
  elif(preTrainedModelType == "IMG_VOC0712_CITY_TI_CAT"):
    #det 39.xx
    pretrain_model = "models/JACINTO_V2/CITY_720x368_TI_CATEGORY/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_shfl_T/JACINTO_V2_CITY_720x368_TI_CATEGORY_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_shfl_T_iter_22000.caffemodel"
  return pretrain_model 


def pretrain_vgg():
  pretrain_model= ""
  if(preTrainedModelType == "IMGNET"):
    # Imagenet pre-trained model for VGG Net
    # The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
    pretrain_model = "models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
  elif(preTrainedModelType == "IMGNET_VOC0712"):
    # Imagenet+VOC0712 pre-trained model for SSD(VGG)
    #pretrain_model = "models/VGGNet/VOC0712_preTrained/SSD_500x500/VGG_VOC0712_SSD_500x500_iter_60000.caffemodel"
    #Imgnet + Coco + VOC07++12 (ft), 83.2%
    #pretrain_model = "models/VGGNet/pre-trained-models/VOC0712/SSD_512x512_ft/VGG_VOC0712_SSD_512x512_ft_iter_120000.caffemodel"
    #ImageNet+VOC0712 pre-trained for SSD(VGG). 79.8%
    #pretrain_model = "models/VGGNet/pre-trained-models/VOC0712/SSD_512x512/VGG_VOC0712_SSD_512x512_iter_120000.caffemodel"
    #Soyeb's retrained gets 0.5% better (80.32%) 
    pretrain_model = "models/VGG/VOC0712/SSD_512x512_Pre_IMG_frz_F_bn_b_bnStart_F_tr_IMG_norm_T/VGG_VOC0712_SSD_512x512_Pre_IMG_frz_F_bn_b_bnStart_F_tr_IMG_norm_T_iter_120000.caffemodel"
  elif(preTrainedModelType == "IMGNET_VOC0712_CITY"):
    #IMGNET-VOC0712-CITY pre-trained model for SSD(VGG)
    #old path del later
    #pretrain_model = "models/TITrainedOp/20161101_PASCAL_CITYSCAPE/SSD_CITYSCAPE_500x500/store/VGG_VOC0712_CITYSCAPE_SSD_CITYSCAPE_500x500_iter_20000.caffemodel"
    #pretrain_model = "models/VGG/CITY/SSD_CITYSCAPE_500x500/store/VGG_VOC0712_CITYSCAPE_SSD_CITYSCAPE_500x500_iter_20000.caffemodel"
    #detEval=40.96 on cityscapes
    #pretrain_model = "models/VGG/cityscape/SSD_512x512_Pre_CUS_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_detEval_40.96/VGG_cityscape_temp_SSD_512x512_Pre_CUS_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_iter_19000.caffemodel"

    #detEval=41.95 on VOC0712+cityscapes+city_720x368(5k ite)
    #pretrain_model = "models/VGG/cityscape/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_FregHd19/VGG_cityscape_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_FregHd19_iter_4844.caffemodel"
    
    #detEval=57.85%(50.62%) on VOC0712+cityscapes(1024x512)
    #pretrain_model = "models/VGG/cityscape/SSD_1024x512_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_regHd19/VGG_cityscape_SSD_1024x512_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_regHd19_iter_18000.caffemodel"

     #50.72% on cityscape, 1024x512
     #pretrain_model = "/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/cityscape/SSD_1024x512_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_regHd19_50.72/VGG_cityscape_SSD_1024x512_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_regHd19_50.72_iter_14000.caffemodel"

    #50.59% on city_720x368
    pretrain_model = "models/VGG/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_FregHd19_50.59/VGG_CITY_720x368_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_FregHd19_50.59_iter_26000.caffemodel"

  elif(preTrainedModelType == "IMGNET_VOC0712_CITY_TI"):
    pretrain_model="models/VGG/TI2017_V7_106_MulTiles/SSD_512x512_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_66.42/VGG_TI2017_V7_106_MulTiles_SSD_512x512_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_iter_25000.caffemodel"

  elif(preTrainedModelType == "IMGNET_VOC0712_CITY_TI_CAT"):
    #53.14 on city_720x368 TI Category
    pretrain_model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_720x368_TI_CATEGORY/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_53.14/VGG_CITY_720x368_TI_CATEGORY_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_53.14_iter_9900.caffemodel"

  elif(preTrainedModelType == "IMGNET_COCO"):
    pretrain_model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGGNet/pre-trained-models/coco/SSD_512x512/VGG_coco_SSD_512x512_iter_360000.caffemodel"

  elif(preTrainedModelType == "IMGNET_COCO_VOC0712Plus"):
    pretrain_model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGGNet/pre-trained-models/VOC0712Plus/SSD_512x512_ft/VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.caffemodel"
  elif(preTrainedModelType == "CUSTOM"):
    #IMGNET-KITTI pre-trained model for SSD(VGG)
    #pretrain_model="models/VGG/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_CUSTOM_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed/jobs/VGG_KITTI_Per_Car_Cyclist_SSD_1248x384_Pre_CUSTOM_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed_iter_21000.caffemodel"
    #pretrain_model="models/VGG/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_detEval_32.10/VGG_KITTI_Per_Car_Cyclist_SSD_1248x384_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_iter_42000.caffemodel"
    #one round of SSD training with IMGNET+VOC0712+cityscapes with train_diff_gt=F and lr=5E-5
    pretrain_model="models/VGG/cityscape/SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_detEval_26.62/VGG_cityscape_SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_detEval_26.62_iter_19000.caffemodel"
    #SSD_512 (40.1%)
    #pretrain_model = "models/VGG/CITY_512/SSD_512x512_Pre_CUS_frz_T_bn_b_bnStart_F_tr_SSD_norm_T/VGG_CITY_512_SSD_512x512_Pre_CUS_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_iter_19000.caffemodel"
    #pretrain_model="models/VGG/TI_201712_720x368_V1/SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.04/VGG_TI_201712_720x368_V1_SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.04_iter_14000.caffemodel"

  elif(preTrainedModelType == "CITY_TI_CAT"):
    #53.14%
    pretrain_model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_720x368_TI_CATEGORY/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_53.14/VGG_CITY_720x368_TI_CATEGORY_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_53.14_iter_9900.caffemodel"

  elif(preTrainedModelType == "IMG_CITY_TI201712_V1"):
    pretrain_model="models/VGG/TI_201712_720x368_V1/SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.04/VGG_TI_201712_720x368_V1_SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.04_iter_14000.caffemodel"
  
  return pretrain_model 

def pretrain_RES101():
  pretrain_model= ""
  if(preTrainedModelType == "IMGNET"):
    # The pretrained ResNet101 model from https://github.com/KaimingHe/deep-residual-networks.
    pretrain_model = "models/ResNet/ResNet-101-model.caffemodel"
  elif(preTrainedModelType == "IMGNET_VOC0712"):
    sys.exit()
  elif(preTrainedModelType == "IMGNET_VOC0712_CITY"):
    sys.exit()
  return pretrain_model 

def pretrain_JINET():
  pretrain_model= ""
  if(preTrainedModelType == "IMGNET"):
    # The pretrained (top-1: 59.6, top-5: 82.4).
    pretrain_model = "examples/imagenet/JacintoNet/training_JINet/logs/JINet_iter_160000.caffemodel"

  return pretrain_model 


def pretrain_RESTEN():
  pretrain_model= ""
  if(preTrainedModelType == "IMGNET"):
    # The pretrained ResNet101 model from https://github.com/KaimingHe/deep-residual-networks.
    pretrain_model = "models/ResNet/cvjenna/resnet10_cvgj_iter_320000.caffemodel"
  elif(preTrainedModelType == "IMGNET_VOC0712"):
    sys.exit()
  elif(preTrainedModelType == "IMGNET_VOC0712_CITY"):
    sys.exit()
  return pretrain_model 

def pretrain_ENET():
  pretrain_model= ""
  if(preTrainedModelType == "KITTI"):
    #KITTI ENET trained detection_eval 77.95%
    pretrain_model = "models/ENET/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed_bvlcBN_v2_detEval_77.95/ENET_KITTI_Per_Car_Cyclist_SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed_bvlcBN_v2_iter_41000_cleanedUpNames.caffemodel"

  elif(preTrainedModelType == "IMGNET"):
    # The pretrained (top-1: 65.6, top-5: 86.6) using NVIDIA-Caffe
    #pretrain_model = "examples/imagenet/training_ENET_nvidia/Top1_65.6_Top5_86.6/ENet_NVIDIA_BN_iter_640000.caffemodel"
    # The pretrained (top-1: 63.8, top-5: 84.9). fine tuned with BVLC-caffe
    #pretrain_model = "examples/imagenet/training_ENET_bvlc_dil_0_Top1_63.8/FineTuneFromNVIDIA_BN_Top1_63.8_Top5_84.9/ENet_NVIDIA_BN_iter_740000.caffemodel"
    
    # changed names from above model
    # The pretrained (top-1: 63.8, top-5: 84.9). fine tuned with BVLC-caffe
    #pretrain_model = "examples/imagenet/training_ENET_bvlc_dil_0_Top1_63.8/FineTuneFromNVIDIA_BN_Top1_63.8_Top5_84.9/CleanedUpNames/ENet_NVIDIA_BN_iter_740000_cleanedUpNames.caffemodel"
    #ENet_Dil1_BF Top-1 66.76, converted to BVLC BN model
    pretrain_model = "examples/imagenet/20170609_training_ENET_bvlc_dil_1_BF_Top1_xx.xx/ENet_BVLC_BN_iter_640000.caffemodel"

  elif(preTrainedModelType == "IMGNET_VOC0712"):
    #ENET_Dil1_BF_VOC_70.14  
    pretrain_model = "models/ENET/VOC0712/SSD_512x512_Pre_IMG_frz_F_bn_b_bnStart_T_tr_SSD_norm_T_DilENet_T_exNet_F_dilBF_1_70.14/ENET_VOC0712_SSD_512x512_Pre_IMG_frz_F_bn_b_bnStart_T_tr_SSD_norm_T_DilENet_T_exNet_F_dilBF_1_70.14_iter_120000.caffemodel"

  elif(preTrainedModelType == "IMGNET_VOC0712_CITY"):
    #pretrain_model = "models/ENET/CITY_512/SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_T_tr_SSD_norm_F_DilENet_T_exNet_F_dilBF_1_poly_32.42/ENET_CITY_512_SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_T_tr_SSD_norm_F_DilENet_T_exNet_F_dilBF_1_poly_32.42_iter_24000.caffemodel"	
    pretrain_model = "models/ENET/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_T_tr_SSD_norm_F_fc_F_DilENet_T_exNet_F_dilBF_1/ENET_CITY_720x368_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_T_tr_SSD_norm_F_fc_F_DilENet_T_exNet_F_dilBF_1_38.79_iter_36000.caffemodel"

  return pretrain_model

def pretrain_EXNET():
  pretrain_model= ""
  if(preTrainedModelType == "KITTI"):
    # Kitti EXNET trained detection_eval 77.37%
    pretrain_model = "models/ENET/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_IMGNET_frz_F_wideAR_F_minRatio_15_bn_b_DilENet_T_exNet_T_bnStart_T_tr_SSD_77.37/jobs/ENET_KITTI_Per_Car_Cyclist_temp_SSD_1248x384_Pre_IMGNET_frz_F_wideAR_F_minRatio_15_bn_b_DilENet_T_exNet_T_bnStart_T_tr_SSD_iter_40700.caffemodel"

  elif(preTrainedModelType == "IMGNET"):
    if enable_dilation_enet:
      # The pretrained exnet with dilation(bvlc top-1: 61.8, (nvidia-62.38).
      pretrain_model = "examples/imagenet/training_EXNET_bvlc_dil_1_Top1_61.81/store/EXNet_Gr4_BVLC_BN_iter_187600.caffemodel"
      #use SSD model trained ds=2 in conv5 layer, ssd_detection_eval = 74.65%
      #pretrain_model = "models/ENET/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_bn_bvlc_DilENet_True_exNetLateDS_True_cifar10_False_bnAtStart_True_detEval_74.65/ENET_KITTI_Per_Car_Cyclist_SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_bn_bvlc_DilENet_True_exNetLateDS_True_cifar10_False_bnAtStart_True_iter_41000.caffemodel"
    else:
      # The pretrained exnet w/o dilation (top-1: 64.56 nvidia bn absorbed)
      #This failed
      #pretrain_model = "examples/imagenet/training_EXNET_nvidia_dil_0_Top1_64.56/absorb_bn/op/EXNet_Gr4_Dil_0_NVIDIA_BN_iter_640000_bn_absorbed.caffemodel"
      #Converted weigths from nvidia BN to bvlc BN (top-1: 64.56). SSD is
      #not learning with this
      pretrain_model = "examples/imagenet/training_EXNET_nvidia_dil_0_Top1_64.56/absorb_bn/EXNet_Gr4_Dil_0_NVIDIA_BN_iter_640000_converted_to_bvlc_bn.caffemodel"
      #Converted weigths from nvidia BN to bvlc BN (top-1: 64.56). 
      #Further finetuned (16k ite, top-1: 64.71)
      #pretrain_model = "examples/imagenet/training_EXNET_nvidia_dil_0_Top1_64.56/test_bvlc/test_bn_converted_to_bvlc/EXNet_Gr4_Dil_0_NVIDIA_BN_iter_15000.caffemodel"
      #Nvidia BN (top-1: 64.56). Fine tuned from absorb BN weights
      #(70k,top-1:61.98). It's working ok. need to run more. 
      #pretrain_model = "examples/imagenet/training_EXNET_nvidia_dil_0_Top1_64.56/train_bvlc/logs_ite_70k_Top1_61.98/EXNet_Gr4_Dil_0_BVLC_BN_iter_70000.caffemodel"
  elif(preTrainedModelType == "IMGNET_VOC0712"):
    #ENET_Dil1_BF_VOC_70.14  
    pretrain_model = "models/ENET/VOC0712/SSD_512x512_Pre_IMG_frz_F_bn_b_bnStart_T_tr_SSD_norm_T_DilENet_T_exNet_F_dilBF_1_70.14/ENET_VOC0712_SSD_512x512_Pre_IMG_frz_F_bn_b_bnStart_T_tr_SSD_norm_T_DilENet_T_exNet_F_dilBF_1_70.14_iter_120000.caffemodel"

  return pretrain_model


def pretrain_mobile():
  pretrain_model = "" 
  if(preTrainedModelType == "IMGNET"):
    if depth_mul == 1.0:  
      #IMGNET-70.82, cleaned up names
      pretrain_model = "examples/imagenet/mobilenet/MobileNet-Caffe-master_70.82/cleanedUpNames/imagenet_mobilenet_iter_10000_cleanedUp.caffemodel"
    elif depth_mul == 0.5:
      #IMGNET-63.6, depth_mul=0.5, trained on caffe-0.16 by Manu
      pretrain_model = "/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/examples/imagenet/imagenet_mobilenet-0.5_2017-08-24_18-29-42_(63.6%)/initial/imagenet_mobilenet-0.5_iter_320000.caffemodel"
  elif (preTrainedModelType == "IMGNET_VOC0712"):
    #IMGNET+ VOC (61.35%, reg. head at conv4_1)
    #pretrain_model = "models/mobile/VOC0712/SSD_512x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_F/mobile_VOC0712_SSD_512x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_F_iter_125000.caffemodel"
    
    #IMGNET+ VOC (67.8%, reg. head at conv5_5)
    #pretrain_model = "models/mobile/VOC0712/SSD_300x300_Pre_IMG_frz_T_bn_b_bnStart_F_tr_IMG_norm_F_fc_F_regHdDwn_close2Chuanqi/mobile_VOC0712_SSD_300x300_Pre_IMG_frz_T_bn_b_bnStart_F_tr_IMG_norm_F_fc_F_regHdDwn_close2Chuanqi_iter_105000.caffemodel"
    
    #det_eval = 69.57%
    #pretrain_model = "models/mobile/VOC0712/SSD_300x300_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_F_regHdDwn_close2Chuanqi/mobile_VOC0712_SSD_300x300_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_F_regHdDwn_close2Chuanqi_iter_90000.caffemodel"
    #VOC0712 71.66%
    #pretrain_model = "models/mobile/VOC0712/exp1_SSD_512x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_IMG_norm_F_fc_F_regHdDwn_close2Chuanqi/mobile_VOC0712_exp1_SSD_512x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_IMG_norm_F_fc_F_regHdDwn_close2Chuanqi_iter_120000.caffemodel"

    #VOC0712_tr_SSD 73.63%
    pretrain_model = "models/mobile/VOC0712/SSD_512x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_F_regHd19_73.63/mobile_VOC0712_SSD_512x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_F_regHd19_73.63_iter_125000.caffemodel"

    #VOC0712 71.54%
    if heads_same_as_vgg:
      pretrain_model = "models/mobile/VOC0712/SSD_512x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_vggHead_T_71.54/mobile_VOC0712_SSD_512x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_vggHead_T_71.54_iter_118000.caffemodel"

    if ssd_mobile_chuanqi:
      #VOC0712 - 72.3%  
      pretrain_model = "models/mobile/external/MobileNet-SSD/MobileNetSSD_train.caffemodel"
   
    if depth_mul == 0.5:
      #VOC0712-62.14 -mobilenet-0.5
      pretrain_model = "models/mobile/VOC0712/SSD_512x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_vggHead_T_depth_0.5_62.14/mobile_VOC0712_SSD_512x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_vggHead_T_depth_0.5_iter_120000.caffemodel"
  elif (preTrainedModelType == "IMGNET_VOC0712_CITY512"):
    #34.60%(30.28%) on CITY_512
    pretrain_model = "models/mobile/CITY_512/SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_FregHd19_30.28/mobile_CITY_512_SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_FregHd19_30.28_iter_33000.caffemodel"

  elif (preTrainedModelType == "IMGNET_VOC0712_CITY_720x368"):
    #40.54% on CITY_720x368  
    pretrain_model = "/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/mobile/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_FregHd19_40.54/mobile_CITY_720x368_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_FregHd19_40.54_iter_36000.caffemodel"
  elif (preTrainedModelType == "IMGNET_VOC0712_CITY720x368_TICAT"):
    if heads_same_as_vgg:  
      #43.1% on CITY_720x368 TI Cat  
      pretrain_model = "models/mobile/CITY_720x368_TI_CATEGORY/SSD_720x368_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_vggHead_T_43.1/mobile_CITY_720x368_TI_CATEGORY_SSD_720x368_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_vggHead_T_43.1_iter_36000.caffemodel"
  return pretrain_model

#Select Pre-train model 
pretrain_model="" 
if (baseNetwork == "JACINTO"):
  pretrain_model = pretrain_jacinto()
elif (baseNetwork == "JACINTO_V2"): 
  pretrain_model = pretrain_jacinto_v2()
elif(baseNetwork == "VGG"):
  pretrain_model = pretrain_vgg()
elif(baseNetwork == "RES101"):
  pretrain_model = pretrain_RES101()
elif(baseNetwork == "RESTEN"):
  pretrain_model = pretrain_RESTEN()
elif(baseNetwork == "JINET"):
  pretrain_model = pretrain_JINET()
elif(baseNetwork == "ENET"):
  if ex_net: 
    pretrain_model = pretrain_EXNET()
  else:  
    pretrain_model = pretrain_ENET()
elif(baseNetwork == "mobile"):
  pretrain_model = pretrain_mobile()
else:
  print("wrong base n/w")  
  sys.exit()  


if (dataset=="VOC0712"):
    # Stores the test image names and sizes. Created by data/VOC0712/create_list.sh
    name_size_file = "data/VOC0712/test_name_size.txt"
    # Stores LabelMapItem.
    label_map_file = "data/VOC0712/labelmap_voc.prototxt"
    num_classes = 21
    num_train_image = 16551
    # Evaluate on whole test set.
    num_test_image = 4952
    conf_name = "mbox_conf"
    #default:225
    numEpoch = 225
    train_on_diff_gt = True
    # this name size order should match with that was genearted while creating LMDB
    train_name_size_file = "data/VOC0712_Temp/trainval_name_size.txt"

elif (dataset=="cityscape"):
    name_size_file = "data/cityscape/test_name_size.txt"
    label_map_file = "data/cityscape/labelmap_cityscape.prototxt"
    num_classes = 9
    num_train_image = 2726
    num_test_image = 498
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 200

elif (dataset=="cityscapes_wo_ignored"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 8
    num_train_image = 2726
    num_test_image = 498
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 200

elif (dataset=="CITY_512"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 9
    num_train_image = 2726
    num_test_image = 498
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 400

elif (dataset=="CITY_720x368"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    #removed ignored class
    num_classes = 8
    num_train_image = 2726
    num_test_image = 498
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 400

elif (dataset=="CITY_720x368_TI_CATEGORY"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    #removed ignored class
    num_classes = 4
    num_train_image = 2726
    num_test_image = 498
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 400

elif (dataset=="CITY_1024x512"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    #removed ignored class
    num_classes = 8
    num_train_image = 2726
    num_test_image = 498
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 400

elif (dataset=="TI2016"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap_ti.prototxt".format(dataset)
    num_classes = 5
    num_train_image = 1590
    num_test_image = 526
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 500

elif (dataset=="TI2017"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap_ti.prototxt".format(dataset)
    num_classes = 5
    num_train_image = 29208
    num_test_image = 3156
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 50
    train_on_diff_gt = False


elif (dataset=="TI2017_Tiny") or  (dataset=="TI2017_Tiny_MulTiles_minSize12x32") or (dataset=="TI2017_Tiny_MulTiles_minSize24x56"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap_ti.prototxt".format(dataset)
    num_classes = 2
    num_train_image = 32
    num_test_image = 32
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 3000

elif (dataset== "TI2017_SingleImg"):
    #it is actually 4 images
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    #ImageNet- VGGNet
    #if(baseNetwork == "VGG"):
        #pretrain_model ="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI2017_Tiny_MulTiles_minSize12x32/SSD_BN_512x512_freeze_True_wideAR_True/store/VGG_TI2017_Tiny_MulTiles_SSD_BN_512x512_freeze_True_wideAR_True_iter_2902.caffemodel"
    label_map_file = "data/{}/labelmap_ti.prototxt".format(dataset)
    num_classes = 2
    num_train_image = 4
    num_test_image = 4
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 2000

elif (dataset=="TI2017_Tiny_MulTiles_mulCat"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap_ti.prototxt".format(dataset)
    num_classes = 3
    num_train_image = 32
    num_test_image = 32
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 3000
    train_on_diff_gt = False

elif (dataset=="TI2017_V105"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap_ti.prototxt".format(dataset)
    num_classes = 4
    num_train_image = 2743
    num_test_image = 2743
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 1200
    train_on_diff_gt = False

elif (dataset=="TI2017_V7_106_MulTiles"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap_ti.prototxt".format(dataset)
    num_classes = 4
    num_train_image = 5100
    num_test_image = 2743
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 150
    train_on_diff_gt = False

elif (dataset=="KITTI_MulTiles"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image = 20998
    num_test_image = 3146
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 100

elif (dataset=="KITTI"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 3
    num_train_image = 6500
    num_test_image = 981
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 100

elif (dataset=="KITTI_Per_Car_Cyclist"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image = 6500
    num_test_image = 981
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 200

elif (dataset=="KITTI_Per_Car_Cyclist_P1P2"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image = 7481
    num_test_image = 981
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 50

elif (dataset=="TI_mulTS_1024X512_V007_008_106"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image = 23293
    num_test_image = 2622
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 32
    # this name size order should match with that was genearted while creating LMDB
    train_name_size_file = "data/{}/train_name_size.txt".format(dataset)
    train_on_diff_gt = False

elif (dataset=="TI_201708"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image = 298 
    num_test_image = 1000 #3609 
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 32
    # this name size order should match with that was genearted while creating LMDB
    train_name_size_file = "data/{}/train_name_size.txt".format(dataset)
    train_on_diff_gt = False

elif (dataset=="TI_201708_720x368"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image = 2851  
    num_test_image = 1000 #3609 
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 200
    # this name size order should match with that was genearted while creating LMDB
    train_name_size_file = "data/{}/train_name_size.txt".format(dataset)
    train_on_diff_gt = False

elif (dataset=="TI_201708_720x368_V106"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image = 547
    num_test_image = 547 
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 200
    # this name size order should match with that was genearted while creating LMDB
    train_name_size_file = "data/{}/train_name_size.txt".format(dataset)
    train_on_diff_gt = False

elif (dataset=="TI_201708_720x368_V2"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image =2304
    num_test_image = 547 
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 200
    # this name size order should match with that was genearted while creating LMDB
    train_name_size_file = "data/{}/train_name_size.txt".format(dataset)
    train_on_diff_gt = False

elif (dataset=="TI_201708_720x368_V3"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image =2304
    num_test_image = 547 
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 200
    # this name size order should match with that was genearted while creating LMDB
    train_name_size_file = "data/{}/train_name_size.txt".format(dataset)
    train_on_diff_gt = False

elif (dataset=="TI_201708_720x368_V4"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image =5913
    num_test_image = 547 
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 400
    # this name size order should match with that was genearted while creating LMDB
    train_name_size_file = "data/{}/train_name_size.txt".format(dataset)
    train_on_diff_gt = False

elif (dataset=="TI_201708_720x368_CITY"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image =9137
    num_test_image = 547 
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 200
    # this name size order should match with that was genearted while creating LMDB
    train_name_size_file = "data/{}/train_name_size.txt".format(dataset)
    train_on_diff_gt = False

elif (dataset=="TI_201712_CITY_720x368_V1"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image =4243
    num_test_image = 3609 
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 200
    # this name size order should match with that was genearted while creating LMDB
    train_name_size_file = "data/{}/train_name_size.txt".format(dataset)
    train_on_diff_gt = False

elif (dataset=="TI_201712_CITY_720x368_V2"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image =4836
    num_test_image = 3609 
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 200
    # this name size order should match with that was genearted while creating LMDB
    train_name_size_file = "data/{}/train_name_size.txt".format(dataset)
    train_on_diff_gt = False

elif (dataset=="TI_201712_CITY_720x368_V3"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image =6473
    num_test_image = 3609 
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 200
    # this name size order should match with that was genearted while creating LMDB
    train_name_size_file = "data/{}/train_name_size.txt".format(dataset)
    train_on_diff_gt = False
	
elif (dataset=="TI_201712_720x368_V1"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image =3250
    num_test_image = 3609 
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 200
    # this name size order should match with that was genearted while creating LMDB
    train_name_size_file = "data/{}/train_name_size.txt".format(dataset)
    train_on_diff_gt = False

elif (dataset=="TI_201712_1024x512"):
    name_size_file = "data/{}/test_name_size.txt".format(dataset)
    label_map_file = "data/{}/labelmap.prototxt".format(dataset)
    num_classes = 4
    num_train_image =3250
    num_test_image = 3609 
    conf_name = "mbox_conf_{}class".format(num_classes)
    numEpoch = 200
    # this name size order should match with that was genearted while creating LMDB
    train_name_size_file = "data/{}/train_name_size.txt".format(dataset)
    train_on_diff_gt = False

# MultiBoxLoss parameters.
share_location = True
background_label_id=0
normalization_mode = P.Loss.VALID
code_type = P.PriorBox.CENTER_SIZE
ignore_cross_boundary_bbox = False
mining_type = P.MultiBoxLoss.MAX_NEGATIVE
neg_pos_ratio = 3.
loc_weight = (neg_pos_ratio + 1.) / 4.
multibox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'loc_weight': loc_weight,
    'num_classes': num_classes,
    'share_location': share_location,
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.5,
    'use_prior_for_matching': True,
    'background_label_id': background_label_id,
    'use_difficult_gt': train_on_diff_gt,
    'mining_type': mining_type,
    'neg_pos_ratio': neg_pos_ratio,
    'neg_overlap': 0.5,
    'code_type': code_type,
    'ignore_cross_boundary_bbox': ignore_cross_boundary_bbox,
    }
loss_param = {
    'normalization': normalization_mode,
    }

# parameters for generating priors.

# SSD_Size : 512x512
# conv4_3 ==> 64 x 64
# fc7 ==> 32 x 32
# conv6_2 ==> 16 x 16
# conv7_2 ==> 8 x 8
# conv8_2 ==> 4 x 4
# conv9_2 ==> 2 x 2
# conv10_2 ==> 1 x 1

# SSD_Size : 300x300
# conv4_3 ==> 38 x 38
# fc7 ==> 19 x 19
# conv6_2 ==> 10 x 10
# conv7_2 ==> 5 x 5
# conv8_2 ==> 3 x 3
# conv9_2 ==> 1 x 1

if(baseNetwork == "VGG") or (baseNetwork == "VGG_G4"):
    mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
elif(baseNetwork == "JACINTO"):
    mbox_source_layers = ['res3a_branch2b_relu', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
elif(baseNetwork == "JACINTO_V2"):
    if fully_conv_at_end: 
      mbox_source_layers = ['res3a_branch2b_relu', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
    else:  
      mbox_source_layers = ['res3a_branch2b_relu', 'res5a_branch2b_relu', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
elif(baseNetwork == "RES101"):
    # minimum dimension of input image (300x300)
    # res3b3_relu ==> 38 x 38
    # res5c_relu ==> 19 x 19
    # res5c_relu/conv1_2 ==> 10 x 10
    # res5c_relu/conv2_2 ==> 5 x 5
    # res5c_relu/conv3_2 ==> 3 x 3
    mbox_source_layers = ['res3b3_relu', 'res5c_relu', 'res5c_relu/conv1_2', 'res5c_relu/conv2_2', 'res5c_relu/conv3_2']
elif(baseNetwork == "RESTEN"):
    # For dimension of input image (300x300)
    # res3b3_relu ==> 38 x 38
    # res5c_relu ==> 19 x 19
    # res5c_relu/conv1_2 ==> 10 x 10
    # res5c_relu/conv2_2 ==> 5 x 5
    # res5c_relu/conv3_2 ==> 3 x 3

    #for input image (500x500)
    # layer_128_1_sum ==> 64x64x128
    # last_relu       ==> 32x32x256
    mbox_source_layers = ['layer_128_1_sum', 'last_relu', 'last_relu/conv1_2', 'last_relu/conv2_2', 'last_relu/conv3_2']
elif(baseNetwork == "JINET"):
   mbox_source_layers = ['res3a_debot',  'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
elif(baseNetwork == "ENET"):
   mbox_source_layers = ['bot2_2',  'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2',]
elif(baseNetwork == "mobile"):
   if heads_same_as_vgg:
     if depth_mul == 1.0:
       mbox_source_layers = ['head1', 'conv6{}sep{}relu'.format(postfix_char,postfix_char), 'conv{}_2'.format(first_idx), 'conv{}_2'.format(first_idx+1), 'conv{}_2'.format(first_idx+2), 'conv{}_2'.format(first_idx+3)]
     elif depth_mul == 0.5:
       mbox_source_layers = ['head1', 'head2', 'conv{}_2'.format(first_idx), 'conv{}_2'.format(first_idx+1), 'conv{}_2'.format(first_idx+2), 'conv{}_2'.format(first_idx+3)]
   else:
     #mbox_source_layers = ['conv4_1_sep_relu', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
     #mbox_source_layers = ['conv4_1_sep_relu', 'conv6_sep_relu', 'conv{}_2'.format(first_idx), 'conv{}_2'.format(first_idx+1),
     #      'conv{}_2'.format(first_idx+2), 'conv{}_2'.format(first_idx+3)]
     #moved regression head below like done by chuanqi/tensor flow based SSD 
     mbox_source_layers = ['conv5_5_sep_relu', 'conv6_sep_relu', 'conv{}_2'.format(first_idx), 'conv{}_2'.format(first_idx+1),
          'conv{}_2'.format(first_idx+2), 'conv{}_2'.format(first_idx+3)]
   if use_shuffle:
     mbox_source_layers = ['conv5_5_dw', 'conv6_dw', 'conv{}_2'.format(first_idx), 'conv{}_2'.format(first_idx+1),
        'conv{}_2'.format(first_idx+2), 'conv{}_2'.format(first_idx+3)]
  
   #adding regression head at the beginning too
   #mbox_source_layers = ['conv4_1_sep_relu', 'conv5_5_sep_relu', 'conv6_sep_relu', 'conv{}_2'.format(first_idx), 'conv{}_2'.format(first_idx+1),
   #     'conv{}_2'.format(first_idx+2), 'conv{}_2'.format(first_idx+3)]

   if ssd_mobile_chuanqi:
     mbox_source_layers = ['conv11', 'conv13',  'conv{}_2'.format(first_idx), 'conv{}_2'.format(first_idx+1),
         'conv{}_2'.format(first_idx+2), 'conv{}_2'.format(first_idx+3)]
else:
  sys.exit()

# for 500x500 attach one more layer
if (ssd_size == '512x512') and (training_type == 'SSD'):
  if (baseNetwork != 'RESTEN') and (baseNetwork != 'RES101'):
    mbox_source_layers.append('conv{}_2'.format(first_idx+4))


if ssd_mobile_chuanqi:
  step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 1)))
  min_sizes = []
  max_sizes = []

  for ratio in xrange(min_ratio, max_ratio + 1, step):
    min_sizes.append(min_dim * ratio / 100.)
    max_sizes.append(min_dim * min(ratio + step, 100.0) / 100.)
  max_sizes[0] = 0  
else:
  step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
  min_sizes = []
  max_sizes = []

  for ratio in xrange(min_ratio, max_ratio + 1, step):
    min_sizes.append(min_dim * ratio / 100.)
    max_sizes.append(min_dim * (ratio + step) / 100.)

print('step:', step)   
print('min_sizes:', min_sizes)   
print('max_sizes:', max_sizes)   

if ssd_size == '512x512':
  if small_objs:
    min_size_mul = 4 
    max_size_mul = 10
  else:  
    min_size_mul = 7
    max_size_mul = 15 
elif (ssd_size == '300x300') or (ssd_size == '256x256'):
  if small_objs:
    min_size_mul = 7
    max_size_mul = 15 
  else:
    min_size_mul = 10
    max_size_mul = 20

if ssd_mobile_chuanqi == False:
  min_sizes = [min_dim * min_size_mul / 100.] + min_sizes
  max_sizes = [min_dim * max_size_mul / 100.] + max_sizes

print('min_sizes:', min_sizes)   
print('max_sizes:', max_sizes)  

use_dflt_steps = False 
if resize_width != resize_height:   
  use_dflt_steps = True

if ssd_mobile_chuanqi or (baseNetwork == 'mobile'):
  use_dflt_steps = True

steps = []
if use_dflt_steps == False:
  if (ssd_size == '512x512') and (training_type == 'SSD'):
    #steps = [8, 16, 32, 64, 128, 256, 512]
    steps = [min_dim>>6, min_dim>>5, min_dim>>4, min_dim>>3, min_dim>>2, min_dim>>1, min_dim]
    #steps = [resize_width>>6, resize_width>>5, resize_width>>4, resize_width>>3, resize_width>>2, resize_width>>1, resize_width]
  if (ssd_size == '512x512') and (training_type == 'IMGNET'):
    #steps = [16, 32, 64, 128, 256, 512]
    #:SN, not sure about these values
    steps = [min_dim>>5, min_dim>>4, min_dim>>3, min_dim>>2, min_dim>>1, min_dim]
  elif (ssd_size == '300x300') or (ssd_size == '256x256'):
    steps = [8, 16, 32, 64, 100, 300]

print("steps:", steps)

if (wideAR == True):
  #more  aspect ratios
  aspect_ratios = [[2, 3, 4], [2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5]]
elif (ssd_size == '512x512') and (training_type == 'SSD'):    
  # aspect used by original SSD_512x512
  aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
else:    
  # aspect used by original SSD_300x300
  aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
  if ssd_mobile_chuanqi:
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2,3], [2,3]]

#JACINTO, VGG, VGG_G4, RES101,RESTEN, "JINET", "ENET"

normalizations = []
#need it for VGG. Also used for Jacinto by accident. Need to experiment with
#removing it.
if (force_norm == True):
  # L2 normalize conv4_3.
  if ssd_size == '512x512':    
    normalizations = [20, -1, -1, -1, -1, -1, -1]
  elif (ssd_size == '300x300') or (ssd_size == '256x256'):    
    normalizations = [20, -1, -1, -1, -1, -1]

# variance used to encode/decode prior bboxes.
if code_type == P.PriorBox.CENTER_SIZE:
  prior_variance = [0.1, 0.1, 0.2, 0.2]
else:
  prior_variance = [0.1]
flip = True
clip = False

if 'KITTI' in dataset:
  clip = True

# Solver parameters.
# Defining which GPUs to use.
gpus = "0,1" #gpus = "0,1,2,3"
gpulist = gpus.split(",")
num_gpus = len(gpulist)

if non_sq_dataset:
  # surprisingly batch size of only 1 (per GPU) is supported when FIT_SMALL_SIZE (used
  # with non SQ dataset) is used!!!
  batch_size = num_gpus
else:  
  batch_size = 16
accum_batch_size = 32

if (ssd_size == '300x300') or (ssd_size == '256x256'):
  batch_size = 16

if (dataset== "TI2017_SingleImg"):
  gpus = "0" #gpus = "0,1,2,3"
  gpulist = gpus.split(",")
  num_gpus = len(gpulist)
  batch_size = 4 
  accum_batch_size = 4

iter_size = accum_batch_size / batch_size
solver_mode = P.Solver.CPU
device_id = 0
# Divide the mini-batch to different GPUs.
batch_size_per_device = batch_size
if num_gpus > 0:
  batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
  iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
  solver_mode = P.Solver.GPU
  device_id = int(gpulist[0])

if normalization_mode == P.Loss.NONE:
  base_lr /= batch_size_per_device
elif normalization_mode == P.Loss.VALID:
  base_lr *= 25. / loc_weight
elif normalization_mode == P.Loss.FULL:
  # Roughly there are 2000 prior bboxes per image.
  # TODO(weiliu89): Estimate the exact # of priors.
  base_lr *= 2000.

# with CUDNN enabled it does not converge with default learnign rate
#use_cudnn = 1
#if use_cudnn:
#    base_lr /= 4

freeze_layers = []
if freezeInitFewLayers: 
  # Which layers to freeze (no backward) during training.
  if(baseNetwork == "VGG") or (baseNetwork == "VGG_G4"):
    freeze_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2']
  elif(baseNetwork == "JACINTO"):
    #freeze few init layer for small dataset fine tuning 
    #freeze_layers = ['conv1a', 'conv1b', 'res2a_branch2a', 'res2a_branch2b', 'res3a_branch2a', 'res3a_branch2b']
    freeze_layers = ['conv1a', 'conv1b', 'res2a_branch2a', 'res2a_branch2b']
  elif(baseNetwork == "JACINTO_V2"):
    #freeze few init layer for small dataset fine tuning 
    freeze_layers = ['conv1a', 'conv1b', 'res2a_branch2a', 'res2a_branch2b']
  elif(baseNetwork == "JINET"):
    freeze_layers = ['conv1a', 'conv1a_bot', 'conv1b', 'conv1_debot', 'res2a_bot', 'res2a_branch2a', 'res2a_branch2b', 'res2a_debot']
  elif(baseNetwork == "ENET"):
    #freeze_layers = ['mean_sub','init', 'bottleNeck1', 'bottleNeck2']
    #while re-tranining after dil bf
    freeze_layers = ['mean_sub','init', 'bottleNeck1']
  elif(baseNetwork == "mobile"):
    freeze_layers = []

print ("freeze_layers: ", freeze_layers)

# Evaluate on whole test set.
#num_test_image = 4952
test_batch_size = 1
# Ideally test_batch_size should be divisible by num_test_image,
# otherwise mAP will be slightly off the true value.
test_iter = int(math.ceil(float(num_test_image) / test_batch_size))

max_iter = (numEpoch * num_train_image/accum_batch_size)

# make max_iteration multiple of test_interval so that for last 
# iteration test wil be performed
max_iter = ((max_iter + test_interval)/test_interval)*test_interval  
# so that for last iteration snapshot is written 
max_iter = ((max_iter + snapshot_interval)/snapshot_interval)*snapshot_interval  

print ("max_iter: ", max_iter)
#stepsize = max_iter/2
solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': weight_decay,
    'lr_policy': LR_POLICY,
    'power': 1.0,
    'stepvalue': [int(stepvalue_fac[0]*max_iter), int(stepvalue_fac[1]*max_iter), max_iter],
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': iter_size,
    'max_iter': max_iter,
    'snapshot': snapshot_interval,
    'display': 100,
    'average_loss': 10,
    'type': "SGD",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': test_interval,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': True,
    'show_per_class_result':True,
    }

if log_train_accu:
  solver_param['test_iter'] = [test_iter, test_iter]
  solver_param['test_state'] = [{'stage' : ["TEST_LMDB"]}, {'stage' : ["TRAIN_LMDB"]}]

save_output_param = {
        'output_directory': output_result_dir,
        'output_name_prefix': "comp4_det_test_",
        'output_format': "VOC",
        'label_map_file': label_map_file,
        'name_size_file': name_size_file,
        'num_test_image': num_test_image,
        #'save_file': 'saveTrainingVideo.MP4',
        }

if non_sq_dataset:
  save_output_param['resize_param'] = resize_param_test_transform


# parameters for generating detection output.
det_out_param = {
    'num_classes': num_classes,
    'share_location': share_location,
    'background_label_id': background_label_id,
    'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
    'save_output_param': save_output_param,
    'keep_top_k': 200,
    'confidence_threshold': 0.01,
    'code_type': code_type,
}

# parameters for evaluating detection results.
det_eval_param = {
    'num_classes': num_classes,
    'background_label_id': background_label_id,
    'overlap_threshold': 0.5,
    'evaluate_difficult_gt': evaluate_difficult_gt,
    'name_size_file': name_size_file,
          }

if non_sq_dataset:
  det_eval_param['resize_param'] = resize_param_test_transform

#if training accuracy needs to be logged.
if log_train_accu:
  det_train_eval_param = copy.deepcopy(det_eval_param)
  det_train_eval_param['name_size_file'] = train_name_size_file
  
  det_train_out_param = copy.deepcopy(det_out_param)  
  det_train_out_param['save_output_param']['name_size_file'] = train_name_size_file

### Hopefully you don't need to change the following ###
# Check file.
check_if_exist(train_data)
check_if_exist(test_data)
if log_train_accu:
  check_if_exist(train_accu_data)
check_if_exist(label_map_file)
check_if_exist(pretrain_model)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

# Create train net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,
        train=True, output_label=True, label_map_file=label_map_file,
        transform_param=train_transform_param, batch_sampler=batch_sampler)

if(baseNetwork == "VGG"):
    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers)
elif(baseNetwork == "JACINTO"):
    JacintoNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers, bn_type=bn_type,
        caffe_fork=caffe_fork, training_type=training_type)
elif(baseNetwork == "JACINTO_V2"):
    JacintoNetV2Body(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers, bn_type=bn_type,
        caffe_fork=caffe_fork, training_type=training_type, fully_conv_at_end=fully_conv_at_end)
elif(baseNetwork == "RES101"):
    ResNet101Body(net, from_layer='data', use_pool5=False, use_dilation_conv5=True)
elif(baseNetwork == "RESTEN"):
    ResNetTenBody(net, from_layer='data', use_pool5=False, use_dilation_conv5=True)
elif(baseNetwork == "JINET"):
    JINetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers)
elif(baseNetwork == "ENET"):
    ENetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers, bn_type=bn_type, 
        enable_dilation=enable_dilation_enet, ex_net=ex_net,
        bn_at_start=bn_at_start, caffe_fork=caffe_fork,
        training_type = training_type, dil_bf=dil_bf)
elif(baseNetwork == "mobile"):
    MobileNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers, bn_type=bn_type, 
        bn_at_start=bn_at_start, caffe_fork=caffe_fork,
        training_type = training_type, depth_mul=depth_mul, ssd_mobile_chuanqi=ssd_mobile_chuanqi,
        dil_when_stride_removed=dil_when_stride_removed,
        use_shuffle=use_shuffle, heads_same_as_vgg=heads_same_as_vgg,
        postfix_char=postfix_char)
else:
    print ("wrong base network selected")
    sys.exit()

AddExtraLayers(net, use_batchnorm, lr_mult=lr_mult, baseNetwork=baseNetwork,
    ssd_size=ssd_size, training_type=training_type)

if non_sq_dataset:
  img_height_temp=resize_height
  img_width_temp=resize_width
else:
  # for SQ images this was set to zero
  img_height_temp=0
  img_width_temp=0

kernel_size_mbox = 3
pad_mbox = 1
if ssd_mobile_chuanqi:
  kernel_size_mbox = 1    
  pad_mbox = 0

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
      use_batchnorm=use_batchnorm_mbox, min_sizes=min_sizes, max_sizes=max_sizes,
      aspect_ratios=aspect_ratios, steps=steps, img_height=img_height_temp,
      img_width=img_width_temp, normalizations=normalizations,
      num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
      prior_variance=prior_variance, kernel_size=kernel_size_mbox, pad=pad_mbox, lr_mult=lr_mult, conf_name=conf_name,
      ssd_mobile_chuanqi=ssd_mobile_chuanqi)

# Create the MultiBoxLossLayer.
name = "mbox_loss"
mbox_layers.append(net.label)
net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
        loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
        propagate_down=[True, True, False, False])

with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)
shutil.copy(train_net_file, job_dir)

# Create test net.
net = caffe.NetSpec()
if log_train_accu == False:
  net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
        train=False, output_label=True, label_map_file=label_map_file,
        transform_param=test_transform_param)
else:
  net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
        train=False, output_label=True, label_map_file=label_map_file,
        transform_param=test_transform_param, stage='TEST_LMDB')
  saved_net_to_proto = net.to_proto()
  net.data, net.label = CreateAnnotatedDataLayer(train_accu_data, batch_size=test_batch_size,
          train=False, output_label=True, label_map_file=label_map_file,
          transform_param=test_transform_param, stage='TRAIN_LMDB')

if(baseNetwork == "VGG"):
    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers)
elif(baseNetwork == "JACINTO"):
    JacintoNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers, bn_type=bn_type, caffe_fork=caffe_fork, 
        training_type=training_type)
elif(baseNetwork == "JACINTO_V2"):
    JacintoNetV2Body(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers, bn_type=bn_type,
        caffe_fork=caffe_fork, training_type=training_type, fully_conv_at_end=fully_conv_at_end)
elif(baseNetwork == "RES101"):
    ResNet101Body(net, from_layer='data', use_pool5=False, use_dilation_conv5=True)
elif(baseNetwork == "RESTEN"):
    ResNetTenBody(net, from_layer='data', use_pool5=False, use_dilation_conv5=True)
elif(baseNetwork == "JINET"):
    JINetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers)
elif(baseNetwork == "ENET"):
    ENetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers, bn_type=bn_type, 
        enable_dilation=enable_dilation_enet, ex_net=ex_net, 
        bn_at_start=bn_at_start, caffe_fork=caffe_fork, 
        training_type = training_type, dil_bf=dil_bf)
elif(baseNetwork == "mobile"):
    MobileNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers, bn_type=bn_type, 
        bn_at_start=bn_at_start, caffe_fork=caffe_fork,
        training_type = training_type, depth_mul=depth_mul, ssd_mobile_chuanqi=ssd_mobile_chuanqi,
        dil_when_stride_removed=dil_when_stride_removed,
        use_shuffle=use_shuffle, heads_same_as_vgg=heads_same_as_vgg,
        postfix_char=postfix_char)
else:
    print ("wrong base network selected")
    sys.exit()

# Use batch norm for the newly added layers.
AddExtraLayers(net, use_batchnorm, lr_mult=lr_mult, baseNetwork=baseNetwork, ssd_size=ssd_size, 
        training_type=training_type)

if non_sq_dataset:
  img_height_temp=resize_height
  img_width_temp=resize_width
else:
  # for SQ images this was set to zero
  img_height_temp=0
  img_width_temp=0

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
      use_batchnorm=use_batchnorm_mbox, min_sizes=min_sizes, max_sizes=max_sizes,
      aspect_ratios=aspect_ratios, steps=steps, img_height=img_height_temp,
      img_width=img_width_temp, normalizations=normalizations,
      num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
      prior_variance=prior_variance, kernel_size=kernel_size_mbox , pad=pad_mbox, lr_mult=lr_mult, conf_name=conf_name, 
      ssd_mobile_chuanqi=ssd_mobile_chuanqi)

if multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX:
  reshape_name = "{}_reshape".format(conf_name)
  net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
  softmax_name = "{}_softmax".format(conf_name)
  net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
  flatten_name = "{}_flatten".format(conf_name)
  net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
  mbox_layers[1] = net[flatten_name]
elif multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
  sigmoid_name = "{}_sigmoid".format(conf_name)
  net[sigmoid_name] = L.Sigmoid(net[conf_name])
  mbox_layers[1] = net[sigmoid_name]


if log_train_accu:
  net.detection_out = L.DetectionOutput(*mbox_layers,
      detection_output_param=det_out_param,
      include=dict(phase=caffe_pb2.Phase.Value('TEST'), stage='TEST_LMDB'))
  net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
      detection_evaluate_param=det_eval_param,
      include=dict(phase=caffe_pb2.Phase.Value('TEST'), stage='TEST_LMDB'))

  net.detection_out_train_accu = L.DetectionOutput(*mbox_layers,
      detection_output_param=det_train_out_param,
      include=dict(phase=caffe_pb2.Phase.Value('TEST'), stage='TRAIN_LMDB'))
  net.detection_eval_train_accu = L.DetectionEvaluate(net.detection_out_train_accu, net.label,
      detection_evaluate_param=det_train_eval_param,
      include=dict(phase=caffe_pb2.Phase.Value('TEST'), stage='TRAIN_LMDB'))

else:  
  net.detection_out = L.DetectionOutput(*mbox_layers,
      detection_output_param=det_out_param,
      include=dict(phase=caffe_pb2.Phase.Value('TEST')))
  net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
      detection_evaluate_param=det_eval_param,
      include=dict(phase=caffe_pb2.Phase.Value('TEST')))

with open(test_net_file, 'w') as f:
    print('name: "{}_test"'.format(model_name), file=f)
    if log_train_accu:
      print(saved_net_to_proto, file=f)
    print(net.to_proto(), file=f)
shutil.copy(test_net_file, job_dir)

# Create deploy net.
# Remove the first and last layer from test net.
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]
    del net_param.layer[-1]
    if log_train_accu:
      del net_param.layer[-2]
      del net_param.layer[-1]

    net_param.name = '{}_deploy'.format(model_name)
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
    print(net_param, file=f)
shutil.copy(deploy_net_file, job_dir)

# Create solver.
if log_train_accu:
  solver = caffe_pb2.SolverParameter(
          train_net=train_net_file,
          test_net=[test_net_file, test_net_file],
          snapshot_prefix=snapshot_prefix,
          **solver_param)
else:
  solver = caffe_pb2.SolverParameter(
          train_net=train_net_file,
          test_net=[test_net_file],
          snapshot_prefix=snapshot_prefix,
          **solver_param)

with open(solver_file, 'w') as f:
    print(solver, file=f)
shutil.copy(solver_file, job_dir)

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(snapshot_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter

train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

if remove_old_models:
  # Remove any snapshots smaller than max_iter.
  for file in os.listdir(snapshot_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))

# Create job file.
with open(job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  f.write('./build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(solver_file))
  f.write(train_src_param)
  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee -a {}/{}.log\n'.format(gpus, job_dir, model_name))
  else:
    f.write('2>&1 | tee -a {}/{}.log\n'.format(job_dir, model_name))

# Copy the python script to job_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, job_dir)

# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.call(job_file, shell=True)
