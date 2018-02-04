import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

#########################################################
# Get blob shape from bottom blob
def getBotBlobShape():
  #Get IpW and IpH from bottom blob (input) shape
  bot_layer_idx = net._bottom_ids(layer_idx)[0]
  #print("bot_layer_idx: ", bot_layer_idx)
  bot_blob_name = net._blob_names[bot_layer_idx]
  #print("bot_blob_name: ", bot_blob_name)
  bot_blob_shape = net.blobs[bot_blob_name].data.shape
  #print("bot_blob_shape: ", bot_blob_shape)
  
  if(len(bot_blob_shape) == 4):
    Ni = bot_blob_shape[1]
    IpW = bot_blob_shape[2]
    IpH = bot_blob_shape[3]
  elif (len(bot_blob_shape) == 2):  
    #if prev layer was flatten layer
    assert (bot_blob_shape[0] == 1), "assumption that prev layer was flatten had blob_shape 1xN is wrong" 
    Ni = bot_blob_shape[1]
    IpW = 1
    IpH = 1

  return [Ni,IpW,IpH] 

#########################################################

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples

import sys
import os
sys.path.insert(0, caffe_root + 'python')

import caffe

# configure plotting
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Load the net, list its data and params, and filter an example image.
caffe.set_mode_cpu()
print ("Afr  setting cpu")

#model = '/user/a0875091/files/work/github/weiliu89/caffe-ssd/examples/imagenet/training_ENET_bvlc_dil_0_Top1_63.8/FineTuneFromNVIDIA_BN_Top1_63.8_Top5_84.9/CleanedUpNames/ENet_deploy.prototxt'
#model = '/user/a0875091/files/work/github/weiliu89/caffe-ssd/examples/imagenet/Xception-caffe/deploy.prototxt'
#model = '/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/bvlc_googlenet/deploy.prototxt'
#weights = '/user/a0875091/files/work/github/weiliu89/caffe-ssd/examples/imagenet/training_ENET_bvlc_dil_0_Top1_63.8/FineTuneFromNVIDIA_BN_Top1_63.8_Top5_84.9/ENet_NVIDIA_BN_iter_740000.caffemodel'
#model = '/user/a0875091/files/work/bitbucket_TI/caffe-nvidia-16/models/bvlc_googlenet/deploy.prototxt'
#model = '/user/a0875091/files/work/github/zynqnet/zynqnet cnn/prototxt models/inceptionv4_deploy.prototxt'
#model = '/user/a0875091/files/work/github/zynqnet/zynqnet cnn/prototxt models/inceptionv3_deploy.prototxt'
model = '/user/a0875091/files/work/github/weiliu89/caffe-ssd/examples/imagenet/JacintoNetV2/Pre-Trained/deploy.prototxt'

net = caffe.Net(model, caffe.TEST)
print ("Afr  loading net")

print("blobs {}\nparams {}".format(net.blobs.keys()[0:], net.params.keys()[0:]))

print("layers {}\n".format(net.layers[0]))
#print ("net.params['fc1000'][0].data.shape: ", net.params['fc1000'][0].data.shape)
#print ("net.params['fc1000'][1].data.shape: ", net.params['fc1000'][1].data.shape)

print("==================================")
numBlobKeys = len(net.blobs.keys())
for idx,blob_key in zip(range(0,numBlobKeys),net.blobs.keys()):
#for blob_key in net.blobs.keys():
  print("idx:blob_key ", idx, blob_key)

print("==================================")
print ("num layers: ", len(net.layers))
for idx,layer in zip(range(0,len(net.layers)),net.layers):
  print("idx:layer.type: ", idx, layer.type)

print("==================================")

nOps = 0
ipVolume = 0 
opVolume = 0
numParams = 0

#following layers are not used for any computation
#FIX_ME:Need to add compute needed for LRN, Flatten
ignored_layers = ["Slice", "Concat", "Softmax", "Input", "ReLU", "Dropout", "Split", "BatchNorm", "Scale", "LRN", "Flatten"]

print("layer_name,layer_type,Ni,No,Kw,Kh,Ipw,IpH,Opw,OpH,nOpsThisLayer, ipVolThisLayer,opVolThisLayer, numParams, comp2comm")

for layer,layer_name,layer_idx in zip(net.layers,net._layer_names,range(0,len(net.layers))):
  print_layer = False

  #init 
  No = Ni = Kw = Kh = OpW = OpH = IpW = IpH = nOpsThisLayer = -1

  if(layer.type == "Convolution"):
    print_layer = True
    blob_shape = net.blobs[layer_name].data.shape
    No = layer.blobs[0].num
    Ni = layer.blobs[0].channels
    Kw = layer.blobs[0].width 
    Kh = layer.blobs[0].height 

    #Ni*No*Kw*Kh*OpW*OpH
    OpW = blob_shape[2]
    OpH = blob_shape[3]

    nOpsThisLayer = Ni * No * Kw * Kh * OpW * OpH  
     
    [Ni,IpW,IpH] = getBotBlobShape()

    #compute ip, op volume
    ipVolThisLayer = IpW*IpH*Ni 
    opVolThisLayer = OpW*OpH*No
   
    #accumulate only params for weights
    param_idx_wt = 0
    param_idx_bias = 1
    numParamsThisLayer = 1
    for param in net.params[layer_name][param_idx_wt].data.shape:
      numParamsThisLayer = numParamsThisLayer * param 

  elif (layer.type == "Eltwise"):
    print_layer = True
    Kw = 1
    Kh = 1
    blob_shape = net.blobs[layer_name].data.shape

    #No = layer.blobs[0].num
    #Ni = layer.blobs[0].channels
    #Kw = layer.blobs[0].width 
    #Kh = layer.blobs[0].height 
    No = blob_shape[1]
    OpW = blob_shape[2]
    OpH = blob_shape[3]
    N_BLOB_GETTING_ADDED = 2
    nOpsThisLayer = No * OpW * OpH * N_BLOB_GETTING_ADDED

    [Ni,IpW,IpH] = getBotBlobShape()
    #Ni = bot_blob_shape[1]
    #IpW = bot_blob_shape[2]
    #IpH = bot_blob_shape[3]
    
    #compute ip, op volume
    ipVolThisLayer = IpW*IpH*Ni*N_BLOB_GETTING_ADDED
    opVolThisLayer = OpW*OpH*No

    numParamsThisLayer = 0
  elif (layer.type == "Pooling"):
    print_layer = True
    blob_shape = net.blobs[layer_name].data.shape

    #No = layer.blobs[0].num
    #Ni = layer.blobs[0].channels
    #Kw = layer.blobs[0].width 
    #Kh = layer.blobs[0].height 
    #FIX_ME:SN, remove hard coding
    Kw = 2
    Kh = 2
    No = blob_shape[1]
    OpW = blob_shape[2]
    OpH = blob_shape[3]

    [Ni,IpW,IpH] = getBotBlobShape()

    # for pool assuming for kerneal size of pxp, p^2 operations
    nOpsThisLayer = No * IpW * IpH
    #compute ip, op volume
    ipVolThisLayer = IpW*IpH*Ni 
    opVolThisLayer = OpW*OpH*No

    numParamsThisLayer = 0

  elif(layer.type == "InnerProduct"):
    print_layer = True
    blob_shape = net.blobs[layer_name].data.shape
    No = layer.blobs[0].num
    Ni = layer.blobs[0].channels
    Kw = layer.blobs[0].width 
    Kh = layer.blobs[0].height 

    OpW = blob_shape[0]
    OpH = blob_shape[1]
    
    nOpsThisLayer = Ni * No

    [Ni,IpW,IpH] = getBotBlobShape()
    
    #compute ip, op volume
    ipVolThisLayer = IpW*IpH*Ni 
    opVolThisLayer = No

    numParamsThisLayer = Ni*No

  elif layer.type in ignored_layers:
    nOpsThisLayer = 0
    ipVolThisLayer = 0 
    opVolThisLayer = 0
    numParamsThisLayer = 0
  else: 
    print("layer.type : ", layer.type)
    assert (False),"Layer not handled !"

  # parse input image dimension
  if(layer.type == "Input"):
    blob_shape = net.blobs[layer_name].data.shape
    input_ch = blob_shape[1]
    input_w = blob_shape[2]
    input_h = blob_shape[3]
    input_image_vol = input_ch * input_w * input_h 

  if print_layer == True:
    comp2commThisLayer = float(nOpsThisLayer) /(opVolThisLayer + ipVolThisLayer+numParamsThisLayer)  
    print layer_name,layer.type,Ni,No,Kw,Kh,IpW,IpH,OpW,OpH,nOpsThisLayer, ipVolThisLayer,opVolThisLayer, numParamsThisLayer, comp2commThisLayer
    assert (Ni >= 0),"Unintialized value !"
    assert (No >= 0),"Unintialized value !"
    assert (IpH >= 0),"Unintialized value !"
    assert (IpW >= 0),"Unintialized value !"
    assert (OpH >= 0),"Unintialized value !"
    assert (OpW >= 0),"Unintialized value !"
    assert (Kh >= 0),"Unintialized value !"
    assert (Kw >= 0),"Unintialized value !"
    assert (nOpsThisLayer >= 0),"Unintialized value !"

  #accumulate global stat
  nOps = nOps + nOpsThisLayer 
  ipVolume = ipVolume + ipVolThisLayer 
  opVolume = opVolume + opVolThisLayer
  numParams = numParams + numParamsThisLayer 

print("=================")
print("nOps in million: ", nOps/1000000.0)
print("ip Volume in MB: ", ipVolume/1000000.0)
print("op Volume in MB: ", opVolume/1000000.0)
print("ip expansion ratio: ", (float(ipVolume)/input_image_vol))
print("op expansion ratio: ", (float(opVolume)/input_image_vol))
print("numParams million: ", numParams/1000000.0)

