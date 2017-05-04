#!/usr/bin/env python

import os
import os.path
import time
import sys
import matplotlib
import matplotlib.pyplot as plt
import ntpath

sys.path.insert(0, '/user/a0393754/work/caffe/caffe-jacinto/python')
model_path = '/user/a0393754/work/cfar10_jnet/deploy_final_sparse_quant_jacintonet11_nobn_iter_32000.prototxt'
pretrained_path = '/user/a0393754/work/cfar10_jnet/final_sparse_quant_jacintonet11_nobn_iter_32000.caffemodel'
input_name = '/user/a0393754/work/cfar10_jnet/dog4_128x128.png'

import caffe
caffe.set_mode_cpu()
from caffe.proto import caffe_pb2
import cv2
import numpy as np
import math
import string
from google.protobuf import text_format

def writeNPAryAsRaw(ipFrame, fileName, opDataType=np.float32, opQ=0):
    if opDataType != np.float32:
        opMult = 1<<opQ
        qFrame = np.rint(ipFrame * opMult)
    else:
        qFrame = ipFrame
            
    fileHandle = open(fileName, 'wb')
    ip1DAry = np.reshape(qFrame, (1, np.prod(qFrame.shape)))
    ip1DAry = ip1DAry.astype(opDataType)
    fileHandle.write(ip1DAry)
    fileHandle.close()
       
def predict(model_path, pretrained_path, image, frameNum, blobs=None):
    net = caffe.Net(model_path, pretrained_path, caffe.TEST)
    #model = type('', (), {})()
    #model.net = net

    input_dims = net.blobs['data'].shape
    #output_dims = net.blobs['prob'].shape

    #label_margin = config.MARGIN
    print ("input dim from desc", input_dims[2], input_dims[3])
    #print ("output_dim from desc", output_dims[2], output_dims[3])

    batch_size, num_channels, input_height, input_width = input_dims
    caffe_in = np.zeros(input_dims, dtype=np.float32)

    caffe_in[0] = image.transpose([2, 0, 1])
    out_blobs = net.forward_all(blobs, **{net.inputs[0]: caffe_in})
    
    return out_blobs, net

def getLayerByName(net_proto, layer_name):
    for layer in net_proto.layer:
       if layer.name == layer_name:
          return layer
    return None
    
def infer():
    caffe.set_mode_cpu()
    mean_pixel = [0, 0, 0]
    num = 0

    net_proto = caffe_pb2.NetParameter()
    text_format.Merge(open(model_path).read(), net_proto)
    
    # moved image reading out from predict()
    image = cv2.imread(input_name, 1).astype(np.float32) - mean_pixel
    layer_names=['data', 'conv1a_relu', 'fc10', 'prob']    
    blob_names=['data', 'conv1a', 'fc10', 'prob']
    out_blobs, net = predict(model_path, pretrained_path, image, num, blobs=blob_names)
    
    print (out_blobs['prob'])   
       
    if 'data' in out_blobs.keys():
        writeNPAryAsRaw(out_blobs['data'], 'data'+'_uint8'+'.bin', opDataType=np.uint8, opQ=8)       
          
    for blobName in out_blobs.keys():
       layerIndex = blob_names.index(blobName)
       layerName = layer_names[layerIndex]
       print layerName, blobName
       layerParam =  getLayerByName(net_proto, layerName)  
       if layerParam:  
           layer = net.layers[list(net._layer_names).index(blobName)]
           if layerParam.quantization_param.quantize_layer_out:       
               if layerParam.quantization_param.unsigned_layer_out:   
                   writeNPAryAsRaw(out_blobs[blobName], blobName+'_uint8'+'.bin', opDataType=np.uint8, opQ=layerParam.quantization_param.fl_layer_out)
               else:
                   writeNPAryAsRaw(out_blobs[blobName], blobName+'_int8'+'.bin', opDataType=np.int8, opQ=layerParam.quantization_param.fl_layer_out)       
           else:
               writeNPAryAsRaw(out_blobs[blobName], blobName+'_float32'+'.bin', opDataType=np.float32)  
           
def main():
    infer()

if __name__ == '__main__':
    main()
