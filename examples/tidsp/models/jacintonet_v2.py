from __future__ import print_function
import caffe
from caffe.model_libs import *

def jacintonet11(net, from_layer=None, use_batchnorm=True, use_relu=True, num_output=1000, freeze_layers=[]):   
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':2, 'stride':2}
   in_place = False #Top and Bottom blobs must be different for NVCaffe BN caffe-0.15
   
   out_layer = 'conv1a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=32, kernel_size=[5,5], pad=2, stride=2, group=1, in_place=in_place)  
   
   from_layer = out_layer
   out_layer = 'conv1b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=32, kernel_size=[3,3], pad=1, stride=1, group=4, in_place=in_place)       
   
   from_layer = out_layer
   out_layer = 'pool1'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)    
   #--
   
   from_layer = out_layer
   out_layer = 'res2a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=1, stride=1, group=1, in_place=in_place)   
   
   from_layer = out_layer
   out_layer = 'res2a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=1, stride=1, group=4, in_place=in_place)     
   
   from_layer = out_layer
   out_layer = 'pool2'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)    
   #--
      
   from_layer = out_layer
   out_layer = 'res3a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=128, kernel_size=[3,3], pad=1, stride=1, group=1, in_place=in_place)   
   
   from_layer = out_layer
   out_layer = 'res3a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=128, kernel_size=[3,3], pad=1, stride=1, group=4, in_place=in_place)    
   
   from_layer = out_layer
   out_layer = 'pool3'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)      
   #--
      
   from_layer = out_layer
   out_layer = 'res4a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=256, kernel_size=[3,3], pad=1, stride=1, group=1, in_place=in_place)   
   
   from_layer = out_layer 
   out_layer = 'res4a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=256, kernel_size=[3,3], pad=1, stride=1, group=4, in_place=in_place)        
   
   from_layer = out_layer
   out_layer = 'pool4'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)    
   #--
      
   from_layer = out_layer
   out_layer = 'res5a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=512, kernel_size=[3,3], pad=1, stride=1, group=1, in_place=in_place)   
   
   from_layer = out_layer
   out_layer = 'res5a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=512, kernel_size=[3,3], pad=1, stride=1, group=4, in_place=in_place) 
   #--   
   
   # Add global pooling layer.
   from_layer = out_layer
   out_layer = 'pool5'
   net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.AVE, global_pooling=True)
       
   from_layer = out_layer 
   out_layer = 'fc'+str(num_output)
   kwargs = { 'num_output': num_output, 
     'param': [{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}], 
     'inner_product_param': { 
         'weight_filler': { 'type': 'msra' }, 
         'bias_filler': { 'type': 'constant', 'value': 0 }   
     },
   }
   net[out_layer] = L.InnerProduct(net[from_layer], **kwargs)    
   
   return out_layer
   
   
    
