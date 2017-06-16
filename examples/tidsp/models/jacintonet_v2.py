from __future__ import print_function
import caffe
from models.model_libs import *

def jacintonet11(net, from_layer=None, use_batchnorm=True, use_relu=True, num_output=1000, stride_list=None, dilation_list=None, freeze_layers=None):  
   in_place = False #Top and Bottom blobs must be different for NVCaffe BN caffe-0.15
   if stride_list == None:
     stride_list = [2,2,2,2,2]
   if dilation_list == None:
     dilation_list = [1,1,1,1,1]
   
   #--     
   stage=0
   stride=stride_list[stage]
   dilation=dilation_list[stage]   
   
   out_layer = 'conv1a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=32, kernel_size=[5,5], pad=2*dilation, stride=stride, group=1, dilation=dilation, in_place=in_place)  
   
   from_layer = out_layer
   out_layer = 'conv1b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=32, kernel_size=[3,3], pad=dilation, stride=1, group=4, dilation=dilation, in_place=in_place)       
   
   #--     
   stage=1
   stride=stride_list[stage]
   dilation=dilation_list[stage]   
        
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':stride, 'stride':stride}	 
   from_layer = out_layer
   out_layer = 'pool1'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)       
      
   from_layer = out_layer
   out_layer = 'res2a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=dilation, stride=1, group=1, dilation=dilation, in_place=in_place)   
   
   from_layer = out_layer
   out_layer = 'res2a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=dilation, stride=1, group=4, dilation=dilation, in_place=in_place)     
     
   #--     
   stage=2
   stride=stride_list[stage]
   dilation=dilation_list[stage] 
        
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':stride, 'stride':stride}       
   from_layer = out_layer
   out_layer = 'pool2'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)     
         
   from_layer = out_layer
   out_layer = 'res3a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=128, kernel_size=[3,3], pad=dilation, stride=1, group=1, dilation=dilation, in_place=in_place)   
   
   from_layer = out_layer
   out_layer = 'res3a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=128, kernel_size=[3,3], pad=dilation, stride=1, group=4, dilation=dilation, in_place=in_place)    
   
   #--       
   stage=3
   stride=stride_list[stage]
   dilation=dilation_list[stage] 
      
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':stride, 'stride':stride}   
   from_layer = out_layer
   out_layer = 'pool3'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)         
         
   from_layer = out_layer
   out_layer = 'res4a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=256, kernel_size=[3,3], pad=dilation, stride=1, group=1, dilation=dilation, in_place=in_place)   
   
   from_layer = out_layer 
   out_layer = 'res4a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=256, kernel_size=[3,3], pad=dilation, stride=1, group=4, dilation=dilation, in_place=in_place)        
      
   #--      
   stage=4
   stride=stride_list[stage]
   dilation=dilation_list[stage]         
   
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':stride, 'stride':stride}   
   from_layer = out_layer
   out_layer = 'pool4'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)     

   from_layer = out_layer
   out_layer = 'res5a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=512, kernel_size=[3,3], pad=dilation, stride=1, group=1, dilation=dilation, in_place=in_place)   
   
   from_layer = out_layer
   out_layer = 'res5a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=512, kernel_size=[3,3], pad=dilation, stride=1, group=4, dilation=dilation, in_place=in_place) 
   
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
   
   
def jsegnet21(net, from_layer=None, use_batchnorm=True, use_relu=True, num_output=20, stride_list=None, dilation_list=None, freeze_layers=None): 
   in_place = False #Top and Bottom blobs must be different for NVCaffe BN caffe-0.15
   if stride_list == None:
     stride_list = [2,2,2,2,1]
   if dilation_list == None:
     dilation_list = [1,1,1,1,2]

   #--
   stage=0
   stride=stride_list[stage]
   dilation=dilation_list[stage]
      	    
   out_layer = 'conv1a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=32, kernel_size=[5,5], pad=2*dilation, stride=stride, group=1, dilation=dilation, in_place=in_place)  
	    
   from_layer = out_layer
   out_layer = 'conv1b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=32, kernel_size=[3,3], pad=dilation, stride=1, group=4, dilation=dilation, in_place=in_place)       
   
   #--	    	 	 
   stage=1
   stride=stride_list[stage]
   dilation=dilation_list[stage] 
      	 	 
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':stride, 'stride':stride}      
   from_layer = out_layer
   out_layer = 'pool1'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)    
   
   from_layer = out_layer
   out_layer = 'res2a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=dilation, stride=1, group=1, dilation=dilation, in_place=in_place)   
   
   from_layer = out_layer
   out_layer = 'res2a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=dilation, stride=1, group=4, dilation=dilation, in_place=in_place)     
   	 
   #--   
   stage=2
   stride=stride_list[stage]
   dilation=dilation_list[stage] 
      	 
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':stride, 'stride':stride}      
   from_layer = out_layer
   out_layer = 'pool2'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)    

   from_layer = out_layer
   out_layer = 'res3a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=128, kernel_size=[3,3], pad=dilation, stride=1, group=1, dilation=dilation, in_place=in_place)   
   
   from_layer = out_layer
   out_layer = 'res3a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=128, kernel_size=[3,3], pad=dilation, stride=1, group=4, dilation=dilation, in_place=in_place)    
   
   #--   
   stage=3
   stride=stride_list[stage]
   dilation=dilation_list[stage] 
      
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':stride, 'stride':stride}      
   from_layer = out_layer
   out_layer = 'pool3'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)  
 
   from_layer = out_layer
   out_layer = 'res4a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=256, kernel_size=[3,3], pad=dilation, stride=1, group=1, dilation=dilation, in_place=in_place)   
   
   from_layer = out_layer 
   out_layer = 'res4a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=256, kernel_size=[3,3], pad=dilation, stride=1, group=4, dilation=dilation, in_place=in_place)        
   
   #--   
   #by default, removes this pooling and dilates the subsequent layers
   stage=4
   stride=stride_list[stage]
   dilation=dilation_list[stage]    
   
   pooling_param = {'pool':P.Pooling.MAX, 'kernel_size':stride, 'stride':stride}      
   from_layer = out_layer
   out_layer = 'pool4'
   net[out_layer] = L.Pooling(net[from_layer], pooling_param=pooling_param)     
      
   from_layer = out_layer
   out_layer = 'res5a_branch2a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=512, kernel_size=[3,3], pad=dilation, stride=1, group=1, dilation=dilation, in_place=in_place)   
   
   from_layer = out_layer
   out_layer = 'res5a_branch2b'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=512, kernel_size=[3,3], pad=dilation, stride=1, group=4, dilation=dilation, in_place=in_place) 
   
   #--   
   from_layer = out_layer
   out_layer = 'out5a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=4, stride=1, group=2, dilation=4, in_place=in_place) 
   
   #frozen upsampling layer
   from_layer = out_layer 
   out_layer = 'out5a_up2'  
   deconv_kwargs = {  'param': { 'lr_mult': 0, 'decay_mult': 0 },
       'convolution_param': { 'num_output': 64, 'bias_term': False, 'pad': 1, 'kernel_size': 4, 'group': 64, 'stride': 2, 
       'weight_filler': { 'type': 'bilinear' } } }
   net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs)    
   
   from_layer = 'res3a_branch2b' if in_place else 'res3a_branch2b/bn'
   out_layer = 'out3a'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=1, stride=1, group=2, dilation=1, in_place=in_place) 
   
   from_layer = out_layer   
   out_layer = 'out3_out5_combined'
   net[out_layer] = L.Eltwise(net['out5a_up2'], net[from_layer])
   
   from_layer = out_layer
   out_layer = 'ctx_conv1'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=1, stride=1, group=1, dilation=1, in_place=in_place) 
      
   from_layer = out_layer
   out_layer = 'ctx_conv2'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=4, stride=1, group=1, dilation=4, in_place=in_place) 
   
   from_layer = out_layer
   out_layer = 'ctx_conv3'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=4, stride=1, group=1, dilation=4, in_place=in_place) 
   
   from_layer = out_layer
   out_layer = 'ctx_conv4'
   out_layer = ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=64, kernel_size=[3,3], pad=4, stride=1, group=1, dilation=4, in_place=in_place) 
       
   from_layer = out_layer
   out_layer = 'ctx_final'
   conv_kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='msra'),
        'bias_term': True, 
        'bias_filler': dict(type='constant', value=0) }   
   net[out_layer] = L.Convolution(net[from_layer], num_output=num_output, kernel_size=[3,3], pad=1, stride=1, group=1, dilation=1, **conv_kwargs) 
         
   from_layer = out_layer
   out_layer = 'ctx_final/relu'
   net[out_layer] = L.ReLU(net[from_layer], in_place=True) 
               
   #frozen upsampling layer         
   from_layer = out_layer
   out_layer = 'out_deconv_final_up2'   
   deconv_kwargs = {  'param': { 'lr_mult': 0, 'decay_mult': 0 },
       'convolution_param': { 'num_output': num_output, 'bias_term': False, 'pad': 1, 'kernel_size': 4, 'group': num_output, 'stride': 2, 
       'weight_filler': { 'type': 'bilinear' } } }       
   net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs)   
           
   from_layer = out_layer
   out_layer = 'out_deconv_final_up4'       
   net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs) 
   
   from_layer = out_layer
   out_layer = 'out_deconv_final_up8'       
   net[out_layer] = L.Deconvolution(net[from_layer], **deconv_kwargs)     
                            
   return out_layer    
