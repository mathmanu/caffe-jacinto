import os

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import copy

def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def UnpackVariable(var, num):
  assert len > 0
  if type(var) is list and len(var) == num:
    return var
  else:
    ret = []
    if type(var) is list:
      assert len(var) == 1
      for i in xrange(0, num):
        ret.append(var[0])
    else:
      for i in xrange(0, num):
        ret.append(var)
    return ret

def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
    use_conv=True,bn_in_place=True, bias_term_swap=False, scale_filler=True, 
    out_layer_name_ResTen=False, bias_decay_mul_in_conv=0,
    **bn_params):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    bias_term = False

    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=lr_mult, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': bias_term,
        }
    eps = bn_params.get('eps', 0.001)
    moving_average_fraction = bn_params.get('moving_average_fraction', 0.999)
    use_global_stats = bn_params.get('use_global_stats', False)

    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        'moving_average_fraction': moving_average_fraction,
        }
    bn_lr_mult = lr_mult
    if use_global_stats:
      # only specify if use_global_stats is explicitly provided;
      # otherwise, use_global_stats_ = this->phase_ == TEST;
      bn_kwargs = {
          'param': [
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0)],
          'eps': eps,
          'use_global_stats': use_global_stats,
          }
      # not updating scale/bias parameters
      bn_lr_mult = 0
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      if scale_filler:
        sb_kwargs = {
          'bias_term': True,
          'param': [
              dict(lr_mult=bn_lr_mult, decay_mult=0),
              dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }

      else:
        #RESTEN does not use filler in scale layers
        sb_kwargs = {
            'bias_term': True,
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2,decay_mult=1)],
            }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    bias_term = True


  if bias_term_swap:
    bias_term = 1-bias_term

  if bias_term:
    kwargs = {
        'param': [dict(lr_mult=lr_mult, decay_mult=1), dict(lr_mult=2 * lr_mult, decay_mult=bias_decay_mul_in_conv)],
        'weight_filler': dict(type='msra'),
        'bias_filler': dict(type='constant', value=0)
        }
  else:  
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='msra'),
        'bias_term': False,
        }

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if use_conv:
      if kernel_h == kernel_w:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
            kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
      else:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
            kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
            stride_h=stride_h, stride_w=stride_w, **kwargs)
  else:
      conv_name=from_layer


  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})

  out_layer_name = conv_name
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=bn_in_place, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
      out_layer_name = sb_name
      print("sb_name:" , sb_name)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
      out_layer_name = bias_name

  # in it was in place operation then conv_name should be used
  if bn_in_place:
      out_layer_name = conv_name

  if use_relu:
    if out_layer_name_ResTen:
      relu_name = bn_name.replace('bn','relu')
    else:
      relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[out_layer_name], in_place=True)
    print("relu_name:" , relu_name)

def ConvBNLayerENet(net, ip_layer_name='', block_name='', num_output=0,
    kernel_size=1, pad=0, stride=1, eps=0.001, prePostFix='',
    kwArgs='', bn_type='bvlc', isFrozen=False):

  bn_in_place = True
  if bn_type == 'nvidia':
    #nvidia Caffe does not allow inplace in BN layer
    bn_in_place = False
 
  # parameters for convolution layer with batchnorm.
  conv_name = '{}{}{}'.format(prePostFix.conv_prefix, block_name, prePostFix.conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  print "debug here conv_name: ", conv_name
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[ip_layer_name], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h,
        **kwArgs.kwargs_conv[isFrozen])
  else:
    net[conv_name] = L.Convolution(net[ip_layer_name], num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, **kwArgs.kwargs_conv[isFrozen])

  if bn_type == 'none':
    bn_name = conv_name
  else:  
    bn_name = '{}{}{}'.format(prePostFix.bn_prefix, block_name,
        prePostFix.bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=bn_in_place,
        **kwArgs.bn_kwargs[isFrozen])
 
  if bn_type == 'bvlc':
    sb_name = '{}{}{}'.format(prePostFix.scale_prefix, block_name, prePostFix.scale_postfix)
    net[sb_name] = L.Scale(net[bn_name], in_place=True,
        **kwArgs.sb_kwargs[isFrozen])
  else:
    sb_name = bn_name
  
  #relu_name = '{}_relu'.format(conv_name)
  relu_name = '{}{}{}'.format(prePostFix.relu_prefix, block_name, prePostFix.relu_postfix)
  net[relu_name] = L.PReLU(net[sb_name], in_place=True,
      **kwArgs.prelu_kwargs[isFrozen])
  return relu_name

def ConvBNLayerJacNet(net, from_layer, out_layer, use_relu=True, num_output=0,
    kernel_size=3, pad=0, stride=1, dilation=1, group=1,
    prePostFix='', kwArgs='',isFrozen=False, bn_type='bvlc', use_shuffle=False):

  if (group <> 1) or (dilation <> 1):
    kwargs_conv_grp = kwArgs.kwargs_conv_grp_dil
  else:  
    kwargs_conv_grp = kwArgs.kwargs_conv
 
  bn_in_place = True
  if bn_type == 'nvidia':
    #nvidia Caffe does not allow inplace in BN layer
    bn_in_place = False

  conv_name = '{}{}{}'.format(prePostFix.conv_prefix, out_layer, prePostFix.conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  print("conv_name: ", conv_name)
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, group=group,
        **kwargs_conv_grp[isFrozen])
  else:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, group=group,
        **kwargs_conv_grp[isFrozen])

  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})
  
  op_layer_name = conv_name
  if bn_type <> 'none':
    bn_name = '{}{}{}'.format(prePostFix.bn_prefix, out_layer, prePostFix.bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=bn_in_place, **kwArgs.bn_kwargs[isFrozen])
    op_layer_name = bn_name
    if bn_type == 'bvlc':
      #in BVLC type BN one nees explictly scale/bias layer
      sb_name = '{}{}{}'.format(prePostFix.scale_prefix, out_layer, prePostFix.scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **kwArgs.sb_kwargs[isFrozen])
      op_layer_name = sb_name

  if use_relu:
    relu_name = '{}{}'.format(conv_name, prePostFix.relu_postfix)
    net[relu_name] = L.ReLU(net[op_layer_name], in_place=True)
    op_layer_name = relu_name

  if use_shuffle:
    shuffle_name = '{}{}'.format(conv_name, prePostFix.shuffle_postfix)
    net[shuffle_name] = L.ShuffleChannel(net[op_layer_name])
    op_layer_name = shuffle_name
  return op_layer_name   

def BNLayerENet(net, from_layer, block_name='', prePostFix='', 
    kwArgs='', bn_type='bvlc', freeze_layers='', isFrozen=False):
 
  bn_name = '{}{}{}'.format(prePostFix.bn_prefix, block_name, prePostFix.bn_postfix)
  sb_name = '{}{}{}'.format(prePostFix.scale_prefix, block_name, prePostFix.scale_postfix)
  if bn_type == 'bvlc':
    net[bn_name] = L.BatchNorm(net[from_layer], **kwArgs.bn_kwargs[isFrozen])
    net[sb_name] = L.Scale(net[bn_name], in_place=True, **kwArgs.sb_kwargs[isFrozen])
    op_layer_name = sb_name
  else:
    net[bn_name] = L.BatchNorm(net[from_layer], **kwArgs.bn_kwargs[isFrozen])
    op_layer_name = bn_name

  return op_layer_name

#function _bottleneck(internal_scale, use_relu, asymetric, dilated, input, output, downsample)
def BottleNeckLayer(net, ip_layer_name='', block_name='', eps=0.001,
    internal_scale=4, use_relu=1, asymetric=0, dilation=1, num_input_ch=0, num_output_ch=0, downsample=False,
    prePostFix='', kwArgs='', bn_type='bvlc', ex_net=False, isFrozen=False,
    dil_due_to_stride=False):

  #function _bottleneck(internal_scale, use_relu, asymetric, dilated, num_input_ch, num_output_ch, downsample)
  bn_in_place = True
  if bn_type == 'nvidia':
    #nvidia Caffe does not allow inplace in BN layer
    bn_in_place = False

  #########################
  #projection layer
  #########################
  num_output = num_input_ch/internal_scale
  sub_block_name = '{}_proj'.format(block_name) 
  proj_conv_name = '{}{}{}'.format(prePostFix.conv_prefix, sub_block_name, prePostFix.conv_postfix)
  if downsample==True:
    print "num_input_ch: ", num_input_ch
    print "internal_scale: ", internal_scale
    print "num_output: ", num_output
    #conv2x2_s2
    net[proj_conv_name] = L.Convolution(net[ip_layer_name], num_output=num_output,
          kernel_size=2, pad=0, stride=2, **kwArgs.kwargs_conv[isFrozen])
  else:
    #conv1x1
    print "proj_conv_name : ", proj_conv_name
    print "ip_layer_name : ", ip_layer_name
    net[proj_conv_name] = L.Convolution(net[ip_layer_name], num_output=num_output,
          kernel_size=1, pad=0, stride=1, **kwArgs.kwargs_conv[isFrozen])

  bn_name = '{}{}{}'.format(prePostFix.bn_prefix, sub_block_name, prePostFix.bn_postfix)
  if bn_type == 'none':
    bn_name = proj_conv_name
  else:  
    net[bn_name] = L.BatchNorm(net[proj_conv_name], in_place=bn_in_place, **kwArgs.bn_kwargs[isFrozen])

  if bn_type == 'bvlc': 
    sb_name = '{}{}{}'.format(prePostFix.scale_prefix, sub_block_name, prePostFix.scale_postfix)
    net[sb_name] = L.Scale(net[bn_name], in_place=True, **kwArgs.sb_kwargs[isFrozen])
  else:  
    sb_name = bn_name

  relu_name = '{}{}{}'.format(prePostFix.relu_prefix, sub_block_name, prePostFix.relu_postfix)
  net[relu_name] = L.PReLU(net[sb_name], in_place=True, **kwArgs.prelu_kwargs[isFrozen])

  op_proj_layer = relu_name

  #########################
  #Main conv layer
  #########################
  sub_block_name = '{}_main'.format(block_name) 
  main_conv_name = '{}{}{}'.format(prePostFix.conv_prefix, sub_block_name, prePostFix.conv_postfix)

  group = 1
  # for Xception type 2D separable conv 
  if ex_net:
    #same as suggested by Xception. But it is too slow to train.
    #group = num_output 
    group = 4
    kwargs_grp = kwArgs.kwargs_conv_grp_dil
  else:  
    kwargs_grp = kwArgs.kwargs_conv

  if dil_due_to_stride:
    print("dil_due_to_stride: ", dil_due_to_stride) 
    print("dilation: ", dilation) 
    dilation = dilation*2 

  if asymetric == 0:
    pad = int((3 + (dilation - 1) * 2) - 1) / 2
    if dilation <> 1:
      net[main_conv_name] = L.Convolution(net[relu_name], num_output=num_output,
          kernel_size=3, pad=pad, stride=1,dilation=dilation, group=group,
          **kwArgs.kwargs_conv_grp_dil[isFrozen])
    else:  
      net[main_conv_name] = L.Convolution(net[relu_name], num_output=num_output,
          kernel_size=3, pad=pad, stride=1,dilation=dilation, group=group,
          **kwargs_grp[isFrozen])
  else:
    #5x1 or 7x1
    pad = (asymetric-1)/2
    conv_name_Nx1 = '{}{}{}{}'.format(prePostFix.conv_prefix, sub_block_name, '_Nx1',prePostFix.conv_postfix)
    net[conv_name_Nx1] = L.Convolution(net[relu_name], num_output=num_output,
        kernel_h=asymetric, kernel_w=1, pad_h=pad, pad_w=0,
        stride_h=1, stride_w=1, group=group, **kwargs_grp[isFrozen])

    #1x5 or 1x7
    net[main_conv_name] = L.Convolution(net[conv_name_Nx1], num_output=num_output,
        kernel_h=1, kernel_w=asymetric, pad_h=0, pad_w=pad,
        stride_h=1, stride_w=1, group=group, **kwargs_grp[isFrozen])


  bn_name = '{}{}{}'.format(prePostFix.bn_prefix, sub_block_name, prePostFix.bn_postfix)

  if bn_type == 'none':
    bn_name = main_conv_name
  else:  
    net[bn_name] = L.BatchNorm(net[main_conv_name], in_place=bn_in_place,
        **kwArgs.bn_kwargs[isFrozen])

  if bn_type == 'bvlc':
    sb_name = '{}{}{}'.format(prePostFix.scale_prefix, sub_block_name, prePostFix.scale_postfix)
    net[sb_name] = L.Scale(net[bn_name], in_place=True, **kwArgs.sb_kwargs[isFrozen])
  else :
    sb_name = bn_name

  #relu_name = '{}_relu'.format(bn_name)
  relu_name = '{}{}{}'.format(prePostFix.relu_prefix, sub_block_name, prePostFix.relu_postfix)
  net[relu_name] = L.PReLU(net[sb_name], in_place=True, **kwArgs.prelu_kwargs[isFrozen])

  main_conv_op_name = relu_name

  if ex_net:
    #########################
    #Short circuit to Main conv layer aka Xception net
    #########################
    print "op_proj_layer : ", op_proj_layer
    print "ip_layer_name : ", ip_layer_name
    sub_block_name = '{}_main_short_circuit'.format(block_name) 
    short_conv_name = '{}{}{}'.format(prePostFix.conv_prefix, sub_block_name, prePostFix.conv_postfix)
    net[short_conv_name] = L.Convolution(net[op_proj_layer], num_output=num_output,
            kernel_size=1, pad=0, stride=1, **kwArgs.kwargs_conv[isFrozen])

    if bn_type == 'none':
      bn_name = short_conv_name
    else:  
      bn_name = '{}{}{}'.format(prePostFix.bn_prefix, sub_block_name, prePostFix.bn_postfix)
      net[bn_name] = L.BatchNorm(net[short_conv_name], in_place=bn_in_place, **kwArgs.bn_kwargs[isFrozen])

    if bn_type == 'bvlc': 
      sb_name = '{}{}{}'.format(prePostFix.scale_prefix, sub_block_name, prePostFix.scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **kwArgs.sb_kwargs[isFrozen])
    else:  
      sb_name = bn_name

    relu_name = '{}{}{}'.format(prePostFix.relu_prefix, sub_block_name, prePostFix.relu_postfix)
    net[relu_name] = L.PReLU(net[sb_name], in_place=True, **kwArgs.prelu_kwargs[isFrozen])
    short_circuit_name = relu_name

    #########################
    #merge main conv and Short circuit
    #########################
    main_merged_name =  sub_block_name + '_merged' 
    net[main_merged_name] = L.Eltwise(net[main_conv_op_name], net[short_circuit_name])
  else:
    main_merged_name = main_conv_op_name

  #########################
  #expansion layer
  #########################
  #conv1x1
  sub_block_name = '{}_expand'.format(block_name) 
  expand_conv_name = '{}{}{}'.format(prePostFix.conv_prefix, sub_block_name, prePostFix.conv_postfix)
  print "expand_conv_name: ", expand_conv_name 
  net[expand_conv_name] = L.Convolution(net[main_merged_name],
      num_output=num_output_ch, kernel_size=1, pad=0, stride=1, **kwArgs.kwargs_conv[isFrozen])
  
  if bn_type == 'none':
    bn_name = expand_conv_name
  else:  
    bn_name = '{}{}{}'.format(prePostFix.bn_prefix, sub_block_name, prePostFix.bn_postfix)
    net[bn_name] = L.BatchNorm(net[expand_conv_name], in_place=bn_in_place, **kwArgs.bn_kwargs[isFrozen])

  if bn_type == 'bvlc':
    sb_name = '{}{}{}'.format(prePostFix.scale_prefix, sub_block_name, prePostFix.scale_postfix)
    net[sb_name] = L.Scale(net[bn_name], in_place=True, **kwArgs.sb_kwargs[isFrozen])
  else:
    sb_name = bn_name

  relu_name = '{}{}{}'.format(prePostFix.relu_prefix, sub_block_name, prePostFix.relu_postfix)
  net[relu_name] = L.PReLU(net[sb_name], in_place=True, **kwArgs.prelu_kwargs[isFrozen])
  main_br_name = relu_name

  #########################
  #Max Pool layer - The other branch
  #########################
  other_br_out_name = ip_layer_name
  if downsample==True:
    pool_name = block_name + '_aux_br' + '_pool'
    net[pool_name]= L.Pooling(net[ip_layer_name], pool=P.Pooling.MAX, kernel_size=2, stride=2, pad=0)
    other_br_out_name = pool_name
 
  ##################################################
  #Merge aux br with main br by eltwise
  #need to pad auc branch if #op ch is higher than #ip ch 
  ##################################################
 
  p1_name = block_name + '_p1'
  p2_name = block_name + '_p2'
  p1_summed_name = block_name + '_p1_summed'
  if num_input_ch <> num_output_ch:
    net[p1_name], net[p2_name] = L.Slice(net[main_br_name],slice_param=dict(axis=1, slice_point=num_input_ch), ntop=2)
    net[p1_summed_name] = L.Eltwise(net[p1_name], net[other_br_out_name])
    net[block_name] = L.Concat(net[p1_summed_name], net[p2_name], axis=1)
  else:  
    net[block_name] = L.Eltwise(net[main_br_name], net[other_br_out_name])

  return block_name

def ResBody(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1, dilation=1, **bn_param):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'res{}_'.format(block_name)
  conv_postfix = ''
  bn_prefix = 'bn{}_'.format(block_name)
  bn_postfix = ''
  scale_prefix = 'scale{}_'.format(block_name)
  scale_postfix = ''
  use_scale = True

  if use_branch1:
    branch_name = 'branch1'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
        num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
    branch1 = from_layer

  branch_name = 'branch2a'
  ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
      num_output=out2a, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2b'
  if dilation == 1:
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
  else:
    pad = int((3 + (dilation - 1) * 2) - 1) / 2
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=pad, stride=1, use_scale=use_scale,
        dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2c'
  ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
      num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
  branch2 = '{}{}'.format(conv_prefix, branch_name)

  res_name = 'res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2])
  relu_name = '{}_relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)

def ResBodyTenCore(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1, dilation=1):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  print "Inside ResBodytenCore"
  #conv_prefix = 'res{}_'.format(block_name)
  #conv_postfix = ''
  #bn_prefix = 'bn{}_'.format(block_name)
  #bn_postfix = ''
  #scale_prefix = 'scale{}_'.format(block_name)
  #scale_postfix = ''
  
  epsResTen = 1E-5
  bias_decay_mul_in_conv = 1
  conv_postfix = ''
  bn_prefix = '{}_bn'.format(block_name) 
  bn_postfix = ''
  scale_prefix = '{}_scale'.format(block_name) 
  scale_postfix = ''
  use_scale = True

  conv_prefix = '{}_conv_expand'.format(block_name) 
  if use_branch1:
    branch_name = ''
    ConvBNLayer(net, from_layer, branch_name, use_bn=False, use_relu=False,
        num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
        eps=epsResTen,conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix,
        bias_term_swap=True,scale_filler=False,
        out_layer_name_ResTen=True, 
        bias_decay_mul_in_conv=bias_decay_mul_in_conv)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
    branch1 = from_layer

  conv_prefix = '{}_conv'.format(block_name) 

  branch_name = '1'
  ConvBNLayer(net, from_layer, branch_name, use_bn=False, use_relu=False,
      num_output=out2a, kernel_size=3, pad=1, stride=stride, use_scale=use_scale,
      eps=epsResTen,conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix,
      bias_term_swap=True,scale_filler=False,
      out_layer_name_ResTen=True,
      bias_decay_mul_in_conv=bias_decay_mul_in_conv)

  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = '2'
  #if dilation == 1:
  # next layer after the layer where stride is removed should have dilation 
  # not this layer
  if True:
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
        eps=epsResTen,conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix,
        use_conv=False,bias_term_swap=True,scale_filler=False,
        out_layer_name_ResTen=True,
        bias_decay_mul_in_conv=bias_decay_mul_in_conv)
  else:
    pad = int((3 + (dilation - 1) * 2) - 1) / 2
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=pad, stride=1, use_scale=use_scale,
        eps=epsResTen,dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix,
        use_conv=False,bias_term_swap=True,scale_filler=False,
        out_layer_name_ResTen=True,
        bias_decay_mul_in_conv=bias_decay_mul_in_conv)

  if dilation == 1:
    ConvBNLayer(net, out_name, branch_name, use_bn=False, use_relu=False,
        num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
        eps=epsResTen,conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix,
        scale_postfix=scale_postfix,bias_term_swap=True,scale_filler=False,
        out_layer_name_ResTen=True,
        bias_decay_mul_in_conv=bias_decay_mul_in_conv)
  else:
    pad = int((3 + (dilation - 1) * 2) - 1) / 2
    ConvBNLayer(net, out_name, branch_name, use_bn=False, use_relu=False,
        num_output=out2b, kernel_size=3, pad=pad, stride=1, use_scale=use_scale,
        eps=epsResTen, dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix,
        scale_postfix=scale_postfix,bias_term_swap=True,scale_filler=False,
        out_layer_name_ResTen=True,
        bias_decay_mul_in_conv=bias_decay_mul_in_conv)
 
  out_name = '{}{}'.format(conv_prefix, branch_name)

  #branch_name = 'branch2c'
  #ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
  #    num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
  #    conv_prefix=conv_prefix, conv_postfix=conv_postfix,
  #    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
  #    scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  branch2 = '{}{}'.format(conv_prefix, branch_name)

  res_name = '{}_sum'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2])
  print("op_layer_name: ", res_name)
  print "end of ResBodytenCore"
  #ResNet10 does not use RELU after elt-wise
  #relu_name = '{}_relu'.format(res_name)
  #net[relu_name] = L.ReLU(net[res_name], in_place=True)


def InceptionTower(net, from_layer, tower_name, layer_params, **bn_param):
  use_scale = False
  for param in layer_params:
    tower_layer = '{}/{}'.format(tower_name, param['name'])
    del param['name']
    if 'pool' in tower_layer:
      net[tower_layer] = L.Pooling(net[from_layer], **param)
    else:
      param.update(bn_param)
      ConvBNLayer(net, from_layer, tower_layer, use_bn=True, use_relu=True,
          use_scale=use_scale, **param)
    from_layer = tower_layer
  return net[from_layer]

def CreateAnnotatedDataLayer(source, batch_size=32, backend=P.Data.LMDB,
        output_label=True, train=True, label_map_file='', anno_type=None,
        transform_param={}, batch_sampler=[{}]):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    ntop = 1
    if output_label:
        ntop = 2
    annotated_data_param = {
        'label_map_file': label_map_file,
        'batch_sampler': batch_sampler,
        }
    if anno_type is not None:
        annotated_data_param.update({'anno_type': anno_type})
    return L.AnnotatedData(name="data", annotated_data_param=annotated_data_param,
        data_param=dict(batch_size=batch_size, backend=backend, source=source),
        ntop=ntop, **kwargs)


def ZFNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        dilated=False, dropout=True, need_fc8=False, freeze_layers=[]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    net.conv1 = L.Convolution(net[from_layer], num_output=96, pad=3, kernel_size=7, stride=2, **kwargs)
    net.relu1 = L.ReLU(net.conv1, in_place=True)

    net.norm1 = L.LRN(net.relu1, local_size=3, alpha=0.00005, beta=0.75,
            norm_region=P.LRN.WITHIN_CHANNEL, engine=P.LRN.CAFFE)

    net.pool1 = L.Pooling(net.norm1, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=2)

    net.conv2 = L.Convolution(net.pool1, num_output=256, pad=2, kernel_size=5, stride=2, **kwargs)
    net.relu2 = L.ReLU(net.conv2, in_place=True)

    net.norm2 = L.LRN(net.relu2, local_size=3, alpha=0.00005, beta=0.75,
            norm_region=P.LRN.WITHIN_CHANNEL, engine=P.LRN.CAFFE)

    net.pool2 = L.Pooling(net.norm2, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=2)

    net.conv3 = L.Convolution(net.pool2, num_output=384, pad=1, kernel_size=3, **kwargs)
    net.relu3 = L.ReLU(net.conv3, in_place=True)
    net.conv4 = L.Convolution(net.relu3, num_output=384, pad=1, kernel_size=3, **kwargs)
    net.relu4 = L.ReLU(net.conv4, in_place=True)
    net.conv5 = L.Convolution(net.relu4, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu5 = L.ReLU(net.conv5, in_place=True)

    if need_fc:
        if dilated:
            name = 'pool5'
            net[name] = L.Pooling(net.relu5, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            name = 'pool5'
            net[name] = L.Pooling(net.relu5, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=5, kernel_size=3, dilation=5, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=5, kernel_size=6, dilation=2, **kwargs)
            else:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=2, kernel_size=3, dilation=2,  **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=2, kernel_size=6, **kwargs)

            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc7 = L.Convolution(net.relu6, num_output=4096, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=4096)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=4096)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
    if need_fc8:
        from_layer = net.keys()[-1]
        if fully_conv:
            net.fc8 = L.Convolution(net[from_layer], num_output=1000, kernel_size=1, **kwargs)
        else:
            net.fc8 = L.InnerProduct(net[from_layer], num_output=1000)
        net.prob = L.Softmax(net.fc8)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net


def VGGNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        dilated=False, nopool=False, dropout=True, freeze_layers=[], dilate_pool4=False, group=1):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    net.conv1_1 = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)

    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1,
        kernel_size=3, group=group, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    if nopool:
        name = 'conv1_3'
        net[name] = L.Convolution(net.relu1_2, num_output=64, pad=1,
            kernel_size=3, stride=2, group=group, **kwargs)
    else:
        name = 'pool1'
        net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, group=group, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, group=group, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    if nopool:
        name = 'conv2_3'
        net[name] = L.Convolution(net.relu2_2, num_output=128, pad=1, kernel_size=3, stride=2, group=group, **kwargs)
    else:
        name = 'pool2'
        net[name] = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, group=group, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, group=group, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, group=group, **kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

    if nopool:
        name = 'conv3_4'
        net[name] = L.Convolution(net.relu3_3, num_output=256, pad=1, kernel_size=3, stride=2, group=group, **kwargs)
    else:
        name = 'pool3'
        net[name] = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, group=group, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, group=group, **kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, group=group, **kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    if nopool:
        name = 'conv4_4'
        net[name] = L.Convolution(net.relu4_3, num_output=512, pad=1, kernel_size=3, stride=2, group=group, **kwargs)
    else:
        name = 'pool4'
        if dilate_pool4:
            net[name] = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=3, stride=1, pad=1)
            dilation = 2
        else:
            net[name] = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)
            dilation = 1

    kernel_size = 3
    pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
    net.conv5_1 = L.Convolution(net[name], num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation, group=group,**kwargs)
    net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
    net.conv5_2 = L.Convolution(net.relu5_1, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation, group=group,**kwargs)
    net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
    net.conv5_3 = L.Convolution(net.relu5_2, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation, group=group,**kwargs)
    net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    dilation = dilation * 6
                    kernel_size = 3
                    num_output = 1024
                else:
                    dilation = dilation * 2
                    kernel_size = 7
                    num_output = 4096
            else:
                if reduced:
                    dilation = dilation * 3
                    kernel_size = 3
                    num_output = 1024
                else:
                    kernel_size = 7
                    num_output = 4096
            pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
            net.fc6 = L.Convolution(net[name], num_output=num_output, pad=pad, kernel_size=kernel_size, dilation=dilation, **kwargs)

            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc7 = L.Convolution(net.relu6, num_output=4096, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=4096)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=4096)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net

#TI Base N/W defined known as JacinctoNet11V2, Base Network
def JacintoNetV2Body(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        dilated=False, nopool=False, dropout=True, freeze_layers=[],
        bn_type='bvlc', caffe_fork='', training_type='SSD', fully_conv_at_end=True):

    down_sample_list = [2,2,2,2,2]
    dilation_list = [1,1,1,1,1]
    
    # for SSD don't use strides in conv5 layer
    if training_type=='SSD':
      down_sample_list[4] = 1
      dilation_list[4] = 2

    #"bvlc", "nvidia"
    if caffe_fork == '':
      caffe_fork = bn_type

    prePostFix = PrePostFixJacintoNetV2(bn_type)
    kwArgs = KW_Args(caffe_fork,eps=0.0001)

    assert from_layer in net.keys()

    ##################
    stage=0
    dilation=dilation_list[stage]
    stride=down_sample_list[stage]
    #conv1a uses strided conv. So no pool layer preceeds it
    block_name = 'conv1a'
    isFrozen='conv1a' in freeze_layers
    op_layer_name = ConvBNLayerJacNet(net, from_layer, block_name , bn_type=bn_type, use_relu=True,
        num_output=32, kernel_size=5, pad=2*dilation, stride=stride,
        prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen,dilation=dilation)

    isFrozen='conv1b' in freeze_layers
    ip_layer_name = op_layer_name
    block_name = 'conv1b'
    op_layer_name = ConvBNLayerJacNet(net, ip_layer_name, block_name, bn_type=bn_type, use_relu=True,
        num_output=32, kernel_size=3, pad=dilation, stride=1,  prePostFix=prePostFix, 
        kwArgs=kwArgs,isFrozen=isFrozen, group=4, dilation=dilation)

    ##################
    stage=1
    dilation=dilation_list[stage]
    stride=down_sample_list[stage]
    
    ip_layer_name = op_layer_name
    op_layer_name = 'pool1'
    net[op_layer_name] = L.Pooling(net[ip_layer_name], pool=P.Pooling.MAX, kernel_size=stride, stride=stride)

    isFrozen='res2a_branch2a' in freeze_layers
    ip_layer_name = op_layer_name
    block_name = 'res2a_branch2a'
    op_layer_name = ConvBNLayerJacNet(net, ip_layer_name , block_name , bn_type=bn_type, use_relu=True,
        num_output=64, kernel_size=3, pad=dilation, stride=1,  prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, group=1, dilation=dilation)

    isFrozen='res2a_branch2b' in freeze_layers
    ip_layer_name = op_layer_name
    block_name = 'res2a_branch2b'
    op_layer_name = ConvBNLayerJacNet(net, ip_layer_name, block_name, bn_type=bn_type, use_relu=True,
        num_output=64, kernel_size=3, pad=dilation, stride=1,  prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, group=4, dilation=dilation)

    ##################
    stage=2
    dilation=dilation_list[stage]
    stride=down_sample_list[stage]
    
    ip_layer_name = op_layer_name
    op_layer_name = 'pool2'
    net[op_layer_name] = L.Pooling(net[ip_layer_name], pool=P.Pooling.MAX, kernel_size=stride, stride=stride)

    isFrozen='res3a_branch2a' in freeze_layers
    ip_layer_name = op_layer_name
    block_name = 'res3a_branch2a'

    op_layer_name = ConvBNLayerJacNet(net, ip_layer_name, block_name , bn_type=bn_type, use_relu=True,
        num_output=128, kernel_size=3, pad=1, stride=1,  prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, group=1, dilation=dilation)

    isFrozen='res3a_branch2b' in freeze_layers
    ip_layer_name = op_layer_name
    block_name = 'res3a_branch2b'

    op_layer_name = ConvBNLayerJacNet(net, ip_layer_name , block_name, bn_type=bn_type, use_relu=True,
        num_output=128, kernel_size=3, pad=dilation, stride=1,  prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, group=4, dilation=dilation)

    ##################
    stage=3
    dilation=dilation_list[stage]
    stride=down_sample_list[stage]

    ip_layer_name = op_layer_name
    op_layer_name = 'pool3'
    net[op_layer_name] = L.Pooling(net[ip_layer_name], pool=P.Pooling.MAX, kernel_size=stride, stride=stride)

    isFrozen='res4a_branch2a' in freeze_layers
    ip_layer_name = op_layer_name
    block_name = 'res4a_branch2a'

    op_layer_name = ConvBNLayerJacNet(net, ip_layer_name, block_name , bn_type=bn_type, use_relu=True,
        num_output=256, kernel_size=3, pad=1, stride=1,  prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, group=1, dilation=dilation)

    isFrozen='res4a_branch2b' in freeze_layers
    ip_layer_name = op_layer_name
    block_name = 'res4a_branch2b'
    op_layer_name = ConvBNLayerJacNet(net, ip_layer_name , block_name, bn_type=bn_type, use_relu=True,
        num_output=256, kernel_size=3, pad=1, stride=1,  prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, group=4, dilation=dilation)

    ##################
    stage=4
    dilation=dilation_list[stage]
    stride=down_sample_list[stage]
 
    if stride != 1:
      ip_layer_name = op_layer_name
      op_layer_name = 'pool4'
      net[op_layer_name] = L.Pooling(net[ip_layer_name], pool=P.Pooling.MAX, kernel_size=stride, stride=stride)

    isFrozen='res5a_branch2a' in freeze_layers
    ip_layer_name = op_layer_name
    block_name = 'res5a_branch2a'
    
    kernel_size=3
    pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
    op_layer_name = ConvBNLayerJacNet(net, ip_layer_name, block_name , bn_type=bn_type, use_relu=True,
        num_output=512, kernel_size=kernel_size, pad=pad, stride=1,  prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, group=1, dilation=dilation)

    isFrozen='res5a_branch2b' in freeze_layers
    ip_layer_name = op_layer_name
    block_name = 'res5a_branch2b'
    op_layer_name = ConvBNLayerJacNet(net, ip_layer_name, block_name, bn_type=bn_type, use_relu=True,
        num_output=512, kernel_size=kernel_size, pad=pad, stride=1, dilation=dilation,  prePostFix=prePostFix, kwArgs=kwArgs,
        isFrozen=isFrozen, group=4)
    
    ##################
    if fully_conv_at_end:
      #layer 'fc6'
      #ip_name = 'res5a_branch2b_relu'
      ip_name = op_layer_name
      isFrozen='fc6' in freeze_layers
      net.fc6 = L.Convolution(net[ip_name], num_output=1024, pad=6, kernel_size=3, dilation=6, **kwArgs.kwargs_conv[isFrozen])
      net.relu6 = L.ReLU(net.fc6, in_place=True)

      #layer 'fc7'
      isFrozen='fc7' in freeze_layers
      net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwArgs.kwargs_conv[isFrozen])
      net.relu7 = L.ReLU(net.fc7, in_place=True)

    #lossOp = L.Softmax(net.fc7)
    #accuracyOpTop1 = L.AccuracyLayer(net.fc7)
    #accuracyOpTop5 = L.AccuracyLayer(net.fc7,top_k=5)

    return net

#TI Base N/W defined known as Jacincto Base Net
def JacintoNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        dilated=False, nopool=False, dropout=True, freeze_layers=[],
        bn_type='bvlc', caffe_fork='', training_type='SSD', fully_conv_at_end=True):

    #"bvlc", "nvidia"
    if caffe_fork == '':
      caffe_fork = bn_type

    prePostFix = PrePostFixJacintoNet(bn_type)
    kwArgs = KW_Args(caffe_fork,eps=0.0001)

    assert from_layer in net.keys()
    #net.conv1a = L.Convolution(net[from_layer], num_output=32, pad=2, kernel_size=5, stride=2, **kwargs)
    isFrozen='conv1a' in freeze_layers
    ConvBNLayerJacNet(net, from_layer, 'conv1a' , bn_type=bn_type, use_relu=True,
        num_output=32, kernel_size=5, pad=2, stride=2,
         prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen)

    isFrozen='conv1b' in freeze_layers
    ConvBNLayerJacNet(net, 'conv1a' , 'conv1b' , bn_type=bn_type, use_relu=True,
        num_output=32, kernel_size=3, pad=1, stride=1,  prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, group=4)

    isFrozen='res2a_branch2a' in freeze_layers
    ConvBNLayerJacNet(net, 'conv1b' , 'res2a_branch2a' , bn_type=bn_type, use_relu=True,
        num_output=64, kernel_size=3, pad=1, stride=2,  prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, group=1)

    isFrozen='res2a_branch2b' in freeze_layers
    ConvBNLayerJacNet(net, 'res2a_branch2a', 'res2a_branch2b', bn_type=bn_type, use_relu=True,
        num_output=64, kernel_size=3, pad=1, stride=1,  prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, group=4)

    isFrozen='res3a_branch2a' in freeze_layers
    ConvBNLayerJacNet(net, 'res2a_branch2b', 'res3a_branch2a' , bn_type=bn_type, use_relu=True,
        num_output=128, kernel_size=3, pad=1, stride=2,  prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, group=1)

    isFrozen='res3a_branch2b' in freeze_layers
    ConvBNLayerJacNet(net, 'res3a_branch2a' , 'res3a_branch2b', bn_type=bn_type, use_relu=True,
        num_output=128, kernel_size=3, pad=1, stride=1,  prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, group=4)

    isFrozen='res4a_branch2a' in freeze_layers
    ConvBNLayerJacNet(net, 'res3a_branch2b', 'res4a_branch2a' , bn_type=bn_type, use_relu=True,
        num_output=256, kernel_size=3, pad=1, stride=2,  prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, group=1)

    isFrozen='res4a_branch2b' in freeze_layers
    ConvBNLayerJacNet(net, 'res4a_branch2a' , 'res4a_branch2b', bn_type=bn_type, use_relu=True,
        num_output=256, kernel_size=3, pad=1, stride=1,  prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, group=4)

    if training_type=='SSD':
       strideInConv5 = 1
       dilInConv5 = 2
    else: #imagenet
       strideInConv5 = 2
       dilInConv5 = 1

    isFrozen='res5a_branch2a' in freeze_layers
    kernel_size = 3
    dilation = 1
    pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
    ConvBNLayerJacNet(net, 'res4a_branch2b', 'res5a_branch2a' , bn_type=bn_type, use_relu=True,
        num_output=512, kernel_size=kernel_size, pad=pad, stride=strideInConv5,  prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen, group=1)

    isFrozen='res5a_branch2b' in freeze_layers
    kernel_size = 3
    dilation = dilInConv5
    pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
    ConvBNLayerJacNet(net, 'res5a_branch2a' , 'res5a_branch2b', bn_type=bn_type, use_relu=True,
        num_output=512, kernel_size=kernel_size, pad=pad, stride=1, dilation=dilation,  prePostFix=prePostFix, kwArgs=kwArgs,
        isFrozen=isFrozen, group=4)
  
    if fully_conv_at_end:
      #layer 'fc6'
      ip_name = 'res5a_branch2b_relu'
      isFrozen='fc6' in freeze_layers
      net.fc6 = L.Convolution(net[ip_name], num_output=1024, pad=6, kernel_size=3, dilation=6, **kwArgs.kwargs_conv[isFrozen])
      net.relu6 = L.ReLU(net.fc6, in_place=True)

      #layer 'fc7'
      isFrozen='fc7' in freeze_layers
      net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwArgs.kwargs_conv[isFrozen])
      net.relu7 = L.ReLU(net.fc7, in_place=True)

    #lossOp = L.Softmax(net.fc1000)
    #accuracyOpTop1 = L.AccuracyLayer(net.fc1000)
    #accuracyOpTop5 = L.AccuracyLayer(net.fc1000,top_k=5)

    return net

#TI Base N/W defined known as Jacincto Base Net
def JINetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        dilated=False, nopool=False, dropout=True, freeze_layers=[]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)],
            'weight_filler': dict(type='msra'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    #net.conv1a = L.Convolution(net[from_layer], num_output=32, pad=2, kernel_size=5, stride=2, **kwargs)
    use_bn_flag = True


    #############################################################################################################
    # BN layer before anything
    #############################################################################################################
    bn_prefix = ''
    bn_postfix = '_bn'
    scale_prefix = ''
    scale_postfix = '_scale'
    conv_prefix = ''
    conv_postfix = ''

    layer_label = 'data'
    ConvBNLayer(net, from_layer, layer_label, use_bn=True, use_relu=False,
        num_output=64, kernel_size=7, pad=3, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix,
        use_conv=False, bn_in_place=False,scale_filler=False,
        out_layer_name_ResTen=True)

    op_layer_name ='{}{}{}'.format(scale_prefix, layer_label, scale_postfix)

    #############################################################################################################
    #conv1
    #############################################################################################################
    # reduce kernel size to 3x3 from 5x5
    ConvBNLayerJacNet(net, op_layer_name, 'conv1a' , use_bn=use_bn_flag, use_relu=True,
        num_output=64, kernel_size=3, pad=1, stride=2, eps=0.0001, use_scale=True)

    #add bottleneck layer
    ConvBNLayerJacNet(net, 'conv1a', 'conv1_bot' , use_bn=use_bn_flag, use_relu=True,
        num_output=16, kernel_size=1, pad=0, stride=1, eps=0.0001, use_scale=True)

    #removed group as trying with bottleneck layer     
    ConvBNLayerJacNet(net, 'conv1_bot' , 'conv1b' , use_bn=use_bn_flag, use_relu=True,
        num_output=16, kernel_size=3, pad=1, stride=1, eps=0.0001, use_scale=True)

    #Add one more later of conv to compensate for reducing kernel size in conv1a     
    ConvBNLayerJacNet(net, 'conv1b' , 'conv1c' , use_bn=use_bn_flag, use_relu=True,
        num_output=16, kernel_size=3, pad=1, stride=2, eps=0.0001, use_scale=True)

    #add de-bottleneck layer
    ConvBNLayerJacNet(net, 'conv1c', 'conv1_debot' , use_bn=use_bn_flag, use_relu=True,
        num_output=64, kernel_size=1, pad=0, stride=1, eps=0.0001, use_scale=True)
    #############################################################################################################
    # res2a
    #############################################################################################################
    #add bottleneck layer
    ConvBNLayerJacNet(net, 'conv1_debot', 'res2a_bot' , use_bn=use_bn_flag, use_relu=True,
        num_output=32, kernel_size=1, pad=0, stride=1, eps=0.0001, use_scale=True)

    ConvBNLayerJacNet(net, 'res2a_bot' , 'res2a_branch2a' , use_bn=use_bn_flag, use_relu=True,
        num_output=32, kernel_size=3, pad=1, stride=2, eps=0.0001, use_scale=True)

    ConvBNLayerJacNet(net, 'res2a_branch2a', 'res2a_branch2b', use_bn=use_bn_flag, use_relu=True,
        num_output=32, kernel_size=3, pad=1, stride=1, eps=0.0001, use_scale=True)

    #add de-bottleneck layer
    ConvBNLayerJacNet(net, 'res2a_branch2b', 'res2a_debot' , use_bn=use_bn_flag, use_relu=True,
        num_output=128, kernel_size=1, pad=0, stride=1, eps=0.0001, use_scale=True)

    #############################################################################################################
    # res3a
    #############################################################################################################
    #add bottleneck layer
    ConvBNLayerJacNet(net, 'res2a_debot', 'res3a_bot' , use_bn=use_bn_flag, use_relu=True,
        num_output=64, kernel_size=1, pad=0, stride=1, eps=0.0001, use_scale=True)
  
    ConvBNLayerJacNet(net, 'res3a_bot', 'res3a_branch2a' , use_bn=use_bn_flag, use_relu=True,
        num_output=64, kernel_size=3, pad=1, stride=2, eps=0.0001, use_scale=True)

    ConvBNLayerJacNet(net, 'res3a_branch2a' , 'res3a_branch2b', use_bn=use_bn_flag, use_relu=True,
        num_output=64, kernel_size=3, pad=1, stride=1, eps=0.0001, use_scale=True)

    #add de-bottleneck layer
    ConvBNLayerJacNet(net, 'res3a_branch2b', 'res3a_debot' , use_bn=use_bn_flag, use_relu=True,
        num_output=512, kernel_size=1, pad=0, stride=1, eps=0.0001, use_scale=True)

    #############################################################################################################
    # res4a
    #############################################################################################################
     #add bottleneck layer
    ConvBNLayerJacNet(net, 'res3a_debot', 'res4a_bot' , use_bn=use_bn_flag, use_relu=True,
        num_output=128, kernel_size=1, pad=0, stride=1, eps=0.0001, use_scale=True)
  
    ConvBNLayerJacNet(net, 'res4a_bot', 'res4a_branch2a' , use_bn=use_bn_flag, use_relu=True,
        num_output=128, kernel_size=3, pad=1, stride=2, eps=0.0001, use_scale=True)

    ConvBNLayerJacNet(net, 'res4a_branch2a' , 'res4a_branch2b', use_bn=use_bn_flag, use_relu=True,
        num_output=128, kernel_size=3, pad=1, stride=1, eps=0.0001, use_scale=True)

    #add de-bottleneck layer
    ConvBNLayerJacNet(net, 'res4a_branch2b', 'res4a_debot' , use_bn=use_bn_flag, use_relu=True,
        num_output=512, kernel_size=1, pad=0, stride=1, eps=0.0001, use_scale=True)

    #############################################################################################################
    # res5a
    #############################################################################################################
    #add bottleneck layer
    ConvBNLayerJacNet(net, 'res4a_debot', 'res5a_bot' , use_bn=use_bn_flag, use_relu=True,
        num_output=128, kernel_size=1, pad=0, stride=1, eps=0.0001, use_scale=True)
  
    ConvBNLayerJacNet(net, 'res5a_bot', 'res5a_branch2a' , use_bn=use_bn_flag, use_relu=True,
        num_output=128, kernel_size=3, pad=1, stride=1, eps=0.0001, use_scale=True)

    ConvBNLayerJacNet(net, 'res5a_branch2a' , 'res5a_branch2b', use_bn=use_bn_flag, use_relu=True,
        num_output=128, kernel_size=3, pad=1, stride=1, eps=0.0001, dilation=2, use_scale=True)

    #add de-bottleneck layer
    ConvBNLayerJacNet(net, 'res5a_branch2b', 'res5a_debot' , use_bn=use_bn_flag, use_relu=True,
        num_output=1024, kernel_size=1, pad=0, stride=1, eps=0.0001, use_scale=True)

    #############################################################################################################
    #layer 'fc6'
    ip_name = 'res5a_debot'
    net.fc6 = L.Convolution(net[ip_name], num_output=1024, pad=6, kernel_size=3, dilation=6, **kwargs)
    net.relu6 = L.ReLU(net.fc6, in_place=True)

    #layer 'fc7'
    net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
    net.relu7 = L.ReLU(net.fc7, in_place=True)

    #:SN, how do I give two inputs, fc1000 and label
    #lossOp = L.Softmax(net.fc1000)

    #:SN, how do I give two inputs, fc1000 and label
    #:SN, what in include here
    #accuracyOpTop1 = L.AccuracyLayer(net.fc1000)
    
    #:SN, how do I give two inputs, fc1000 and label
    #accuracyOpTop5 = L.AccuracyLayer(net.fc1000,top_k=5)

    return net

class PrePostFix:
    #Class for prefix and postfix
    def __init__(self, bn_type='bvlc'):
      self.bn_prefix = ''
      self.bn_postfix = '_{}_bn'.format(bn_type)
      self.scale_prefix = ''
      self.scale_postfix = '_{}_scale'.format(bn_type)
      self.conv_prefix = ''
      self.conv_postfix = '_conv'
      self.relu_prefix = ''
      self.relu_postfix = '_relu'

class PrePostFixJacintoNet:
    #Class for prefix and postfix
    def __init__(self, bn_type='bvlc'):

      self.bn_prefix = ''
      self.bn_postfix = '_bn'
      self.scale_prefix = ''
      self.scale_postfix = '_scale'
      self.conv_prefix = ''
      self.conv_postfix = ''
      self.relu_prefix = ''
      self.relu_postfix = '_relu'

class PrePostFixMobileNet:
    #Class for prefix and postfix
    def __init__(self, bn_type='bvlc', postfix_char='_'):
      self.bn_prefix = ''
      self.bn_postfix = '_{}_bn'.format(bn_type)
      self.scale_prefix = ''
      self.scale_postfix = '_{}_scale'.format(bn_type)
      self.conv_prefix = ''
      self.conv_postfix = ''
      self.relu_prefix = ''
      self.relu_postfix = '_relu'
      self.shuffle_prefix = ''
      self.shuffle_postfix = '_shfl'

      if postfix_char == '/':
        self.relu_postfix = '/relu'
        self.bn_postfix = '/bn'
        self.scale_postfix = '/scale'
        self.shuffle_postfix = '/shfl'

class PrePostFixJacintoNetV2:
    #Class for prefix and postfix
    def __init__(self, bn_type='bvlc'):

      self.bn_prefix = ''
      self.bn_postfix = '_{}_bn'.format(bn_type)
      self.scale_prefix = ''
      self.scale_postfix = '_{}_scale'.format(bn_type)
      self.conv_prefix = ''
      self.conv_postfix = ''
      self.relu_prefix = ''
      self.relu_postfix = '_relu'


def zeroOutLearnableParam(kwargs=''):
  #print("kwargs: ", kwargs)
  for param in kwargs['param']:
    #print("param: ", param)
    param['lr_mult'] = 0
    param['decay_mult'] = 0

class KW_Args(object):

    kwargs_conv = [] 
    kwargs_conv_grp_dil = []
    sb_kwargs = []
    bias_kwargs = []
    prelu_kwargs = []
    bn_kwargs = []

    #Class for keyword Args
    def __init__(self, caffe_fork='bvlc', eps=0.001, bias_term=True, param_in_sb=True, caffeForDilGrp=True):
      
      #############################################################################################################
      # Params for enet
      #############################################################################################################

      # parameters for convolution layer with batchnorm.
      if bias_term:
        param_conv = {
          'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
          'weight_filler': dict(type='msra', std=0.01),
          'bias_filler': dict(type='constant', value=0),
          'bias_term': bias_term,
          }
      else:  
        param_conv = {
          'param': [dict(lr_mult=1, decay_mult=1)],
          'weight_filler': dict(type='msra', std=0.01),
          'bias_term': bias_term,
          }

      # In BVLC Caffe version, CUDNN does not support 'group' feature or dilated conv
      # So use CAFFE engine instead of CUDNN(def)
      if caffe_fork == 'bvlc':
        if bias_term:
          param_conv_grp_dil = {
              'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
              'weight_filler': dict(type='msra', std=0.01),
              'bias_filler': dict(type='constant', value=0),
              'bias_term': bias_term,
              #'engine': 1, #CAFFE
              }
        else: 
          param_conv_grp_dil = {
              'param': [dict(lr_mult=1, decay_mult=1)],
              'weight_filler': dict(type='msra', std=0.01),
              'bias_term': bias_term,
              #'engine': 1, #CAFFE
              }
        if caffeForDilGrp:
          param_conv_grp_dil['engine'] = 1
      else:
        param_conv_grp_dil = param_conv


      # parameters for batchnorm layer.
      if caffe_fork == "bvlc" :
        param_bn_kwargs = {
            'param': [dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)],
            'eps': eps,
            }
      else : #caffe_fork == "nvidia"
        param_bn_kwargs = {
            #scale, shift/bias,global mean, global var
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=1,decay_mult=1),
              dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
            'eps': eps,
            #'scale_filler': dict(type='constant', value=1),
            #'bias_filler': dict(type='constant', value=0),
            'moving_average_fraction': 0.99
            }

      param_prelu_kwargs = {
          'param': [dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          'channel_shared': False
          }


      # parameters for scale bias layer after batchnorm.
      param_sb_kwargs = {
           'bias_term': True,
           #'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)],
           'filler': dict(type='constant', value=1.0),
           'bias_filler': dict(type='constant', value=0.0),
           }

      if(param_in_sb) :
        param_sb_kwargs['param'] = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)]

      if bias_term:
        param_bias_kwargs = {
          'param': [dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }

      self.kwargs_conv.append(param_conv)
      self.kwargs_conv_grp_dil.append(param_conv_grp_dil)
      self.bn_kwargs.append(param_bn_kwargs)
      self.prelu_kwargs.append(param_prelu_kwargs)
      self.sb_kwargs.append(param_sb_kwargs)

      if bias_term:
        self.bias_kwargs.append(param_bias_kwargs)

      print("param_conv ", param_conv)
      param_conv1 = copy.deepcopy(param_conv)
      param_conv_grp_dil1 = copy.deepcopy(param_conv_grp_dil)
      param_sb_kwargs1 = copy.deepcopy(param_sb_kwargs) 

      if bias_term:
        param_bias_kwargs1 = copy.deepcopy(param_bias_kwargs)
      param_prelu_kwargs1 = copy.deepcopy(param_prelu_kwargs)
      param_bn_kwargs1 = copy.deepcopy(param_bn_kwargs)

      zeroOutLearnableParam(param_conv1)
      zeroOutLearnableParam(param_conv_grp_dil1)
      if param_in_sb:
        zeroOutLearnableParam(param_sb_kwargs1)
      if bias_term:
        zeroOutLearnableParam(param_bias_kwargs1)
      zeroOutLearnableParam(param_prelu_kwargs1)
      zeroOutLearnableParam(param_bn_kwargs1)

      self.kwargs_conv.append(param_conv1)
      self.kwargs_conv_grp_dil.append(param_conv_grp_dil1)
      self.bn_kwargs.append(param_bn_kwargs1)
      self.prelu_kwargs.append(param_prelu_kwargs1)
      self.sb_kwargs.append(param_sb_kwargs1)
      if bias_term:
        self.bias_kwargs.append(param_bias_kwargs1)

      print("param_conv ", self.kwargs_conv[0])
      print("param_conv1 ", self.kwargs_conv[1])

#########################
# ENET
#########################
def ENetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        dilated=False, nopool=False, dropout=True, freeze_layers=[],
        bn_type='bvlc', enable_dilation=False, ex_net=False,
        bn_at_start=True, caffe_fork='', training_type='SSD', dil_bf=False):
  
    assert from_layer in net.keys()
 
    #"bvlc", "nvidia"
    if caffe_fork == '':
      caffe_fork = bn_type

    if bn_type == 'none': 
      bn_at_start = False
    
    #downsample_bot = [init, bot_1,bot_2,bot_3,bot_4]
    if training_type  == 'cifar10':
      # for Cifar-10 have only 2 down sampling layers
      downsample_bot = [False,False,False,True,True]
    elif training_type == 'IMGNET':
      # for imagenet classification training do not have ds in conv5 layer 
      downsample_bot = [True,True,True,True,True]
    else: #for SSD  
      downsample_bot = [True,True,True,True,False]

    #for cifar10 as image size is 32x32. One can have only two downsample.
    # so avoide stride by 2/downsampling in the early stages
    stride_init_layer = 2
    pad_init_pool = 0
    kernel_size_init_pool = 2
    if(downsample_bot[0] == False):
        stride_init_layer = 1
        pad_init_pool = 1
        kernel_size_init_pool = 3

    dil_bot2_last = 1
    dil_bot3_last = 1
    dil_bot4_last = 1
    dil_bot2_non_last = 1
    dil_bot3_non_last = 1
    dil_bot4_non_last = 1

    if enable_dilation:
      dil_bot2_last = 2
      dil_bot3_last = 4
      dil_bot4_last = 8
      if dil_bf == True: 
        #as per ENET_V12 only last bottleneck layer should have dilation
        dil_bot2_non_last = 1
        dil_bot3_non_last = 1
        dil_bot4_non_last = 1
      else:  
        # this was to match earlier wrong behaviour. where dilation was happening in al three layers 
        dil_bot2_non_last = dil_bot2_last
        dil_bot3_non_last = dil_bot3_last
        dil_bot4_non_last = dil_bot4_last


    # dil_due_to_stride: if need to generate train.txt for imagenet training with dilation in
    # conv5. didn't work 
    dil_due_to_stride = False
    prePostFix = PrePostFix(bn_type)
    kwArgs = KW_Args(caffe_fork)

    #############################################################################################################
    # BN layer before anything
    #############################################################################################################
    if bn_at_start:
      layer_label = 'mean_sub'
      isFrozen = 'mean_sub' in freeze_layers
      op_layer_name = BNLayerENet(net, from_layer, block_name=layer_label, prePostFix=prePostFix,
          kwArgs=kwArgs, bn_type=bn_type, isFrozen=isFrozen)
    else:
      op_layer_name = from_layer  

    #############################################################################################################
    #Init layer
    #############################################################################################################
    # conv_3x3_s2
    init_layers =[]

    isFrozen = 'init' in freeze_layers
    op_layer_name_br1 = ConvBNLayerENet(net, ip_layer_name=op_layer_name, block_name = 'init', num_output=13,
        kernel_size=3, pad=1, stride=stride_init_layer, eps=0.0001, prePostFix=prePostFix,
        kwArgs=kwArgs, bn_type=bn_type, isFrozen=isFrozen)

    init_layers.append(net[op_layer_name_br1])

    net['init_max_pool'] = L.Pooling(net[op_layer_name], pool=P.Pooling.MAX,
        kernel_size=kernel_size_init_pool, stride=stride_init_layer, pad=pad_init_pool)
    init_layers.append(net['init_max_pool'])
    net['init_concat'] = L.Concat(*init_layers, axis=1)

    #############################################################################################################
    #Bottleneck1
    #############################################################################################################
    #add bottleneck layer BottleNeck1.0
    isFrozen = 'bottleNeck1' in freeze_layers 
    op_layer_name = BottleNeckLayer(net, ip_layer_name='init_concat', block_name='bot1_0' , num_input_ch=16, 
        num_output_ch=64, internal_scale=1, downsample=downsample_bot[1],
        prePostFix=prePostFix, kwArgs=kwArgs, bn_type=bn_type,
        ex_net=ex_net, isFrozen=isFrozen)

    op_layer_name = BottleNeckLayer(net, ip_layer_name=op_layer_name, block_name='bot1_1' , num_input_ch=64,
        num_output_ch=128,
        prePostFix=prePostFix, kwArgs=kwArgs,bn_type=bn_type, ex_net=ex_net, isFrozen=isFrozen)

    op_layer_name = BottleNeckLayer(net, ip_layer_name=op_layer_name, block_name='bot1_2' , num_input_ch=128,
        num_output_ch=128,
        prePostFix=prePostFix, kwArgs=kwArgs,bn_type=bn_type, ex_net=ex_net, isFrozen=isFrozen)

    #############################################################################################################
    # bottlenec2
    #############################################################################################################
    isFrozen = 'bottleNeck2' in freeze_layers 
    op_layer_name = BottleNeckLayer(net, ip_layer_name=op_layer_name, block_name='bot2_0' , num_input_ch=128,
        num_output_ch=256, downsample=downsample_bot[2], prePostFix=prePostFix, kwArgs=kwArgs,bn_type=bn_type,
        ex_net=ex_net, dilation=dil_bot2_non_last, isFrozen=isFrozen)

    op_layer_name = BottleNeckLayer(net, ip_layer_name=op_layer_name, block_name='bot2_1' , num_input_ch=256,
        num_output_ch=256,prePostFix=prePostFix, kwArgs=kwArgs,bn_type=bn_type, ex_net=ex_net, 
        dilation=dil_bot2_non_last, isFrozen=isFrozen)

    op_layer_name = BottleNeckLayer(net, ip_layer_name=op_layer_name, block_name='bot2_2' , num_input_ch=256,
        num_output_ch=256, prePostFix=prePostFix, kwArgs=kwArgs,bn_type=bn_type, ex_net=ex_net, 
        dilation=dil_bot2_last, isFrozen=isFrozen)


    #############################################################################################################
    # bottlenec3
    #############################################################################################################
    isFrozen = 'bottleNeck3' in freeze_layers 
    op_layer_name = BottleNeckLayer(net, ip_layer_name=op_layer_name, block_name='bot3_0' , num_input_ch=256,
        num_output_ch=512, downsample=downsample_bot[3], prePostFix=prePostFix, kwArgs=kwArgs,bn_type=bn_type, 
        ex_net=ex_net, dilation=dil_bot3_non_last,isFrozen=isFrozen)

    op_layer_name = BottleNeckLayer(net, ip_layer_name=op_layer_name, block_name='bot3_1' , num_input_ch=512,
        num_output_ch=512,prePostFix=prePostFix, kwArgs=kwArgs,bn_type=bn_type, ex_net=ex_net,
        dilation=dil_bot3_non_last, isFrozen=isFrozen)

    op_layer_name = BottleNeckLayer(net, ip_layer_name=op_layer_name, block_name='bot3_2' , num_input_ch=512,
        num_output_ch=512, prePostFix=prePostFix, kwArgs=kwArgs,bn_type=bn_type, ex_net=ex_net, 
        dilation=dil_bot3_last, isFrozen=isFrozen)


    #############################################################################################################
    # bottlenec4
    #############################################################################################################
    isFrozen = 'bottleNeck4' in freeze_layers 
    op_layer_name = BottleNeckLayer(net, ip_layer_name=op_layer_name, block_name='bot4_0', num_input_ch=512,
        num_output_ch=1024, downsample=downsample_bot[4], prePostFix=prePostFix, kwArgs=kwArgs,bn_type=bn_type, 
        ex_net=ex_net, dilation=dil_bot4_non_last, isFrozen=isFrozen, dil_due_to_stride=dil_due_to_stride)

    op_layer_name = BottleNeckLayer(net, ip_layer_name=op_layer_name, block_name='bot4_1' , num_input_ch=1024,
        num_output_ch=1024, prePostFix=prePostFix, kwArgs=kwArgs,bn_type=bn_type, ex_net=ex_net,
        dilation=dil_bot4_non_last, isFrozen=isFrozen)

    op_layer_name = BottleNeckLayer(net, ip_layer_name=op_layer_name, block_name='bot4_2' , num_input_ch=1024,
        num_output_ch=1024, prePostFix=prePostFix, kwArgs=kwArgs,bn_type=bn_type, ex_net=ex_net,
        dilation=dil_bot4_last, isFrozen=isFrozen)

    #############################################################################################################
    #layer 'fc6'
    ip_name = op_layer_name

    isFrozen = 'fc6' in freeze_layers
    net.fc6 = L.Convolution(net[ip_name], num_output=1024, pad=6,
        kernel_size=3, dilation=6, **kwArgs.kwargs_conv[isFrozen])
    net.relu6 = L.ReLU(net.fc6, in_place=True)

    #layer 'fc7'
    isFrozen = 'fc7' in freeze_layers
    net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1,
        **kwArgs.kwargs_conv[isFrozen])
    net.relu7 = L.ReLU(net.fc7, in_place=True)

    #:SN, how do I give two inputs, fc1000 and label
    #lossOp = L.Softmax(net.fc1000)

    #:SN, what in include here
    #accuracyOpTop1 = L.AccuracyLayer(net.fc1000)
    
    #accuracyOpTop5 = L.AccuracyLayer(net.fc1000,top_k=5)

    # Update freeze layers.
    #kwArgs.kwargs_conv['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    #kwArgs.kwargs_conv_grp_dil['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    #layers = net.keys()
    #for freeze_layer in freeze_layers:
    #    if freeze_layer in layers:
    #        net.update(freeze_layer, kwargs_conv)

    return net

###############################################################
def MobileNetBody(net, from_layer='data', fully_conv=False, reduced=False, dilated=False,
        dropout=True, freeze_layers=[], bn_type='bvlc', bn_at_start=True, caffe_fork='',
        training_type='SSD', depth_mul=1.0, ssd_mobile_chuanqi=False,
        dil_when_stride_removed=False, use_shuffle=False,
        heads_same_as_vgg=False, postfix_char='_'):

  assert from_layer in net.keys()
 
  #"bvlc", "nvidia"
  if caffe_fork == '':
    caffe_fork = bn_type

  if bn_type == 'none': 
    bn_at_start = False

  prePostFix = PrePostFixMobileNet(bn_type=bn_type, postfix_char=postfix_char)

  # Caffe_MobileNet does not have params in scale layer
  #param_in_sb = False
  #if ssd_mobile_chuanqi:
  #  param_in_sb = True
  param_in_sb = True
  kwArgs = KW_Args(caffe_fork=caffe_fork, bias_term=False, param_in_sb=param_in_sb, eps=0.00001, caffeForDilGrp=False)

  if ssd_mobile_chuanqi:
    block_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
  else:  
    block_labels = ['1', '2_1', '2_2', '3_1', '3_2', '4_1', '4_2', '5_1', '5_2', '5_3', '5_4', '5_5', '5_6', '6']

  #init with large number
  removed_stride_layer_idx = len(block_labels) + 2
  removed_stride_fac = 1
  if training_type == 'SSD':
    stride_list = [2,1,2,1,2,1,2,1,1,1,1,1,1,1]
    if dil_when_stride_removed:
      removed_stride_layer_idx = 12 
  else:   
    #for imagenet training
    stride_list = [2,1,2,1,2,1,2,1,1,1,1,1,2,1]

  num_dw_outputs =  [32, 32,  64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024]
  num_sep_outputs = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024,1024]

  num_dw_outputs = map(lambda x: int(round(x * depth_mul)), num_dw_outputs)
  num_sep_outputs = map(lambda x: int(round(x * depth_mul)), num_sep_outputs)

  ##################
  stage=1
  block_name = 'conv{}'.format(block_labels[0])
  isFrozen= block_name in freeze_layers
  dilation = 1
  op_layer_name = ConvBNLayerJacNet(net, from_layer, block_name, bn_type=bn_type, use_relu=True,
      num_output=num_dw_outputs[0], kernel_size=3, pad=1, stride=stride_list[0],
      prePostFix=prePostFix, kwArgs=kwArgs,isFrozen=isFrozen,dilation=dilation)

  ##################
  num_stages = len(num_dw_outputs)
  #for num_dw_output,num_sep_output, block_label in zip(num_dw_outputs, num_sep_outputs, block_labels): 
  for stg_idx in range(1,num_stages):
    num_dw_output = num_dw_outputs[stg_idx] 
    num_sep_output = num_sep_outputs[stg_idx] 
    block_label = block_labels[stg_idx] 
    stride = stride_list[stg_idx]

    ip_layer_name = op_layer_name
    block_name = 'conv{}{}dw'.format(block_label, postfix_char)

    group = num_dw_output
    num_output = num_dw_output
    if use_shuffle == True:
      group = 8
      num_output = num_dw_output

    isFrozen= block_name in freeze_layers
    op_layer_name = ConvBNLayerJacNet(net, ip_layer_name, block_name, bn_type=bn_type, use_relu=True,
        num_output=num_output, kernel_size=3, pad=1, stride=stride,  prePostFix=prePostFix, 
        kwArgs=kwArgs,isFrozen=isFrozen, group=group,dilation=dilation*removed_stride_fac,
        use_shuffle=use_shuffle)

    #when shuffle is used after 3x3 dw layer, don't use 1x1 layer 
    if use_shuffle == False:
      ip_layer_name = op_layer_name
      if ssd_mobile_chuanqi:
        block_name = 'conv{}'.format(block_label)
      else:  
        block_name = 'conv{}{}sep'.format(block_label, postfix_char)
      isFrozen= block_name in freeze_layers

      #have dilation for all layers after the layer where stride was removed
      if stg_idx >= removed_stride_layer_idx:
        removed_stride_fac = 2 
      op_layer_name = ConvBNLayerJacNet(net, ip_layer_name, block_name, bn_type=bn_type, use_relu=True,
          num_output=num_sep_output, kernel_size=1, pad=0, stride=1,  prePostFix=prePostFix, 
          kwArgs=kwArgs,isFrozen=isFrozen, group=1, dilation=dilation*removed_stride_fac)
      
      #concat layer to make 2*N channels out of two layers of N channels 
      #this is to match num ch like VGG16
      if heads_same_as_vgg:
        #block_labels[4] = "3-2"
        #block_labels[5] = "4-1"
        if op_layer_name == "conv{}{}sep{}relu".format(block_labels[5], postfix_char,postfix_char):
          net['head1'] = L.Concat(net['conv{}{}sep{}relu'.format(block_labels[5],postfix_char,postfix_char)],
              net['conv{}{}sep{}relu'.format(block_labels[4], postfix_char,postfix_char)], axis=1)

        #block_labels[13] = "6"
        #block_labels[12] = "5_6"
        if op_layer_name == "conv{}{}sep{}relu".format(block_labels[13], postfix_char,postfix_char):
          net['head2'] = L.Concat(net['conv{}{}sep{}relu'.format(block_labels[13],postfix_char,postfix_char)],
              net['conv{}{}sep{}relu'.format(block_labels[12], postfix_char,postfix_char)], axis=1)

  #############################################################################################################

  #layer 'fc6'
  #ip_name = op_layer_name

  #isFrozen = 'fc6' in freeze_layers
  #net.fc6 = L.Convolution(net[ip_name], num_output=1024, pad=6,
  #    kernel_size=3, dilation=6, **kwArgs.kwargs_conv[isFrozen])
  #net.relu6 = L.ReLU(net.fc6, in_place=True)

  ##layer 'fc7'
  #isFrozen = 'fc7' in freeze_layers
  #net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1,
  #    **kwArgs.kwargs_conv[isFrozen])
  #net.relu7 = L.ReLU(net.fc7, in_place=True)
  
  return net

###############################################################
def ResNet101Body(net, from_layer, use_pool5=True, use_dilation_conv5=False, **bn_param):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=7, pad=3, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True, **bn_param)
    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False, **bn_param)
    ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False, **bn_param)

    ResBody(net, 'res2c', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True, **bn_param)

    from_layer = 'res3a'
    for i in xrange(1, 4):
      block_name = '3b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False, **bn_param)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True, **bn_param)

    from_layer = 'res4a'
    for i in xrange(1, 23):
      block_name = '4b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False, **bn_param)
      from_layer = 'res{}'.format(block_name)

    stride = 2
    dilation = 1
    if use_dilation_conv5:
      stride = 1
      dilation = 2

    ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=stride, use_branch1=True, dilation=dilation, **bn_param)
    ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation, **bn_param)
    ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation, **bn_param)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net


def ResNet152Body(net, from_layer, use_pool5=True, use_dilation_conv5=False, **bn_param):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=7, pad=3, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True, **bn_param)
    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False, **bn_param)
    ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False, **bn_param)

    ResBody(net, 'res2c', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True, **bn_param)

    from_layer = 'res3a'
    for i in xrange(1, 8):
      block_name = '3b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False, **bn_param)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True, **bn_param)

    from_layer = 'res4a'
    for i in xrange(1, 36):
      block_name = '4b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False, **bn_param)
      from_layer = 'res{}'.format(block_name)

    stride = 2
    dilation = 1
    if use_dilation_conv5:
      stride = 1
      dilation = 2

    ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=stride, use_branch1=True, dilation=dilation, **bn_param)
    ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation, **bn_param)
    ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation, **bn_param)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net

def ResNetTenBody(net, from_layer, use_pool5=True, use_dilation_conv5=False):
    conv_prefix = ''
    conv_postfix = ''
    #bn_prefix = 'bn_'
    #bn_postfix = ''
    #scale_prefix = 'scale_'
    #scale_postfix = ''

    bn_prefix = ''
    bn_postfix = '_bn'
    scale_prefix = ''
    scale_postfix = '_scale'

    # BN/Scale before doing anything
    layer_label = 'data'
    print ("ip_layer_name: ", from_layer)
    ConvBNLayer(net, from_layer, layer_label, use_bn=True, use_relu=False,
        num_output=64, kernel_size=7, pad=3, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix,
        use_conv=False, bn_in_place=False,scale_filler=False,
        out_layer_name_ResTen=True)

    op_layer_name ='{}{}{}'.format(scale_prefix, layer_label, scale_postfix)
    print ("op_layer_name: ", op_layer_name)

    # First conv layer
    layer_label = 'conv1'
    ip_layer_name = op_layer_name
    # Conv/BN/Scale/Relu
    print ("ip_layer_name: ", ip_layer_name)
    ConvBNLayer(net, ip_layer_name, layer_label, use_bn=True, use_relu=True,
        num_output=64, kernel_size=7, pad=3, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix,
        bias_term_swap=True,scale_filler=False,
        out_layer_name_ResTen=True)

    conv_name = '{}{}{}'.format(conv_prefix, layer_label, conv_postfix)
    layer_name = '{}_relu'.format(conv_name)

    name = 'conv1_pool'
    net[name] = L.Pooling(net[layer_name], pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=2)

    block_name = 'layer_64_1'
    layer_name = 'conv1_pool'
    print ("ip_layer_name: ", layer_name)
    ResBodyTenCore(net, layer_name, block_name, out2a=64, out2b=64, out2c=64,
        stride=1, use_branch1=False)
    res_name = '{}_sum'.format(block_name)

    # BN/Scale/Relu
    print ("ip_layer_name: ", res_name)
    layer_label = 'layer_128_1'

    bn_postfix = '_bn1'
    scale_postfix = '_scale1'
    ConvBNLayer(net, res_name, layer_label, use_bn=True, use_relu=True,
        num_output=128, kernel_size=3, pad=1, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix,
        use_conv=False, bn_in_place=False,scale_filler=False,
        out_layer_name_ResTen=True)

    #conv_name = '{}{}{}'.format(conv_prefix, layer_label, conv_postfix)
    #layer_name = '{}_relu'.format(conv_name)

    bn_name = '{}{}{}'.format(bn_prefix, layer_label, bn_postfix)
    layer_name = bn_name.replace('bn','relu')

    block_name = 'layer_128_1'
    print ("ip_layer_name: ", layer_name)
    ResBodyTenCore(net, layer_name, block_name, out2a=128, out2b=128,
        out2c=128, stride=2, use_branch1=True)
    res_name = '{}_sum'.format(block_name)

    # BN/Scale/Relu
    print ("ip_layer_name: ", res_name)
    layer_label = 'layer_256_1'
    ConvBNLayer(net, res_name, layer_label, use_bn=True, use_relu=True,
        num_output=256, kernel_size=3, pad=1, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix,
        use_conv=False, bn_in_place=False,scale_filler=False,
        out_layer_name_ResTen=True)

    #conv_name = '{}{}{}'.format(conv_prefix, layer_label, conv_postfix)
    #layer_name = '{}_relu'.format(conv_name)
    bn_name = '{}{}{}'.format(bn_prefix, layer_label, bn_postfix)
    layer_name = bn_name.replace('bn','relu')

    block_name = 'layer_256_1'
    print ("ip_layer_name: ", layer_name)
    ResBodyTenCore(net, layer_name, block_name, out2a=256, out2b=256,
        out2c=256, stride=2, use_branch1=True)
    res_name = '{}_sum'.format(block_name)

    # BN/Scale/Relu
    print ("ip_layer_name: ", res_name)
    layer_label = 'layer_512_1'
    ConvBNLayer(net, res_name, layer_label, use_bn=True, use_relu=True,
        num_output=256, kernel_size=3, pad=1, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix,
        use_conv=False, bn_in_place=False,scale_filler=False,
        out_layer_name_ResTen=True)

    #conv_name = '{}{}{}'.format(conv_prefix, layer_label, conv_postfix)
    #layer_name = '{}_relu'.format(conv_name)
    
    bn_name = '{}{}{}'.format(bn_prefix, layer_label, bn_postfix)
    layer_name = bn_name.replace('bn','relu')

    block_name = 'layer_512_1'
    print ("ip_layer_name: ", layer_name)

    if use_dilation_conv5:
      stride = 1
      dilation = 2

    ResBodyTenCore(net, layer_name, block_name, out2a=512, out2b=512,
        out2c=512, stride=stride, use_branch1=True, dilation=dilation)
    res_name = '{}_sum'.format(block_name)
    print("net[-2]: ", res_name) 

    # BN/Scale/Relu
    print ("ip_layer_name: ", res_name)
    layer_label = 'last'
    bn_postfix = '_bn'
    scale_postfix = '_scale'

    ConvBNLayer(net, res_name, layer_label, use_bn=True, use_relu=True,
        num_output=256, kernel_size=7, pad=3, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix,
        use_conv=False,scale_filler=False,
        out_layer_name_ResTen=True)

    #conv_name = '{}{}{}'.format(conv_prefix, layer_label, conv_postfix)
    #layer_name = '{}_relu'.format(conv_name)
    bn_name = '{}{}{}'.format(bn_prefix, layer_label, bn_postfix)
    relu_name = bn_name.replace('bn','relu')


    print("net[-1]: ", relu_name) 
    return net


def InceptionV3Body(net, from_layer, output_pred=False, **bn_param):
  # scale is fixed to 1, thus we ignore it.
  use_scale = False

  out_layer = 'conv'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=32, kernel_size=3, pad=0, stride=2, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'conv_1'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=32, kernel_size=3, pad=0, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'conv_2'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=64, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'pool'
  net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
      kernel_size=3, stride=2, pad=0)
  from_layer = out_layer

  out_layer = 'conv_3'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=80, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'conv_4'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=192, kernel_size=3, pad=0, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'pool_1'
  net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
      kernel_size=3, stride=2, pad=0)
  from_layer = out_layer

  # inceptions with 1x1, 3x3, 5x5 convolutions
  for inception_id in xrange(0, 3):
    if inception_id == 0:
      out_layer = 'mixed'
      tower_2_conv_num_output = 32
    else:
      out_layer = 'mixed_{}'.format(inception_id)
      tower_2_conv_num_output = 64
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=48, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=64, kernel_size=5, pad=2, stride=1),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
        dict(name='conv_2', num_output=96, kernel_size=3, pad=1, stride=1),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=tower_2_conv_num_output, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  # inceptions with 1x1, 3x3(in sequence) convolutions
  out_layer = 'mixed_3'
  towers = []
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=384, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  tower_name = '{}/tower'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
      dict(name='conv_2', num_output=96, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  out_layer = '{}/join'.format(out_layer)
  net[out_layer] = L.Concat(*towers, axis=1)
  from_layer = out_layer

  # inceptions with 1x1, 7x1, 1x7 convolutions
  for inception_id in xrange(4, 8):
    if inception_id == 4:
      num_output = 128
    elif inception_id == 5 or inception_id == 6:
      num_output = 160
    elif inception_id == 7:
      num_output = 192
    out_layer = 'mixed_{}'.format(inception_id)
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_2', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_3', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_4', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  # inceptions with 1x1, 3x3, 1x7, 7x1 filters
  out_layer = 'mixed_8'
  towers = []
  tower_name = '{}/tower'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=320, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  tower_name = '{}/tower_1'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
      dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
      dict(name='conv_3', num_output=192, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  out_layer = '{}/join'.format(out_layer)
  net[out_layer] = L.Concat(*towers, axis=1)
  from_layer = out_layer

  for inception_id in xrange(9, 11):
    num_output = 384
    num_output2 = 448
    if inception_id == 9:
      pool = P.Pooling.AVE
    else:
      pool = P.Pooling.MAX
    out_layer = 'mixed_{}'.format(inception_id)
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=320, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)

    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    subtowers = []
    subtower_name = '{}/mixed'.format(tower_name)
    subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
        dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ], **bn_param)
    subtowers.append(subtower)
    subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
        dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ], **bn_param)
    subtowers.append(subtower)
    net[subtower_name] = L.Concat(*subtowers, axis=1)
    towers.append(net[subtower_name])

    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output2, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=3, pad=1, stride=1),
        ], **bn_param)
    subtowers = []
    subtower_name = '{}/mixed'.format(tower_name)
    subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
        dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ], **bn_param)
    subtowers.append(subtower)
    subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
        dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ], **bn_param)
    subtowers.append(subtower)
    net[subtower_name] = L.Concat(*subtowers, axis=1)
    towers.append(net[subtower_name])

    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=pool, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  if output_pred:
    net.pool_3 = L.Pooling(net[from_layer], pool=P.Pooling.AVE, kernel_size=8, pad=0, stride=1)
    net.softmax = L.InnerProduct(net.pool_3, num_output=1008)
    net.softmax_prob = L.Softmax(net.softmax)

  return net

def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
        use_scale=True, min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
        flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
        conf_postfix='', loc_postfix='',conf_name="mbox_conf", ssd_mobile_chuanqi=False, **bn_param):
    
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth:
            if inter_layer_depth[i] > 0:
                inter_name = "{}_inter".format(from_layer)
                ConvBNLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, lr_mult=lr_mult,
                      num_output=inter_layer_depth[i], kernel_size=3, pad=1, stride=1, **bn_param)
                from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
            if ssd_mobile_chuanqi and (max_size[0] == 0):
              num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_{}{}".format(from_layer, conf_name, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes;
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                clip=clip, variance=prior_variance, offset=offset)
       
        if ssd_mobile_chuanqi:
          if max_size and (max_size[0]>0):
              net.update(name, {'max_size': max_size})
        else:      
          if max_size:
              net.update(name, {'max_size': max_size})

        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = conf_name #"mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_objectness"
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers
