from __future__ import print_function
import caffe
from google.protobuf import text_format
import ast
from models.model_libs import *
import models.jacintonet_v2

import math
import os
import shutil
import stat
import subprocess
import sys
import argparse
from collections import OrderedDict

def get_arguments():
    parser = argparse.ArgumentParser()   
    parser.add_argument('--config_param', type=str, default=None, help='Extra config parameters') 	   
    parser.add_argument('--solver_param', type=str, default=None, help='Extra solver parameters')        
    return parser.parse_args()
      
def main(): 
    args = get_arguments()
   
    if args.solver_param != None:
      args.solver_param = ast.literal_eval(args.solver_param) 
            
    if args.config_param != None:
      args.config_param = ast.literal_eval(args.config_param) 
            
    #Start populating config_param
    config_param = OrderedDict()
	
    #Names
    config_param.config_name = 'image_classification'
    config_param.model_name = "jacintonet11"
    config_param.dataset = "nodataset"    
    config_param.pretrain_model = None
                          
    ### Modify the following parameters accordingly ###
    # The directory which contains the caffe code.
    # We assume you are running the script at the CAFFE_ROOT.
    config_param.caffe_root = os.getcwd()
    
    # Set true if you want to start training right after generating all files.
    config_param.run_soon = True
    # Set true if you want to load from most recently saved snapshot.
    # Otherwise, we will load from the pretrain_model defined below.
    config_param.resume_training = True
    # If true, Remove old model files.
    config_param.remove_old_models = False

    config_param.crop_size = 224
    config_param.image_width = 224
    config_param.image_height = 224

    config_param.train_data = "./data/ilsvrc12_train_lmdb" 
    config_param.test_data = "./data/ilsvrc12_val_lmdb"

    config_param.stride_list = [2,2,2,2,2]
    config_param.dilation_list = [1,1,1,1,1]
    	
    config_param.mean_value = 128 #used in a bias layer in the net.
	    
    # Setup Default values
    # If true, use batch norm for all newly added layers.
    # Currently only the non batch norm version has been tested.
    config_param.use_batchnorm = True

    # Stores LabelMapItem.
    config_param.label_map_file = ""
    # Defining which GPUs to use.
    config_param.gpus = "0,1" #gpus = "0"    
    
    config_param.num_output = 1000
    config_param.batch_size = 128
    config_param.accum_batch_size = 256

    # Which layers to freeze (no backward) during training.
    config_param.freeze_layers = []
                            
    # Evaluate on whole test set.
    config_param.num_test_image = 50000
    config_param.test_batch_size = 50
    config_param.test_batch_size_in_proto = config_param.test_batch_size      
    
    #Update from params given from outside
    #if args.config_param != None:
    #  config_param.update(args.config_param)   
    if args.config_param != None: 
      for k in args.config_param.keys():
        config_param.__setattr__(k,args.config_param[k])
        config_param.__setitem__(k,args.config_param[k])		
              
    # Modify the job name if you want.
    config_param.base_name = config_param.config_name
    config_param.job_name = config_param.base_name
                  
    # Base dir
    config_param.base_dir = config_param.job_name
    # Directory which stores the model .prototxt file.
    config_param.save_dir = config_param.job_name
    # Directory which stores the snapshot of models.
    config_param.snapshot_dir = config_param.job_name
    # Directory which stores the job script and log file.
    config_param.job_dir = config_param.job_name
    # Directory which stores the detection results.
    config_param.output_result_dir = "training/{}".format(config_param.job_name)

    # model definition files.
    config_param.train_net_file = "{}/train.prototxt".format(config_param.save_dir)
    config_param.test_net_file = "{}/test.prototxt".format(config_param.save_dir)
    config_param.deploy_net_file = "{}/deploy.prototxt".format(config_param.save_dir)
    config_param.solver_file = "{}/solver.prototxt".format(config_param.save_dir)
    # snapshot prefix.
    config_param.snapshot_prefix = "{}/{}_{}".format(config_param.snapshot_dir, config_param.model_name, config_param.dataset)
    # job script path.
    config_param.job_file_base = "{}/{}".format(config_param.job_dir, 'train')
    config_param.log_file = "{}.log".format(config_param.job_file_base)    
    config_param.job_file = "{}.sh".format(config_param.job_file_base)

    # Stores the test image names and sizes. Created by data/VOC0712/create_list.sh
    config_param.name_size_file = ""

    # Solver parameters.
    config_param.gpulist = config_param.gpus.split(",")
    config_param.num_gpus = len(config_param.gpulist)

    # Divide the mini-batch to different GPUs.
	# In BVLC caffe, this has to be divided by num GPUs - not required in NVIDIA/caffe
    config_param.train_batch_size_in_proto = config_param.batch_size 
                  
    #Solver params                   
    solver_param = {
        # Train parameters
        'type': "SGD",
        'base_lr': 1e-1,
        'max_iter': 320000,        
        'weight_decay': 0.0001,
        'lr_policy': "poly",
        'power': 1,
        'gamma': 0.1,
        'momentum': 0.9,
        'iter_size': int(math.ceil(config_param.accum_batch_size/config_param.batch_size)),
        'snapshot': 10000,
        'display': 100,
        #'average_loss': 10,
        'solver_mode': P.Solver.GPU if (config_param.num_gpus > 0) else P.Solver.CPU,
        'device_id': int(config_param.gpulist[0]) if (config_param.num_gpus > 0) else 0,
        'debug_info': False,
        'snapshot_after_train': True,
        # Test parameters
        'test_iter': [int(math.ceil(config_param.num_test_image/config_param.test_batch_size))],
        'test_interval': 1000,
        'test_initialization': True,
        'random_seed': 33,
        }

    #if args.solver_param != None:
    #  solver_param.update(args.solver_param)       
    if args.solver_param != None: 
      for k in args.solver_param.keys():
        solver_param.__setitem__(k,args.solver_param[k])	  
        #solver_param.__setattr__(k,args.solver_param[k])
		
    config_param.train_transform_param = {
            'mirror': True,
            'mean_value': [0, 0, 0],
            'crop_size': config_param.crop_size
            }
    config_param.test_transform_param = {
            'mirror': False,
            'mean_value': [0, 0, 0],
            'crop_size': config_param.crop_size
            }
						
    ### Hopefully you don't need to change the following ###
    # Check file.
    check_if_exist(config_param.train_data)
    check_if_exist(config_param.test_data)
    check_if_exist(config_param.label_map_file)
    if config_param.pretrain_model != None:    
      check_if_exist(config_param.pretrain_model)
    
    make_if_not_exist(config_param.base_dir)    
    make_if_not_exist(config_param.save_dir)
    make_if_not_exist(config_param.job_dir)
    make_if_not_exist(config_param.snapshot_dir)
          
    #----------------------
    #Net definition  
    def define_net(phase):
        # Create train net.
        net = caffe.NetSpec()
          
        #if you want the train and test in same proto, 
        #get the proto string for the data layer in train phase seperately and return it
          
        train_proto_str = []
        if phase=='train':                 
          data_kwargs = { 'source': config_param.train_data, 'name': 'data', 'batch_size': config_param.train_batch_size_in_proto, 'backend': caffe_pb2.DataParameter.DB.Value('LMDB'), 'ntop':2 }          
          net['data'], net['label'] = L.Data(transform_param=config_param.train_transform_param, **data_kwargs)
          out_layer = 'data' 
        elif phase=='test':
          data_kwargs = { 'source': config_param.test_data, 'name': 'data', 'batch_size': config_param.test_batch_size_in_proto, 'backend': caffe_pb2.DataParameter.DB.Value('LMDB'), 'ntop':2 }        
          net['data'], net['label'] = L.Data(transform_param=config_param.test_transform_param,**data_kwargs)
          out_layer = 'data'
        elif phase=='deploy':
          net['data'] = L.Input(shape=[dict(dim=[1, 3, config_param.image_height, config_param.image_width])])
          out_layer = 'data'
                         
        bias_kwargs = { #fixed value with lr_mult=0
            'param': [dict(lr_mult=0, decay_mult=0)],
            'filler': dict(type='constant', value=(-config_param.mean_value)),
            }       
        net['data/bias'] = L.Bias(net[out_layer], in_place=False, **bias_kwargs)
        out_layer = 'data/bias'
                            
        if config_param.model_name == 'jacintonet11':
            out_layer = models.jacintonet_v2.jacintonet11(net, from_layer=out_layer,\
            num_output=config_param.num_output,stride_list=config_param.stride_list,dilation_list=config_param.dilation_list,\
            freeze_layers=config_param.freeze_layers)
        else:
            ValueError("Invalid model name")

        if phase=='train' or phase=='test':  
            net["loss"] = L.SoftmaxWithLoss(net[out_layer], net['label'],
                propagate_down=[True, False])

            net["accuracy/top1"] = L.Accuracy(net[out_layer], net['label'],
                include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        
            accuracy_param_top5 = { 'top_k': 5 }            
            net["accuracy/top5"] = L.Accuracy(net[out_layer], net['label'],
                accuracy_param=accuracy_param_top5, include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        elif phase=='deploy':
            net['prob'] = L.Softmax(net[out_layer]) 
                 
        return net
    #----------------------
              
    net = define_net(phase='train')
    with open(config_param.train_net_file, 'w') as f:
        print('name: "{}_train"'.format(config_param.model_name), file=f)
        #if you want the train and test in same proto, 
        #get the proto string for the data layer, train phase.
        #print(train_proto_str, file=f) 
        print(net.to_proto(verbose=False), file=f)
    if config_param.save_dir!=config_param.job_dir:        
      shutil.copy(config_param.train_net_file, config_param.job_dir)

    # Create test net.
    net = define_net(phase='test')
    with open(config_param.test_net_file, 'w') as f:
        print('name: "{}_test"'.format(config_param.model_name), file=f)
        print(net.to_proto(verbose=False), file=f)
    if config_param.save_dir!=config_param.job_dir:
      shutil.copy(config_param.test_net_file, config_param.job_dir)

    # Create deploy net.
    deploy_net = define_net(phase='deploy')
    with open(config_param.deploy_net_file, 'w') as f:
        net_param = deploy_net.to_proto(verbose=False)
        # Remove the few layers first
        del net_param.layer[0]
        #del net_param.layer[-1]
        #del net_param.layer[-1]    
        #del net_param.layer[-1]          
        net_param.name = '{}_deploy'.format(config_param.model_name)
        net_param.input.extend(['data'])
        net_param.input_shape.extend([
            caffe_pb2.BlobShape(dim=[1, 3, config_param.image_height, config_param.image_width])])
        print(net_param, file=f)
    if config_param.save_dir!=config_param.job_dir:        
      shutil.copy(config_param.deploy_net_file, config_param.job_dir)

    # Create solver.
    solver = caffe_pb2.SolverParameter(
            train_net=config_param.train_net_file,
            test_net=[config_param.test_net_file],
            snapshot_prefix=config_param.snapshot_prefix,
            **solver_param)
            
    with open(config_param.solver_file, 'w') as f:
        print(solver, file=f)
    if config_param.save_dir!=config_param.job_dir:        
      shutil.copy(solver_file, job_dir)

    max_iter = 0
    # Find most recent snapshot.
    for file in os.listdir(config_param.snapshot_dir):
      if file.endswith(".solverstate"):
        basename = os.path.splitext(file)[0]
        iter = int(basename.split("{}_iter_".format(config_param.model_name))[1])
        if iter > max_iter:
          max_iter = iter

    train_src_param = None
    if config_param.pretrain_model != None:
      train_src_param = '--weights="{}" \\\n'.format(config_param.pretrain_model)
    if config_param.resume_training:
      if max_iter > 0:
        train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(config_param.snapshot_prefix, max_iter)

    if config_param.remove_old_models:
      # Remove any snapshots smaller than max_iter.
      for file in os.listdir(config_param.snapshot_dir):
        if file.endswith(".solverstate"):
          basename = os.path.splitext(file)[0]
          iter = int(basename.split("{}_iter_".format(config_param.model_name))[1])
          if max_iter > iter:
            os.remove("{}/{}".format(config_param.snapshot_dir, file))
        if file.endswith(".caffemodel"):
          basename = os.path.splitext(file)[0]
          iter = int(basename.split("{}_iter_".format(config_param.model_name))[1])
          if max_iter > iter:
            os.remove("{}/{}".format(config_param.snapshot_dir, file))

    # Create job file.
    with open(config_param.job_file, 'w') as f:
      f.write('cd {}\n'.format(config_param.caffe_root))
      f.write('../../build/tools/caffe train \\\n')
      f.write('--solver="{}" \\\n'.format(config_param.solver_file))
      if train_src_param != None:
        f.write(train_src_param)
      if solver_param['solver_mode'] == P.Solver.GPU:
        f.write('--gpu "{}" 2>&1 | tee {}\n'.format(config_param.gpus, config_param.log_file))
      else:
        f.write('2>&1 | tee {}\n'.format(config_param.log_file))

    # Copy the python script to job_dir.
    py_file = os.path.abspath(__file__)
    shutil.copy(py_file, config_param.job_dir)

    # Run the job.
    os.chmod(config_param.job_file, stat.S_IRWXU)
    if config_param.run_soon:
      subprocess.call(config_param.job_file, shell=True)
  
  
if __name__ == "__main__":
  main()  
