#!/usr/bin/env python

from __future__ import print_function

import sys
import numpy as np
import os, glob
import cv2
import caffe
#import lmdb
from PIL import Image
import argparse
import random
import shutil
import imageio
import math

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to image folder, video or list txt file')
    parser.add_argument('--label', type=str, default=None, help='Path to label folder, video or list txt file')    
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')    
    parser.add_argument('--search', type=str, default='*.png', help='Wildcard. eg. train/*/*.png')
    parser.add_argument('--output', type=str, default=None, help='Path to output folder')  
    parser.add_argument('--num_images', type=int, default=0, help='Max num images to process')            
    parser.add_argument('--model', type=str, default='', help='Path to Model prototxt')  
    parser.add_argument('--weights', type=str, default='', help='Path to pre-trained folder')      
    parser.add_argument('--crop', nargs='+', help='crop-width crop-height')      
    parser.add_argument('--resize', nargs='+', help='resize-width resize-height')   
    parser.add_argument('--blend', action='store_true', help='Do chroma belnding at output for visualization')      
    parser.add_argument('--palette', type=str, default='', help='Color palette')   
    parser.add_argument('--batch_size', type=int, default=1, help='Batch of images to process')   
    parser.add_argument('--resize_back', action="store_true", help='Upsize back a resized label for evaluation')
    parser.add_argument('--label_dict', type=str, default='', help='Lookup to be applied to prediction to match with gt labels')    
    parser.add_argument('--class_dict', type=str, default='', help='Grouping of classes in evaluation. Also ignored classea')   
    return parser.parse_args()

def create_lut(label_dict):
    if label_dict:
        lut = np.zeros(256, dtype=np.uint8)
        for k in range(256):
            lut[k] = k
        for k in label_dict.keys():
            lut[k] = label_dict[k] 
        return lut
    else:
        return None
        
def check_paths(args):
    output_type = None
    if args.output:
      ext = os.path.splitext(args.output)[1]
      if (ext == '.mp4' or ext == '.MP4'):
        output_type = 'video'
      elif (ext == '.png' or ext == '.jpg' or ext == '.jpeg' or ext == '.PNG' or ext == '.JPG' or ext == '.JPEG'):
        output_type = 'image'
      elif (ext == '.txt'):
        output_type = 'list'        
      else:
        output_type = 'folder'    
      if os.path.exists(args.output) and os.path.isdir(args.output):
        shutil.rmtree(args.output)            
      if output_type == 'folder':
        os.mkdir(args.output)
                     
    ext = os.path.splitext(args.input)[1]
    if (ext == '.mp4' or ext == '.MP4'):
        input_type = 'video'
    elif (ext == '.png' or ext == '.jpg' or ext == '.jpeg' or ext == '.PNG' or ext == '.JPG' or ext == '.JPEG'):
        input_type = 'image'
    elif (ext == '.txt'):
        input_type = 'list'          
    else:
        input_type = 'folder'  
                                
    return input_type, output_type

def crop_color_image2(color_image, crop_size):  #size in (height, width)
    image_size = color_image.shape
    extra_h = (image_size[0] - crop_size[0])//2 if image_size[0] > crop_size[0] else 0
    extra_w = (image_size[1] - crop_size[1]) // 2 if image_size[1] > crop_size[1] else 0
    out_image = color_image[extra_h:(extra_h+crop_size[0]), extra_w:(extra_w+crop_size[1]), :]
    return out_image

def crop_gray_image2(color_image, crop_size):   #size in (height, width)
    image_size = color_image.shape
    extra_h = (image_size[0] - crop_size[0])//2 if image_size[0] > crop_size[0] else 0
    extra_w = (image_size[1] - crop_size[1])//2 if image_size[1] > crop_size[1] else 0
    out_image = color_image[extra_h:(extra_h+crop_size[0]), extra_w:(extra_w+crop_size[1])]
    return out_image
    
def chroma_blend(image, color):
    image_yuv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2YUV)
    image_y,image_u,image_v = cv2.split(image_yuv)
    color_yuv = cv2.cvtColor(color.astype(np.uint8), cv2.COLOR_BGR2YUV)
    color_y,color_u,color_v = cv2.split(color_yuv)
    image_y = np.uint8(image_y)
    color_u = np.uint8(color_u)
    color_v = np.uint8(color_v)
    image_yuv = cv2.merge((image_y,color_u,color_v))
    image = cv2.cvtColor(image_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
    return image
        
def resize_image(color_image, size): #size in (height, width)
    im = Image.fromarray(color_image)
    im = im.resize((size[1], size[0]), Image.ANTIALIAS) #(width, height)
    im = np.array(im, dtype=np.uint8)
    return im

def resize_label(label_image, size): #size in (height, width)
    im = Image.fromarray(label_image.astype(np.uint8))
    im = im.resize((size[1], size[0]), Image.NEAREST) #(width, height)
    im = np.array(im, dtype=np.uint8)
    return im
    
def infer_blob(args, net, input_bgr, input_label=None):
    input_bgr_orig = np.array(input_bgr, np.uint8)
    image_size = input_bgr.shape  
    if args.crop:
        print('Croping to ' + str(args.crop))
        input_bgr = crop_color_image2(input_bgr, (args.crop[1], args.crop[0]))
        if (input_label is not None):
          input_label = crop_color_image2(input_label, (args.crop[1], args.crop[0]))

    if args.resize:
        print('Resizing to ' + str(args.resize))
        input_bgr = resize_image(input_bgr, (args.resize[1], args.resize[0]))
        if (input_label is not None) and (not args.resize_back):
          input_label = resize_label(input_label, (args.resize[1], args.resize[0]))  
                
    input_blob = input_bgr.transpose((2, 0, 1))    #Interleaved to planar
    input_blob = input_blob[np.newaxis, ...]
    if net.blobs['data'].data.shape != input_blob.shape:
        #net.blobs['data'].data.reshape(input_blob.shape)
        raise ValueError("Pleae correct the input size in deploy prototxt to match with the input size given here: "+str(input_blob.shape))
    
    blobs = None #['prob', 'argMaxOut']
    out = net.forward_all(blobs=blobs, **{net.inputs[0]: input_blob})

    if 'argMaxOut' in out:
        prob = out['argMaxOut'][0]
        prediction = prob[0].astype(int)
    else:   
        prob = out['prob'][0]
        prediction = np.argmax(prob.transpose([1, 2, 0]), axis=2)
          
    if args.label_dict:
        prediction = args.label_lut[prediction]
        if input_label is not None:
          input_label = args.label_lut[input_label]
        
    if args.resize and args.resize_back:
       prediction = resize_label(prediction, image_size)
       input_bgr = input_bgr_orig
                        
    if args.blend:
        prediction_size = (prediction.shape[0], prediction.shape[1], 3)    
        output_image = args.palette[prediction.ravel()].reshape(prediction_size)
        output_image = crop_color_image2(output_image, image_size)    
        output_image = chroma_blend(input_bgr, output_image)            
    else:           
        prediction_size = (prediction.shape[0], prediction.shape[1])
        output_image = prediction.ravel().reshape(prediction_size)
        output_image = crop_gray_image2(output_image, image_size)
    return output_image, input_label
 
                               
def infer_image_file(args, net):
    input_blob = cv2.imread(args.input)
    output_blob = infer_blob(args, net, input_blob)  
    cv2.imwrite(args.output, output_blob)
    return
            
def infer_image_folder(args, net):  
    print('Getting list of images...', end='')
    image_search = os.path.join(args.input, args.search)
    input_indices = glob.glob(image_search) 
    numFrames = min(len(input_indices), args.num_images)    
    input_indices = input_indices[:numFrames]
    input_indices.sort()
    print('running inference for ', len(input_indices), ' images...');
    for input_name in input_indices:
        print(input_name, end=' ')   
        sys.stdout.flush()         
        input_blob = cv2.imread(input_name)  
        output_blob, _ = infer_blob(args, net, input_blob)  
        output_name = os.path.join(args.output, os.path.basename(input_name));
        cv2.imwrite(output_name, output_blob)
    return
    
def eval_blob(args, net, input_blob, label_blob, confusion_matrix):
    output_blob, label_blob = infer_blob(args, net, input_blob, label_blob)
        
    #for r in range(output_blob.shape[0]):
    #  for c in range(output_blob.shape[1]):
    #    gt_label = label_blob[r][c][0]
    #    det_label = output_blob[r][c]
    #    det_label = min(det_label, args.num_classes)
    #    if gt_label != 255:
    #      confusion_matrix[gt_label][det_label] += 1

    if len(label_blob.shape)>2:
        label_blob = label_blob[:,:,0]        
    gt_labels = label_blob.ravel()
    det_labels = output_blob.ravel().clip(0,args.num_classes)
    gt_labels_valid_ind = np.where(gt_labels != 255)
    gt_labels_valid = gt_labels[gt_labels_valid_ind]
    det_labels_valid = det_labels[gt_labels_valid_ind]
    #print(len(np.where(gt_labels_valid==det_labels_valid)[0]))
    #confusion_matrix[gt_labels_valid][det_labels_valid] += 1
    for r in range(confusion_matrix.shape[0]):
        for c in range(confusion_matrix.shape[1]):
            confusion_matrix[r,c] += np.sum((gt_labels_valid==r) & (det_labels_valid==c))

    return output_blob, confusion_matrix
    
    
def compute_accuracy(args, confusion_matrix):
    if args.class_dict:
      selected_classes = []
      for cls in args.class_dict:
        category = args.class_dict[cls]
        if category >= 0 and category<255:
          selected_classes.extend([category])   
      num_selected_classes = max(selected_classes) + 1
      print('num_selected_classes={}'.format(num_selected_classes))        
      tp = np.zeros(num_selected_classes)
      population = np.zeros(num_selected_classes)
      det = np.zeros(num_selected_classes)
      iou = np.zeros(num_selected_classes)
      for r in range(args.num_classes):
        for c in range(args.num_classes):   
          r0 = args.class_dict[r]
          c0 = args.class_dict[c]        
          if r0 >= 0 and r0 < 255:
            population[r0] += confusion_matrix[r][c]
            if c0 >= 0 and c0 < 255:
              det[c0] += confusion_matrix[r][c]   
            if r == c:
              tp[r0] += confusion_matrix[r][c]    
    else:
      num_selected_classes = args.num_classes
      tp = np.zeros(args.num_classes)
      population = np.zeros(args.num_classes)
      det = np.zeros(args.num_classes)
      iou = np.zeros(args.num_classes)
      for r in range(args.num_classes):
        for c in range(args.num_classes):   
          population[r] += confusion_matrix[r][c]
          det[c] += confusion_matrix[r][c]   
          if r == c:
            tp[r] += confusion_matrix[r][c]
           
    tp_total = 0
    population_total = 0           
    for cls in range(num_selected_classes):
      intersection = tp[cls]
      union = population[cls] + det[cls] - tp[cls]
      iou[cls] = (intersection / union) if union else 0
      tp_total += tp[cls]
      population_total += population[cls]
    
    #print('confusion_matrix={}'.format(confusion_matrix))
    accuracy = tp_total / population_total
     
    num_nonempty_classes = 0 
    for pop in population:
      if pop>0:
        num_nonempty_classes += 1
      
    mean_iou = np.sum(iou) / num_nonempty_classes
    return accuracy, mean_iou, iou
      
          
def infer_image_list(args, net):
    input_indices = []
    label_indices = []  
    print('Getting list of images...', end='')
    with open(args.input) as image_list_file:
      for img_name in image_list_file:
        input_indices.extend([img_name.strip()])
        
    if args.label:
      with open(args.label) as label_list_file:
        for label_name in label_list_file:
          label_indices.extend([label_name.strip()])
          
    if args.num_images:
      input_indices = input_indices[0:min(len(input_indices),args.num_images)]   
      label_indices = label_indices[0:min(len(label_indices),args.num_images)]  
                    
    print('running inference for ', len(input_indices), ' images...');
    output_name_list = []
    if not label_indices:
      for input_name in input_indices:
        print(input_name)   
        sys.stdout.flush()         
        input_blob = cv2.imread(input_name)  
        output_blob,_ = infer_blob(args, net, input_blob)  
        output_name = os.path.join(args.output, os.path.basename(input_name));
        cv2.imwrite(output_name, output_blob) 
        if args.output:        
          output_name_list.append(output_name)    
    else:
      confusion_matrix = np.zeros((args.num_classes, args.num_classes+1))
      total = len(input_indices)
      count = 0
      for (input_name, label_name) in zip(input_indices, label_indices):
        input_name_base = os.path.split(input_name)[-1]
        label_name_base = os.path.split(label_name)[-1]       
        progress = count * 100 / total 
        print((input_name_base, label_name_base, progress))   
        sys.stdout.flush()         
        input_blob = cv2.imread(input_name)  
        label_blob = cv2.imread(label_name) 
        output_blob, confusion_matrix = eval_blob(args, net, input_blob, label_blob, confusion_matrix)  
        if args.output:
          output_name = os.path.join(args.output, os.path.basename(input_name));
          cv2.imwrite(output_name, output_blob) 
          output_name_list.append(output_name)           
        count += 1
                  
      accuracy, mean_iou, iou = compute_accuracy(args, confusion_matrix)   

      print('pixel_accuracy={}, mean_iou={}, iou={}'.format(accuracy, mean_iou, iou))
      
    if args.output:        
      with open(os.path.join(args.output,"output_name_list.txt"), "w") as output_name_list_file:
        print(output_name_list)
        output_name_list_file.write('\n'.join(str(line) for line in output_name_list))
    return
        
def infer_video(args, net):
    videoIpHandle = imageio.get_reader(args.input, 'ffmpeg')
    fps = math.ceil(videoIpHandle.get_meta_data()['fps'])
    print(videoIpHandle.get_meta_data())
    numFrames = min(len(videoIpHandle), args.num_images)
    videoOpHandle = imageio.get_writer(args.output,'ffmpeg', fps=fps)
    for num in range(numFrames):
        print(num, end=' ')
        sys.stdout.flush()
        input_blob = videoIpHandle.get_data(num)   
        input_blob = input_blob[...,::-1]    #RGB->BGR
        output_blob = infer_blob(args, net, input_blob)     
        output_blob = output_blob[...,::-1]  #BGR->RGB            
        videoOpHandle.append_data(output_blob)     
    videoOpHandle.close()        
    return
        

def main(): 
    args = get_arguments()
    print(args)
    if args.num_images == 0:
        args.num_images = sys.maxsize
    
    os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'
    
    if args.palette:
        print('Creating palette')
        exec('palette='+args.palette)
        args.palette = np.zeros((256,3))
        for i, p in enumerate(palette):
        	args.palette[i,0] = p[0]
        	args.palette[i,1] = p[1]
        	args.palette[i,2] = p[2]
        args.palette = args.palette[...,::-1] #RGB->BGR, since palette is expected to be given in RGB format
    
    if args.crop and int(args.crop[0]) != 0:
        args.crop = [int(entry) for entry in args.crop]
    else:
        args.crop = None

    if args.resize and int(args.resize[0]) != 0:
        args.resize = [int(entry) for entry in args.resize]
    else:
        args.resize = None

    input_type, output_type = check_paths(args)
    
    if args.label and args.blend:
      raise ValueError('When doing evaluation by specifying --label, --blend should not be used')
        
    args.label_lut = []  
    if args.label_dict:
      label_dict_string = 'label_dict = ' + args.label_dict
      exec(label_dict_string)
      args.label_dict = label_dict
      print(args.label_dict)    
      args.label_lut = create_lut(args.label_dict)
              
    if args.class_dict:
      class_dict_string = 'class_dict = ' + args.class_dict
      exec(class_dict_string)
      args.class_dict = class_dict
                        
    caffe.set_mode_gpu()
    caffe.set_device(0)
    
    net = caffe.Net(args.model, args.weights, caffe.TEST)
            
    if input_type == 'image':
        print('Infering Images')
        infer_image_file(args, net)        
    elif input_type == 'folder':
        print('Infering Folder')    
        infer_image_folder(args, net)
    elif input_type == 'video':
        print('Infering Video')      
        infer_video(args, net)   
    elif input_type == 'list':
        print('Infering list')      
        infer_image_list(args, net)             
    else:   
        print('Incorrect options')
    

if __name__ == "__main__":
    main()
