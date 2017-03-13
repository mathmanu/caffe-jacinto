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
    parser.add_argument('--input', type=str, required=True, help='Path to image folder')
    parser.add_argument('--search', type=str, default='*.png', help='Wildcard. eg. train/*/*.png')
    parser.add_argument('--output', type=str, default='output', help='Path to output folder')  
    parser.add_argument('--num_images', type=int, default=0, help='Max num images to process')            
    parser.add_argument('--model', type=str, default='', help='Path to Model prototxt')  
    parser.add_argument('--weights', type=str, default='', help='Path to pre-trained folder')      
    parser.add_argument('--crop', nargs='+', help='crop-width crop-height')      
    parser.add_argument('--resize', nargs='+', help='resize-width resize-height')   
    parser.add_argument('--blend', action='store_true', help='Do chroma belnding at output for visualization')      
    parser.add_argument('--palette', type=str, default='', help='Color palette')             
    return parser.parse_args()

    
def create_output(args):
    ext = os.path.splitext(args.output)[1]
    if (ext == '.mp4' or ext == '.MP4'):
        output_type = 'video'
    elif (ext == '.png' or ext == '.jpg' or ext == '.jpeg' or ext == '.PNG' or ext == '.JPG' or ext == '.JPEG'):
        output_type = 'image'
    else:
        output_type = 'folder'    
                        
    if os.path.exists(args.output) and os.path.isdir(args.output):
        shutil.rmtree(args.output)            

    if output_type == 'folder':
        os.mkdir(args.output)
        
    return output_type

def crop_color_image2(color_image, crop_size):  #size in (height, width)
    image_size = color_image.shape
    extra_h = (image_size[0] - crop_size[0])//2 if image_size[0] > crop_size[0] else 0
    extra_w = (image_size[1] - crop_size[1]) // 2 if image_size[1] > crop_size[1] else 0
    out_image = color_image[0:(0+crop_size[0]), 0:(0+crop_size[1]), :]
    return out_image

def crop_gray_image2(color_image, crop_size):   #size in (height, width)
    image_size = color_image.shape
    extra_h = (image_size[0] - crop_size[0])//2 if image_size[0] > crop_size[0] else 0
    extra_w = (image_size[1] - crop_size[1])//2 if image_size[1] > crop_size[1] else 0
    out_image = color_image[0:(0+crop_size[0]), 0:(0+crop_size[1])]
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

def infer_blob(args, net, input_bgr):
    image_size = input_bgr.shape  
    if args.crop:
	print('Croping to ' + str(args.crop))
	input_bgr = crop_color_image2(input_bgr, (args.crop[1], args.crop[0]))

    if args.resize:
	print('Resizing to ' + str(args.resize))
        input_bgr = resize_image(input_bgr, (args.resize[1], args.resize[0]))

    input_blob = input_bgr.transpose((2, 0, 1))    #Interleaved to planar
    input_blob = input_blob[np.newaxis, ...]
    if net.blobs['data'].data.shape != input_blob.shape:
        net.blobs['data'].data.reshape(input_blob.shape)
    
    blobs = None #['prob', 'argMaxOut']
    out = net.forward_all(blobs=blobs, **{net.inputs[0]: input_blob})

    if 'argMaxOut' in out:
        prob = out['argMaxOut'][0]
        prediction = prob[0].astype(int)
    else:   
        prob = out['prob'][0]
        prediction = np.argmax(prob.transpose([1, 2, 0]), axis=2)
          
    if args.blend:
        prediction_size = (prediction.shape[0], prediction.shape[1], 3)    
        output_image = args.palette[prediction.ravel()].reshape(prediction_size)
        output_image = crop_color_image2(output_image, image_size)    
        output_image = chroma_blend(input_bgr, output_image)            
    else:           
        prediction_size = (prediction.shape[0], prediction.shape[1])
        output_image = prediction.ravel().reshape(prediction_size)
        output_image = crop_gray_image2(output_image, image_size)
    return output_image
 
                               
def infer_image_file(args, net):
    input_blob = cv2.imread(args.input)
    output_blob = infer_blob(args, net, input_blob)  
    cv2.imwrite(args.output, output_blob)
    return
            
def infer_image_folder(args, net):
    image_indices = []    
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
        output_blob = infer_blob(args, net, input_blob)  
        output_name = os.path.join(args.output, os.path.basename(input_name));
        cv2.imwrite(output_name, output_blob)        
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

    output_type = create_output(args)
    
    caffe.set_mode_gpu()
    caffe.set_device(0)
    
    net = caffe.Net(args.model, args.weights, caffe.TEST)
            
    if output_type == 'image':
        print('Infering Images')
        infer_image_file(args, net)        
    elif output_type == 'folder':
        print('Infering Folder')    
        infer_image_folder(args, net)
    elif output_type == 'video':
        print('Infering Video')      
        infer_video(args, net)     
    else:   
        print('Incorrect options')

if __name__ == "__main__":
    main()
