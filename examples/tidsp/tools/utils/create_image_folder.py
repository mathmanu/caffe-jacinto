#!/usr/bin/env python

from __future__ import print_function

import sys
import numpy as np
import os, glob
#import caffe
#import lmdb
from PIL import Image
import argparse
import random
import shutil

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', action="store_true", help='Whether the input images are labels')
    parser.add_argument('--list_file', type=str, help='Path to a file containing list of images')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--search_string', type=str, default='*.png', help='Wildcard. eg. train/*/*.png')
    parser.add_argument('--output_dir', type=str, default='label-folder', help='Path to output folder')    
    parser.add_argument('--label_dict', type=str, default=None, help='Label type translation. eg. {17:0, 19:1}')
    parser.add_argument('--width', type=int, default=None, help='Output Image Width')
    parser.add_argument('--height', type=int, default=None, help='Output Image Height')    
    parser.add_argument('--rand_seed', type=int, default=0, help='Rand seed for shuffling')   
    parser.add_argument('--shuffle', action="store_true", help='Shuffle list of images')        
    return parser.parse_args()

def create_lut(args):
    if args.label_dict:
        lut = np.zeros(256, dtype=np.uint8)
        for k in range(256):
            lut[k] = k
        for k in args.label_dict.keys():
            lut[k] = args.label_dict[k] 
        return lut
    else:
        return None
    
def create_folder(args, image_indices):
    if args.label_dict:
        lut = create_lut(args)
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)
    for in_idx, in_ in enumerate(image_indices):  
        print('{} {} '.format(in_idx, in_), end='')           
        im = Image.open(in_) # or load whatever ndarray you need                      
        if args.label:
            im = np.array(im, dtype=np.uint8)     
            shape_orig = im.shape                      
            if args.height and args.width:
                im = Image.fromarray(im, 'P')                
                im = im.resize([args.height, args.width], Image.NEAREST)
            im = np.array(im, dtype=np.uint8)   
            if args.label_dict:
                im = lut[im]                    
            #im = im[np.newaxis, ...]                       
        else:
            im_orig = np.array(im, dtype=np.uint8)  
            shape_orig = im_orig.shape                             
            if args.height and args.width:
                im = im.resize([args.height, args.width], Image.ANTIALIAS)
            im = np.array(im, dtype=np.uint8)
            #im = im[:,:,::-1]          #RGB to BGR
            #im = im.transpose((2,0,1)) #Channel x Height x Width order (switch from H x W x C)
                 
        print(shape_orig)                        
        out_path = os.path.join(args.output_dir, os.path.basename(in_))                              
        im = Image.fromarray(im)
        im.save(out_path)


def main(): 
    args = get_arguments()
    print(args)
    
    if args.label_dict:
        label_dict_string = 'label_dict = ' + args.label_dict
        exec(label_dict_string)
        args.label_dict = label_dict
        print(args.label_dict)
         
    image_indices = []    
    if args.list_file:
        print('Reading image list file...', end='')
        with open(args.list_file) as f:
            image_list = f.readlines()
        for f in image_list:
            f =  f.strip()
            list_ext = os.path.splitext(f)[1]
            search_ext = os.path.splitext(args.search_string)[1]
            image_path = os.path.join(args.image_dir, f)        
            if not list_ext:
                image_path = image_path + search_ext
            image_indices.append(image_path)          
    else:
        print('Getting list of images...', end='')
        image_search = os.path.join(args.image_dir, args.search_string)
        image_indices = glob.glob(image_search)    
    
    if args.rand_seed:
        np.random.seed(args.rand_seed)
    
    if args.shuffle:
        np.random.shuffle(image_indices)
    
    print('done')
    
    create_folder(args, image_indices)

if __name__ == "__main__":
    print('Starting...')
    main()
