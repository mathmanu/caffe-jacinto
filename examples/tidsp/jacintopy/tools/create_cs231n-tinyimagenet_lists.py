#!/usr/bin/env python

import os
import sys
import glob

dataset_path = '/data/hdd/datasets/object-detect/other/ilsvrc/cs231n-tinyimagenet'

#Train
category_names = glob.glob('train/n*')
category_names = [f.split(os.sep)[1] for f in category_names]
category_names.sort()

train_files = glob.glob('train/*/images/*.JPEG')
train_files = [os.path.join(*(f.split(os.sep)[1:])) for f in train_files]

train_category_names = [f.split(os.sep)[0] for f in train_files]
train_category_nums = [category_names.index(f) for f in train_category_names]

with open(os.path.join(dataset_path,'train.txt'), 'w') as fp:
    for i, f in enumerate(train_files):
        idx = train_category_nums[i]
        line = f+' '+str(idx)+'\n'
        fp.write(line)
fp.close()


#Test
with open(os.path.join(dataset_path,'val/val_annotations.txt')) as fp:
    val_files = fp.readlines()

val_files = [f.replace('\t', ' ').strip('\n') for f in val_files]
val_category_names = [f.split(' ')[1] for f in val_files]
val_files = [f.split(' ')[0] for f in val_files]
val_category_nums = [category_names.index(f) for f in val_category_names]

with open(os.path.join(dataset_path,'val.txt', 'w')) as fp:
    for i, f in enumerate(val_files):
        idx = val_category_nums[i]      
        line = f+' '+str(idx)+'\n'
        fp.write(line)
fp.close()
