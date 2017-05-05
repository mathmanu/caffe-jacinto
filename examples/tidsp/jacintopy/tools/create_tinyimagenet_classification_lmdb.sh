#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

DST=./data
DATA=/data/hdd/datasets/object-detect/other/ilsvrc/cs231n-tinyimagenet
TOOLS=../../build/tools

TRAIN_DATA_ROOT=$DATA/train/
VAL_DATA_ROOT=$DATA/val/images/

rm -rf $DST/cs231n-tinyimagenet_*_lmdb
    
# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
# Tinyimagenet is already 64x64. Make it slightly larger to enable random crop
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=72
  RESIZE_WIDTH=72
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

    
echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --encoded \
    --encode_type jpg \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $DST/cs231n-tinyimagenet_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --encoded \
    --encode_type jpg \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $DST/cs231n-tinyimagenet_val_lmdb

echo "Done."
