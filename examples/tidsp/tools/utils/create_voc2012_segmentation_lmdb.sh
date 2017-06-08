#!/bin/bash
function pause(){
  #read -p "$*"
  echo "$*"
}

#-------------------------------------------------------
rm -rf data/*-lmdb
#-------------------------------------------------------

#-------------------------------------------------------
LOG="training/train-log-`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
#-------------------------------------------------------

#-------------------------------------------------------
export PYTHONPATH=../../../../python:$PYTHONPATH
#-------------------------------------------------------

DATASETPATH=/data/ssd/datasets/object-detect/other/pascal/2012/VOCdevkit/VOC2012

./tools/utils/create_segmentation_image_lmdb.py --rand_seed 1 --shuffle --height=480 --width=480 --list_file=$DATASETPATH/ImageSets/Segmentation/train.txt --image_dir=$DATASETPATH/JPEGImages --search_string="*.jpg" --output_dir="data/train-image-lmdb"
./tools/utils/create_segmentation_image_lmdb.py --rand_seed 1 --shuffle --height=480 --width=480 --list_file=$DATASETPATH/ImageSets/Segmentation/val.txt --image_dir=$DATASETPATH/JPEGImages --search_string="*.jpg" --output_dir="data/val-image-lmdb"

./tools/utils/create_segmentation_image_lmdb.py --rand_seed 1 --label --shuffle --height=480 --width=480 --list_file=$DATASETPATH/ImageSets/Segmentation/train.txt --image_dir=$DATASETPATH/SegmentationClass --search_string="*.png" --output_dir="data/train-label-lmdb"
./tools/utils/create_segmentation_image_lmdb.py --rand_seed 1 --label --shuffle --height=480 --width=480 --list_file=$DATASETPATH/ImageSets/Segmentation/val.txt --image_dir=$DATASETPATH/SegmentationClass --search_string="*.png" --output_dir="data/val-label-lmdb"
