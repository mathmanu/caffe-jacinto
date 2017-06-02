#!/bin/bash
function pause(){
  #read -p "$*"
  echo "$*"
}

#-------------------------------------------------------
rm -rf data/train-image-lmdb data/val-image-lmdb data/train-label-lmdb data/val-label-lmdb
#-------------------------------------------------------

#-------------------------------------------------------
LOG="training/train-log-`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
#-------------------------------------------------------

#-------------------------------------------------------
export PYTHONPATH=../../../../python:$PYTHONPATH
#-------------------------------------------------------


./tools/utils/create_segmentation_image_lmdb.py --rand_seed 1 --shuffle --list_file=data/train-image-list.txt --output_dir="data/train-image-lmdb"
./tools/utils/create_segmentation_image_lmdb.py --rand_seed 1 --shuffle --list_file=data/val-image-list.txt --output_dir="data/val-image-lmdb"

./tools/utils/create_segmentation_image_lmdb.py --rand_seed 1 --shuffle --label --list_file=data/train-label-list.txt --output_dir="data/train-label-lmdb"
./tools/utils/create_segmentation_image_lmdb.py --rand_seed 1 --shuffle --label --list_file=data/val-label-list.txt --output_dir="data/val-label-lmdb"
