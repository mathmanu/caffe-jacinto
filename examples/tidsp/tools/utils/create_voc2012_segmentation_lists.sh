#!/bin/bash

#-------------------------------------------------------
rm -rf data/*-folder
rm -rf data/*-list.txt
#-------------------------------------------------------

#-------------------------------------------------------
LOG="training/train-log-`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
#-------------------------------------------------------

#-------------------------------------------------------
export PYTHONPATH=../../../../python:$PYTHONPATH
export DATASETPATH=/data/hdd/datasets/object-detect/other/pascal/2012/VOCdevkit/VOC2012
#-------------------------------------------------------

#The label images are converted copied to local folder. This is because paletted pngs need the index to be stored and image. 
#OpenCV (in caffe) will otherwise use the color values, instead of the index.

#-------------------------------------------------------
echo "creating train lists"
./tools/utils/create_image_folder.py --label --list_file=$DATASETPATH/ImageSets/Segmentation/train.txt --image_dir=$DATASETPATH/SegmentationClass --search_string="*.png" --output_dir="data/train-label-folder"
> data/train-image-list.txt
> data/train-label-list.txt
for f in `cat $DATASETPATH/ImageSets/Segmentation/train.txt`
do
echo $DATASETPATH/JPEGImages/$f.jpg >> data/train-image-list.txt
echo data/train-label-folder/$f.png >> data/train-label-list.txt
done
#-------------------------------------------------------

#-------------------------------------------------------
echo "creating val lists"
./tools/utils/create_image_folder.py --label --list_file=$DATASETPATH/ImageSets/Segmentation/val.txt --image_dir=$DATASETPATH/SegmentationClass --search_string="*.png" --output_dir="data/val-label-folder"
> data/val-image-list.txt
> data/val-label-list.txt
for f in `cat $DATASETPATH/ImageSets/Segmentation/val.txt`
do
echo $DATASETPATH/JPEGImages/$f.jpg >> data/val-image-list.txt
echo data/val-label-folder/$f.png >> data/val-label-list.txt
done

echo "done"
#-------------------------------------------------------

