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
export MODIFY_LABEL=1
export DATASETPATH=/data/ssd/datasets/object-detect/other/cityscapes/data
#-------------------------------------------------------

#-------------------------------------------------------
#Use original 34 classes
#export LABEL_MAP_DICT="[]"

#Default cityscapes mapping - 33 to 19 class
#see definition of the list 'labels' in https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
#export LABEL_MAP_DICT19="{0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255, 7:0, 8:1, 9:255, 10:255, 11:2, 12:3, 13:4, 14:255, 15:255, 16:255, 17:5, 18:255, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:255, 30:255, 31:16, 32:17, 33:18, 255:255}"

#ignored classes mapped to a 20th class (id 19)
export LABEL_MAP_DICT20="{0:255, 1:19, 2:255, 3:255, 4:255, 5:255, 6:255, 7:0, 8:1, 9:255, 10:255, 11:2, 12:3, 13:4, 14:255, 15:255, 16:255, 17:5, 18:255, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:255, 30:255, 31:16, 32:17, 33:18, 255:255}"

#Custom mapping - 33 to 5 class
#export LABEL_MAP_DICT5="{0:255, 1:0, 2:255, 3:255, 4:0, 5:0, 6:0, 7:1, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:3, 20:3, 21:0, 22:0, 23:0, 24:2, 25:2, 26:4, 27:4, 28:4, 29:4, 30:4, 31:4, 32:2, 33:2, 255:255}"
#-------------------------------------------------------


#-------------------------------------------------------
#The label images are converted copied to local folder. This is because paletted pngs need the index to be stored and image. 
#OpenCV (in caffe) will otherwise use the color values, instead of the index.

echo "creating train lists"
> data/train-image-list.txt
> data/train-label-list.txt
find $DATASETPATH/leftImg8bit/train -name *.png | sort > data/train-image-list.txt

if [ $MODIFY_LABEL -eq 1 ]
then #of if
LABEL_FOLDER=data/train-label-folder
WILDCARD="*labelIds.png"
./tools/utils/create_image_folder.py --label --image_dir=$DATASETPATH/gtFine/train --search_string="*/$WILDCARD" --output_dir=$LABEL_FOLDER --label_dict="$LABEL_MAP_DICT20"

else #of if
LABEL_FOLDER=$DATASETPATH/gtFine/train
WILDCARD="*labelTrainIds.png"
fi #of if

find $LABEL_FOLDER -name $WILDCARD | sort > data/train-label-list.txt
#-------------------------------------------------------


#-------------------------------------------------------
echo "creating val lists"
> data/val-image-list.txt
> data/val-label-list.txt
find $DATASETPATH/leftImg8bit/val -name *.png | sort > data/val-image-list.txt
if [ $MODIFY_LABEL -eq 1 ]
then #of if
LABEL_FOLDER=data/val-label-folder
WILDCARD="*labelIds.png"
./tools/utils/create_image_folder.py --label --image_dir=$DATASETPATH/gtFine/val --search_string="*/$WILDCARD" --output_dir=$LABEL_FOLDER --label_dict="$LABEL_MAP_DICT20"
else #of if
LABEL_FOLDER=$DATASETPATH/gtFine/val
WILDCARD="*labelTrainIds.png"
fi #of if

find $LABEL_FOLDER -name $WILDCARD | sort > data/val-label-list.txt
#-------------------------------------------------------


