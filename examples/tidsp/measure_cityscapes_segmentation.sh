#-------------------------------------------------------
LOG="training/train-log-`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
#-------------------------------------------------------

#Thsi can be used if the output was generted without transforming to the cityscapes 34 categoris using --label_dict="$label_dict_20_to_34" option inside the infer script that generated the output images. 
#python /user/a0393608/files/work/code/vision/github/mcordts/cityscapesScripts/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py /user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/examples/tidsp/data/val-label-list.txt /user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/examples/tidsp/data/output_name_list.txt


#Use the original cityscapes label Ids
python /user/a0393608/files/work/code/vision/github/mcordts/cityscapesScripts/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py /user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/examples/tidsp/data/cityscapes_gtFine_val_labels_Ids.txt /user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/examples/tidsp/data/output_name_list.txt




