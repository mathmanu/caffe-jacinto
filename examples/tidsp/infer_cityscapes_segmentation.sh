#-------------------------------------------------------
LOG="training/train-log-`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
#-------------------------------------------------------

#------------------------------------------------------
#palette used to translate id's to colors - for 5 classes
#palette5="[[0,0,0],[128,64,128],[220,20,60],[250,170,30],[0,0,142],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]"

#for 19 or 20 classes training of cityscapes, first convert to original labelIds and then apply the pallete
label_dict_20_to_34="{0:7, 1:8, 2:11, 3:12, 4:13, 5:17, 6:19, 7:20, 8:21, 9:22, 10:23, 11:24, 12:25, 13:26, 14:27, 15:28, 16:31, 17:32, 18:33, 19:0}"

#34 class pallette - for visualization
palette34="[(  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(111, 74,  0),( 81,  0, 81),(128, 64,128),(244, 35,232),(250,170,160),(230,150,140),( 70, 70, 70),(102,102,156),(190,153,153),(180,165,180),(150,100,100),(150,120, 90),(153,153,153),(153,153,153),(250,170, 30),(220,220,  0),(107,142, 35),(152,251,152),( 70,130,180),(220, 20, 60),(255,  0,  0),(  0,  0,142),(  0,  0, 70),(  0, 60,100),(  0,  0, 90),(  0,  0,110),(  0, 80,100),(  0,  0,230),(119, 11, 32),(  0,  0,142)]"

model="models/sparse/cityscapes_segmentation/jsegnet21_maxpool/jsegnet21_maxpool(8)_bn_deploy.prototxt"
weights="training/jsegnet21_maxpool_L1_bn_iter_32000.caffemodel"
num_images=1000 #10 #1000
crop=0 #"1024 512"
resize=0 #"1024 512"


#------------------------------------------------------
#Infer and write chroma blended visualization
#input="input/sample"
#output="output/sample"

input="./data/val-image-list.txt"
output="output/cityscapes"

#Generate output images for chroma blended visualization
#./tools/utils/infer_segmentation.py --crop $crop --resize $resize --model $model --weights $weights --input $input --output $output --num_images $num_images --resize_back --label_dict="$label_dict_20_to_34" --blend --palette="$palette34"


#Generate output images running the IOU measurement (using the measure_...  script)
./tools/utils/infer_segmentation.py --crop $crop --resize $resize --model $model --weights $weights --input $input --output $output --num_images $num_images --resize_back --label_dict="$label_dict_20_to_34"


