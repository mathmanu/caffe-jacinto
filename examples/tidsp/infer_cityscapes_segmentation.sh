#------------------------------------------------------
#palette used to translate id's to colors
palette="[[0,0,0],[128,64,128],[220,20,60],[250,170,30],[0,0,142],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]"

model="models/sparse/cityscapes_segmentation/jsegnet21(8)_bn_deploy.prototxt"

weights="/data/mmcodec_video2_tier3/users/manu/release/segmentation/0.5.1.rc4/convnet10x3c512(c3.1c4.1c5.1)(grp1-4)(eltwise-ctx64)nobn_augument_pad0_sparse_threshold_quant_iter_8000.caffemodel" #"backup/cityscapes_crop640_scale_augument(0.75-1.25)/jsegnet21_train_L2_bn_iter_32000.caffemodel"

num_images=100

crop=0 #"1024 512"

resize="1024 512"

#input="input/stuttgart_00_000000_000001_leftImg8bit.png"
#output=output/output.png

input="input/stuttgart_00"
output="output/stuttgart_00"

#input="input/stuttgart_00.mp4"
#output="output/stuttgart_00.mp4"
#------------------------------------------------------

#------------------------------------------------------
#Actual command
./tools/infer_segmentation.py --blend --crop $crop --resize $resize --model $model --weights $weights --input $input --output $output --palette $palette --num_images $num_images
#------------------------------------------------------

