#------------------------------------------------------
#palette used to translate id's to colors
palette="[[0,0,0],[128,64,128],[220,20,60],[250,170,30],[0,0,142],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]"

nw_path="/data/mmcodec_video2_tier3/users/manu/experiments/object/segmentation"

model="$nw_path/2017.04/2017.04.15.jsegnet21.maxpool.(sparse_bugfix).rc13/tiscapes(sparse85)/final/jacintonet11+seg10_train_L1_bn_noquant_optimized_iter_32000.prototxt"
weights="$nw_path/2017.04/2017.04.15.jsegnet21.maxpool.(sparse_bugfix).rc13/tiscapes(sparse85)/final/jacintonet11+seg10_train_L1_bn_noquant_optimized_iter_32000.caffemodel" 

num_images=1000

crop="1024 512"

resize=0 #"1024 512"


#------------------------------------------------------
#Infer and write chroma blended visualization
input="input/lindau_V105"
output="output/lindau_V105"
./tools/infer_segmentation.py --blend --crop $crop --resize $resize --model $model --weights $weights --input $input --output $output --palette $palette --num_images $num_images
#------------------------------------------------------
#Accuracy measurement
#input="data/val-image-list.txt"
#label="data/val-label-list.txt"
#output="output/sample"
#num_classes=5
#./tools/infer_segmentation.py --crop $crop --resize $resize --model $model --weights $weights --input $input --label $label --num_classes=$num_classes --output $output --palette $palette --num_images=$num_images
#------------------------------------------------------

