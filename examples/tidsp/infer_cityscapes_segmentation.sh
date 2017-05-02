#------------------------------------------------------
#palette used to translate id's to colors
palette="[[0,0,0],[128,64,128],[220,20,60],[250,170,30],[0,0,142],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]"

nw_path="/data/mmcodec_video2_tier3/users/manu/experiments/object/segmentation"

model="$nw_path/2017.04/2017.04.15.jsegnet21.maxpool.(sparse_bugfix).rc13/tiscapes(sparse85)/final/jacintonet11+seg10_train_L1_nobn_quant_final_iter_4000_deploy.prototxt"
weights="$nw_path/2017.04/2017.04.15.jsegnet21.maxpool.(sparse_bugfix).rc13/tiscapes(sparse85)/final/jacintonet11+seg10_train_L1_nobn_quant_final_iter_4000.caffemodel" 

num_images=1000

crop="1024 512"

resize=0 #"1024 512"

#input="input/stuttgart_00_000000_000001_leftImg8bit.png"
#output=output/output.png

input="input/sample"
output="output/sample"

#input="input/stuttgart_00.mp4"
#output="output/stuttgart_00.mp4"
#------------------------------------------------------

#------------------------------------------------------
#Actual command
./tools/infer_segmentation.py --blend --crop $crop --resize $resize --model $model --weights $weights --input $input --output $output --palette $palette --num_images $num_images
#------------------------------------------------------

