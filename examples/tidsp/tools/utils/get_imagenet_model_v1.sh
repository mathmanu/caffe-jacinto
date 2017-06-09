#-----------------------------------------------
#Download the pretrained weights
weights_src="https://github.com/tidsp/caffe-jacinto-models/blob/master/examples/tidsp/models/non_sparse/imagenet_classification/jacintonet11_v1/imagenet_jacintonet11_v1_bn_iter_320000.caffemodel?raw=true"
old_weights_dst="training/imagenet_jacintonet11_v1_bn_iter_320000.caffemodel"
wget $weights_src -O $old_weights_dst
	
#The old stride based model (v1) used different names - do net surgery if needed - not required for the new maxpool model (v2)
#Even after net surgery, this may at best be only an initialization for the new maxpool model
new_weights_dst="training/imagenet_jacintonet11_v2_bn_iter_320000.caffemodel"
python ./tools/utils/net_surgery.py \
--old_model="./models/sparse/imagenet_classification/backup/jacintonet11(1000)_bn_maxpool_deploy_oldBNNames.prototxt" \
--old_weights=$old_weights_dst \
--new_model="./models/sparse/imagenet_classification/jacintonet11_maxpool/jacintonet11(1000)_bn_maxpool_deploy.prototxt" \
--new_weights=$new_weights_dst
