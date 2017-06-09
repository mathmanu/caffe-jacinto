#Download the pretrained weights
weights_src="https://github.com/tidsp/caffe-jacinto-models/blob/master/examples/tidsp/models/non_sparse/imagenet_classification/jacintonet11_v2/imagenet_jacintonet11_v2_bn_iter_160000.caffemodel?raw=true"
weights_dst="training/imagenet_jacintonet11_v2_bn_iter_160000.caffemodel"
wget $weights_src -O $weights_dst
