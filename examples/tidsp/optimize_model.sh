#Optimize step (merge batch norm coefficients to convolution weights - batch norm coefficients will be set to identity after this in the caffemodel)
weights="training/imagenet_jacintonet11_bn_maxpool_L2_iter_160000.caffemodel"
model="models/sparse/imagenet_classification/jacintonet11_maxpool/jacintonet11(1000)_bn_maxpool_deploy.prototxt"
$caffe optimize --model=$model  --gpu=$gpu --weights=$weights --output=$weights_optimized.caffemodel
