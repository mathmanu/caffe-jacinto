import caffe

net_old = caffe.Net('/user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/examples/tidsp/models/sparse/imagenet_classification/jacintonet11_maxpool/jacintonet11(1000)_bn_maxpool_deploy_oldBNNames.prototxt', '/data/mmcodec_video2_tier3/users/manu/experiments/object/classification/2017.04/2017.04.imagenet/jacintonet11_maxpool(60.52%)/original/imagenet_jacintonet11_bn_maxpool_L2_iter_160000.caffemodel', caffe.TEST)

net_new = caffe.Net('/user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/examples/tidsp/models/sparse/imagenet_classification/jacintonet11_maxpool/jacintonet11(1000)_bn_maxpool_deploy.prototxt', '/data/mmcodec_video2_tier3/users/manu/experiments/object/classification/2017.04/2017.04.imagenet/jacintonet11_maxpool(60.52%)/original/imagenet_jacintonet11_bn_maxpool_L2_iter_160000.caffemodel', caffe.TEST)

for i in range(5):
  net_new.params['conv1a/bn'][i].data[...] = net_old.params['bn_conv1a'][i].data[...].ravel()
  net_new.params['conv1b/bn'][i].data[...] = net_old.params['bn_conv1b'][i].data[...].ravel()
  net_new.params['res2a_branch2a/bn'][i].data[...] = net_old.params['bn2a_branch2a'][i].data[...].ravel()
  net_new.params['res2a_branch2b/bn'][i].data[...] = net_old.params['bn2a_branch2b'][i].data[...].ravel()
  net_new.params['res3a_branch2a/bn'][i].data[...] = net_old.params['bn3a_branch2a'][i].data[...].ravel()
  net_new.params['res3a_branch2b/bn'][i].data[...] = net_old.params['bn3a_branch2b'][i].data[...].ravel()
  net_new.params['res4a_branch2a/bn'][i].data[...] = net_old.params['bn4a_branch2a'][i].data[...].ravel()
  net_new.params['res4a_branch2b/bn'][i].data[...] = net_old.params['bn4a_branch2b'][i].data[...].ravel()
  net_new.params['res5a_branch2a/bn'][i].data[...] = net_old.params['bn5a_branch2a'][i].data[...].ravel()
  net_new.params['res5a_branch2b/bn'][i].data[...] = net_old.params['bn5a_branch2b'][i].data[...].ravel()                 
  
print('Completed copying..')  
net_new.save('new_bn_names.caffemodel')




