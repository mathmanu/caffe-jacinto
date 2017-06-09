import caffe
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_model', type=str, required=True, help='Old model name')    
    parser.add_argument('--old_weights', type=str, required=True, help='Pretrained caffemodel name')  
    parser.add_argument('--new_model', type=str, required=True, help='New model name')    
    parser.add_argument('--new_weights', type=str, required=True, help='Output pretrained caffemodel name')           
    return parser.parse_args()
    
def change_names():
    args = get_arguments()
    
    #Old model and weights
    net_old = caffe.Net(args.old_model, args.old_weights, caffe.TEST)

    #Initialize with old wights -> some layers will nto get initialized
    net_new = caffe.Net(args.new_model, args.old_weights, caffe.TEST)

    #Take care of the layers (which have atleast one blob) with names cahnges
    for i in range(5):
      net_new.params['conv1a/bn'][i].data[...] = net_old.params['bn_conv1a'][i].data[...]
      net_new.params['conv1b/bn'][i].data[...] = net_old.params['bn_conv1b'][i].data[...]
      net_new.params['res2a_branch2a/bn'][i].data[...] = net_old.params['bn2a_branch2a'][i].data[...]
      net_new.params['res2a_branch2b/bn'][i].data[...] = net_old.params['bn2a_branch2b'][i].data[...]
      net_new.params['res3a_branch2a/bn'][i].data[...] = net_old.params['bn3a_branch2a'][i].data[...]
      net_new.params['res3a_branch2b/bn'][i].data[...] = net_old.params['bn3a_branch2b'][i].data[...]
      net_new.params['res4a_branch2a/bn'][i].data[...] = net_old.params['bn4a_branch2a'][i].data[...]
      net_new.params['res4a_branch2b/bn'][i].data[...] = net_old.params['bn4a_branch2b'][i].data[...]
      net_new.params['res5a_branch2a/bn'][i].data[...] = net_old.params['bn5a_branch2a'][i].data[...]
      net_new.params['res5a_branch2b/bn'][i].data[...] = net_old.params['bn5a_branch2b'][i].data[...]               
      
    print('Completed copying..')  
    net_new.save(args.new_weights)

def main():
    change_names()
    
if __name__ == "__main__":
    main() 
