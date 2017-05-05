import numpy as np
from PIL import Image
from cStringIO import StringIO
import caffe
import yaml
from LMDBReader import LMDBReader

class ImageDataset(caffe.Layer):
  def setup(self, bottom, top):
    self._layer_params = yaml.load(self.param_str)
    self._source = self._layer_params.get('source')
    self._source_type = self._layer_params.get('source_type')
    self._batch_size = self._layer_params.get('batch_size')
    self._prefetch = self._layer_params.get('prefetch')
    self._fetch_type = self._layer_params.get('fetch_type')
    self._resize = self._layer_params.get('resize'); exec('self._resize='+self._resize);               
    self._crop = self._layer_params.get('crop'); exec('self._crop='+self._crop);   
    self._compressed = self._layer_params.get('compressed')          
    self._reader = LMDBReader(self._source)
    
  def decode_image_str(self,image_data):
    if self._compressed:
      buffer = StringIO(image_data)
      #buffer.seek(0)
      image_data = Image.open(buffer).convert('RGB')
    if self._resize:
      image_data = image_data.resize(self._resize, Image.ANTIALIAS)
    if self._crop:
      image_data = image_data.crop((0, 0, self._crop[0], self._crop[1]))
    image_data = np.array(image_data).astype(np.float32)
    image_bgr = image_data[..., ::-1]              #RGB->BGR
    input_blob = image_bgr.transpose((2, 0, 1))    #Interleaved to planar
    input_blob = input_blob[np.newaxis, ...]       #Introduce the batch dimension
    return input_blob
        
  def next(self):  
    image_batch = []
    label_batch = []    
    for i in range(self._batch_size):
      data, label = self._reader.next()
      image = self.decode_image_str(data)        
      image_batch.extend(image)
      label_batch.extend([label])
    image_batch = np.array(image_batch)
    label_batch = np.array(label_batch).reshape(self._batch_size, 1, 1, 1)
    batch = [image_batch, label_batch]
    return batch
        
  def forward(self, bottom, top):
    blob = self.next()
    for i in range(len(blob)):
      top[i].reshape(*(blob[i].shape))
      top[i].data[...] = blob[i].astype(np.float32, copy=False)
    return

  def backward(self, bottom, top):
    pass
  
  def reshape(self, bottom, top):
    blob = self.next()
    for i in range(len(blob)):
      top[i].reshape(*(blob[i].shape))
    return
      

    
        
    

