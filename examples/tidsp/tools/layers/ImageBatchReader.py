import numpy as np
from ImageDataReader import ImageDataReader

class ImageBatchReader():
  def __init__(self, layer_params):
    self._layer_params = layer_params
    self._source = self._layer_params.get('source')
    self._source_type = self._layer_params.get('source_type')
    self._batch_size = self._layer_params.get('batch_size')
    self._prefetch = self._layer_params.get('prefetch')
    self._fetch_type = self._layer_params.get('fetch_type')
    self._resize = self._layer_params.get('resize'); exec('self._resize='+self._resize);               
    self._crop = self._layer_params.get('crop'); exec('self._crop='+self._crop);   
    self._compressed = self._layer_params.get('compressed')          
    self._reader = ImageDataReader(self._source, self._compressed, self._resize, self._crop)
    return 
      
  def type(self):
    return 'ImageBatchReader'
        
  def next(self):  
    image_batch = []
    label_batch = []    
    for i in range(self._batch_size):
      image, label = self._reader.next()     
      image_batch.extend(image)
      label_batch.extend([label])
    image_batch = np.array(image_batch)
    label_batch = np.array(label_batch).reshape(self._batch_size, 1, 1, 1)
    batch = [image_batch, label_batch]
    return batch
    
    
