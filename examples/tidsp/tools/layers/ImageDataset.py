import numpy as np
from PIL import Image
from cStringIO import StringIO
import caffe
import yaml
from multiprocessing import (Process, Pipe)
import atexit
from ImageBatchReader import ImageBatchReader

class ImageDataset(caffe.Layer):
  def setup(self, bottom, top):
    self._param_str = self.param_str
    self._layer_params = yaml.load(self._param_str)    
    self._prefetch = self._layer_params.get('prefetch')
        
    self._prefetcher = None
    self._batch_reader = None  
    if self._prefetch == True:
      print('Using prefetch')
      self._conn, conn = Pipe()    
      self._prefetcher = ImagePrefetcher(conn, self._layer_params)
      self._prefetcher.start()
      def cleanup():
        self._prefetcher.terminate()
        self._prefetcher.join() 
        self._conn.close()     
      atexit.register(cleanup)
    else:   
      print('Not using prefetch')    
      self._batch_reader = ImageBatchReader(self._layer_params)
            
    self.reshape(bottom, top)
  
  def next(self):
    if self._prefetch == True:
      blob = self._conn.recv()
    else:
      blob = self._batch_reader.next()  
    return blob
  
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
      
    
class ImagePrefetcher(Process):
  def __init__(self, conn, layer_params):
    super(ImagePrefetcher, self).__init__()
    self._layer_params = layer_params  
    self._conn = conn  
    self._batch_reader = ImageBatchReader(self._layer_params)  
    return 
      
  def type(self):
    return 'ImagePrefetcher'
    
  def run(self):
    while True:
      batch = self._batch_reader.next()
      self._conn.send(batch)
