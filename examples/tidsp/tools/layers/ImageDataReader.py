import sys
import pandas as pd
import numpy as np
import lmdb
import os
import caffe
from caffe.io import caffe_pb2 
from PIL import Image
from cStringIO import StringIO

class ImageDataReader():
  def __init__(self, source, compressed, resize, crop):
    self._source = source
    self._compressed = compressed
    self._resize = resize
    self._crop = crop       
    try:
      self._db = lmdb.open(self._source)
      self._data_cursor = self._db.begin().cursor()
    except: 
      raise Exception(str(self._source)+": Could not be opened")
    
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
    if not self._data_cursor.next():
      self._data_cursor = self._db.begin().cursor()
    value_str = self._data_cursor.value()
    datum = caffe_pb2.Datum()
    datum.ParseFromString(value_str)
    data = datum.data or datum.float_data
    image = self.decode_image_str(data)       
    label = datum.label
    return image, label

    
  def __del__(self):
    self._db.close()
