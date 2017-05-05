import sys
import pandas as pd
import numpy as np
import lmdb
import os
import caffe
from caffe.io import caffe_pb2 

class LMDBReader():
  def __init__(self, source):
    self._source = source
    try:
      self._db = lmdb.open(self._source)
      self._data_cursor = self._db.begin().cursor()
    except: 
      raise Exception(str(self._source)+": Could not be opened")
    
  def next(self):
    if not self._data_cursor.next():
      self._data_cursor = self._db.begin().cursor()
    value_str = self._data_cursor.value()
    datum = caffe_pb2.Datum()
    datum.ParseFromString(value_str)
    data = datum.data or datum.float_data
    label = datum.label
    return data, label

    
  def __del__(self):
    self._db.close()
