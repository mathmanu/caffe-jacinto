##################################################################################################
import numpy as np
import matplotlib
#matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt
from nms import non_max_suppression_fast

#%matplotlib inline
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from array import array 
from enum import IntEnum
from copy import deepcopy

import pylab
import imageio
import sys, getopt
import os

#dflt:False
write_boxes_afr_nms = True

def visualize_weights(net, layer_name, padding=4, filename=''):
    # The parameters are a list of [weights, biases]
    if layer_name in net.params:
      print "net.params[layer_name]: ", net.params[layer_name] 
      data = np.copy(net.params[layer_name][0].data)
    else:
      return

    # N is the total number of convolutions
    print len(data.shape)
    if (len(data.shape) > 1):
      N = data.shape[0]*data.shape[1]
      # Ensure the resulting image is square
      filters_per_row = int(np.ceil(np.sqrt(N)))
      # Assume the filters are square
      filter_size = data.shape[2]
      # Size of the result image including padding
      result_size = filters_per_row*(filter_size + padding) - padding
      # Initialize result image to all zeros
      result = np.zeros((result_size, result_size))
 
      # Tile the filters into the result image
      filter_x = 0
      filter_y = 0
      for n in range(data.shape[0]):
          for c in range(data.shape[1]):
              if filter_x == filters_per_row:
                  filter_y += 1
                  filter_x = 0
              for i in range(filter_size):
                  for j in range(filter_size):
                      result[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j] = data[n, c, i, j]
              filter_x += 1
 
      # Normalize image to 0-1
      min = result.min()
      max = result.max()
      result = (result - min) / (max - min)
 
      # Plot figure
      plt.figure(figsize=(10, 10))
      plt.axis('off')
      plt.imshow(result, cmap='gray', interpolation='nearest')
 
      # Save plot if filename is set
      if filename != '':
          print "saving :",  filename 
          plt.savefig(filename, bbox_inches='tight', pad_inches=1)
 
      #plt.show()
      #plt.close()
      #return
#######################################################################################################
def debug(var):
  print var, " = ", eval(var)

class ARType(IntEnum):
  AR_H_SAME_AS_DESC_H = 0
  AR_W_SAME_AS_DESC_W = 1
  AR_PRESERVE = 2

REC_WIDTH = 2
#######################################################################################################
def get_labelname(labelmap, labels):
  from google.protobuf import text_format
  from caffe.proto import caffe_pb2
  
  num_labels = len(labelmap.item)
  labelnames = []
  if type(labels) is not list:
    labels = [labels]
  for label in labels:
    found = False
    for i in xrange(0, num_labels):
      if label == labelmap.item[i].label:
        found = True
        labelnames.append(labelmap.item[i].display_name)
        break
    assert found == True
  return labelnames

##################################################################################################
def processOneCrop(curScaleImage, transformer, net, labelmapFile, drawHandle, 
    detBBoxesCurFrame, offsetX, offsetY, scaleX, scaleY, aspectRatio, confTh) :
  #debug('offsetX') 
  #debug('offsetY') 
  #debug('scaleX') 
  #debug('scaleY') 
  #debug('aspectRatio') 

  transformed_image = transformer.preprocess('data', curScaleImage)
  net.blobs['data'].data[...] = transformed_image

  # Forward pass.
  detections = net.forward()['detection_out']
  #print "local detection.shape ", detections.shape

  # Parse the outputs.
  det_label = detections[0,0,:,1]
  det_conf = detections[0,0,:,2]
  det_xmin = detections[0,0,:,3]
  det_ymin = detections[0,0,:,4]
  det_xmax = detections[0,0,:,5]
  det_ymax = detections[0,0,:,6]

  # Get detections with confidence higher than confTh(def =0.6).
  det_label_list = det_label.astype(np.int).tolist()
  from google.protobuf import text_format
  from caffe.proto import caffe_pb2
  
  file = open(labelmapFile, 'r')
  labelmap = caffe_pb2.LabelMap()
  text_format.Merge(str(file.read()), labelmap)

  #indiates age of the tracked obj. In the frame it gets detected (born) set it to 0
  age=0.0
  
  if type(confTh) is dict:
    confThList = [None] * len(det_label_list)
    for i, det_label_cur_obj in enumerate(det_label_list): 
      if(det_label_cur_obj <> -1):
        confThList[i] = confTh[str(get_labelname(labelmap,det_label_cur_obj)[0])] 
      else:  
        #some thing went wrong. Set conservative th
        confThList[i] = 1.0
    top_indices = [i for i, conf in enumerate(det_conf) if(conf > confThList[i])]
  else:  
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= confTh]
  
  top_conf = det_conf[top_indices]
  top_label_indices = det_label[top_indices].tolist()
  top_labels = get_labelname(labelmap, top_label_indices)
  top_xmin = det_xmin[top_indices]
  top_ymin = det_ymin[top_indices]
  top_xmax = det_xmax[top_indices]
  top_ymax = det_ymax[top_indices]

  #colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
  colors = plt.cm.hsv(np.linspace(0, 1, 255)).tolist()

  #plt.imshow(image)
  #currentAxis = plt.gca()
  #print "curScaleImage.shape : ", curScaleImage.shape
  for i in xrange(top_conf.shape[0]):
    xmin = int(round(top_xmin[i] * curScaleImage.shape[1]))
    ymin = int(round(top_ymin[i] * curScaleImage.shape[0]))
    xmax = int(round(top_xmax[i] * curScaleImage.shape[1]))
    ymax = int(round(top_ymax[i] * curScaleImage.shape[0]))
    score = top_conf[i]
    label = int(top_label_indices[i])
    label_name = top_labels[i]

    if label > 254 :
      print (label)
    color = colors[label]
    #display score and label name
    #print "xmin, ymin, xmax, ymax", xmin, " , ", ymin," , ", xmax," , ", ymax
    #print "scaleX:scaleY ", scaleX, " , ",  scaleY
    bbox = (int(xmin*scaleX)+offsetX, int(ymin*scaleY)+offsetY,
        int(xmax*scaleX)+offsetX, int(ymax*scaleY)+offsetY, label, score, age)
    #print "bbox : ", bbox  
    # store box co-ordinates along with label and score
    detBBoxesCurFrame.append(bbox)
    #detLabelsCurFrame.append(label)
    #detLabelNamesCurFrame.append(label_name)
    #detScoresCurFrame.append(score)
  
  return [drawHandle,detections]

##################################################################################################
def writeOneBox(enable=False, bbox=[], label_name='', score=-1.0, fileHndl='', writeBboxMap=[]):
  if enable:
    # KITTI benchmarking format
    #map to category specified in writeBboxMap
    #e.g. WRITE_BBOX_MAP=[['car','vehicle'], ['train','vehicle'],['bus','vehicle']]
    #print "writeBboxMap in writeDets", writeBboxMap
    for xlation in writeBboxMap:
      #print "xlation ", xlation 
      if label_name == xlation[0]:
        label_name = xlation[1]

    newLine = '{} 0 0 0 {} {} {} {} 0 0 0 0 0 0 0 {} \n'.format(label_name,
      bbox[0],bbox[1],bbox[2], bbox[3], score)
    #print "newLine: ", newLine
    fileHndl.write(newLine)

##################################################################################################
def drawBoxes(params=[], detObjs=[], drawHandle=[], writeBboxMap=[],
    labelmap=[]):

  colors = plt.cm.hsv(np.linspace(0, 1, 255)).tolist()
  for idx in range(detObjs.shape[0]): 
    #if params.enNMS:     
    #  idxBfrNMS = pick[i] 
    #else:
    #  idxBfrNMS = i

    #score = detScoresCurFrame[idxBfrNMS] 
    #label = detLabelsCurFrame[idxBfrNMS] 
    #label_name = detLabelNamesCurFrame[idxBfrNMS] 

    label = int(detObjs[idx][4]) 
    score = detObjs[idx][5] 
    label_name = str(get_labelname(labelmap,label)[0])
   
    #writeOneBox(enable=params.writeBbox, bbox=detObjs[i], label_name=label_name, 
    #  score=score, fileHndl=detObjFileHndl, writeBboxMap=writeBboixMap)
                                                                   
    if type(params.confTh) is dict:                                
      draw_cur_reg = score > params.confTh[label_name]
    else:    
      draw_cur_reg = score > params.confTh

    if draw_cur_reg: 
      display_txt = '%.3s: %.2f'%(label_name[:4], score)
      if label > 254 :
        print (label)
      color = colors[label]

      #/usr/share/fonts/truetype/droid/DroidSans.ttf
      fnt = ImageFont.truetype("DroidSans.ttf", 10)
      #display score and label name
      cor = detObjs[idx]
      drawHandle.text((cor[0]+REC_WIDTH, cor[1]+REC_WIDTH),display_txt,(255,255,255),font=fnt)
         
      # determine what color rect to draw
      rectColorAry = ["violate", "blue","yellow","orange","red","pink", "white", "brown", "green"]
      numColor=len(rectColorAry)
      label = min(label, numColor-1)
      rectColor = rectColorAry[label]

      #draw.rectangle(((xmin,ymin),(xmax,ymax)), outline = "blue")
      # hack to draw rectangle with larger than 1 px thickness. As ImageDraw natively does not support it.
      for i in range(-REC_WIDTH,REC_WIDTH):
        locRec = (cor[0]+i,cor[1]+i, cor[2]+i,cor[3]+i) 
        drawHandle.rectangle(locRec, outline=rectColor)   
    
  return drawHandle
##############################################################
def writeBoxes(params=[], detObjs=[], detObjFileHndl='', writeBboxMap=[],
    labelmap=[]):
 
  for i in range(detObjs.shape[0]): 
    #if len(pick) > 0:
    #  idxBfrNMS = pick[i]
    #else:
    #  idxBfrNMS = i

    #score = detScoresCurFrame[idxBfrNMS] 
    #label = detLabelsCurFrame[idxBfrNMS] 
    #label_name = detLabelNamesCurFrame[idxBfrNMS]

    label = int(detObjs[i][4])
    score = detObjs[i][5] 
    label_name = str(get_labelname(labelmap,label)[0])
   
    writeOneBox(enable=params.writeBbox, bbox=detObjs[i], label_name=label_name, 
      score=score, fileHndl=detObjFileHndl, writeBboxMap=writeBboxMap)
   
  return

##################################################################################################
def wrapMulTiles(imageCurFrame, transformer, net, params, curFrameNum=0, detObjsFile='', writeBboxMap=[]):
  #imageCoreIp = np.asarray(imageCurFrame)
  #print 'imageCoreIp.shape :',  imageCoreIp.shape
  #imageCoreIp = imageCoreIp/255.0
  #imageCurFrameAry = Image.fromarray((imageCoreIp*255).astype(np.uint8))
  imageCurFrameAry = deepcopy(imageCurFrame)

  curFrameDrawHandle = ImageDraw.Draw(imageCurFrameAry)

  #if(params.enCrop):
  #  imageCurFrame = imageCurFrame.crop((params.cropMinX,params.cropMinY,params.cropMaxX,params.cropMaxY))

  if ((params.resizeW <> 0) or (params.resizeH <> 0)):
    imageCurFrame = imageCurFrame.resize((params.resizeW,params.resizeH), Image.ANTIALIAS);
  
  tileSizeX = params.tileSizeW
  tileSizeY = params.tileSizeH
  w, h = imageCurFrame.size
  
  #numTileX = ((w-1) / tileSizeX ) + 1
  #numTileY = ((h-1) / tileSizeY ) + 1
  
  #print w, h, numTileX, numTileY
  detObjRectList =[]
  combinedDetImgOp = Image.new("RGB", (w, h))
  detBBoxesCurFrame = []
  #detLabelsCurFrame = []
  #detLabelNamesCurFrame = []
  #detScoresCurFrame = []
 
  offsetX = range(0,w-tileSizeX/2, params.tileStepX)
  offsetY = range(0,h-tileSizeY/2 ,params.tileStepY)

  #for tileYIdx in range(0, numTileY, params.tileStepY):
  #  for tileXIdx in range(0, numTileX, params.tileStepX):
  
  for left in offsetX:
    for top in offsetY:
      #left = tileXIdx*tileSizeX
      #top = tileYIdx*tileSizeY
  
      right = left + tileSizeX
      bottom = top + tileSizeY
      if(curFrameNum==0):
        print "left, top, right, bottom : ", left, top, right, bottom
  
      imageCrop = imageCurFrame.crop((left, top, right, bottom))
      
      cropW, cropH = imageCrop.size
      #print cropW, cropH

      #processOneCrop expects np array in the range[0,1]
      imageCoreIp = np.asarray(imageCrop)
      imageCoreIp = imageCoreIp/255.0
  
      #imageCopy = Image.fromarray((imageCoreIp*255).astype(np.uint8))
      curTileDrawHandle = ImageDraw.Draw(imageCrop)
      detections = []
      # as rect is drawn inside in case of multiple tiles all the
      # offsets are dummy
      if params.writeBbox:
        #for objective eval write out all boxes
        confTh=0.01
      else:  
        confTh=params.confTh

      processOneCrop(imageCoreIp, transformer, net, params.labelmapFile, curTileDrawHandle, detBBoxesCurFrame,
        offsetX=left, offsetY=top,scaleX=1.0, scaleY=1.0,
        aspectRatio=1.0, confTh=confTh)
      combinedDetImgOp.paste(imageCrop, (left,top))

  detObjRectListNPArray = np.array(detBBoxesCurFrame)
  #print "detObjRectListNPArray: ", detObjRectListNPArray   

  #write detected boxes before NMS
  if write_boxes_afr_nms == False:
    if len(detObjRectListNPArray) and params.writeBbox and ((curFrameNum%params.decFreq) == 0):
      detObjFileHndl = open(detObjsFile, "w")
      writeBoxes(params=params,detObjs=detObjRectListNPArray, detObjFileHndl=detObjFileHndl, 
        writeBboxMap=writeBboxMap, labelmap=labelmap)
      detObjFileHndl.close()

  pick = []
  print " Objs bef,afr NMS, %d " % (len(detObjRectListNPArray )),
  print detObjRectListNPArray,
  if(params.enNMS):
    nmsTh = 0.45 #0.5
    #print detObjRectListNPArray 
    detObjRectListNPArray = non_max_suppression_fast(detObjRectListNPArray, nmsTh, pick)
    print (len(detObjRectListNPArray )),
    print detObjRectListNPArray,

  #write detected boxes Afr NMS
  if write_boxes_afr_nms:
    if len(detObjRectListNPArray) and params.writeBbox and ((curFrameNum%params.decFreq) == 0):
      detObjFileHndl = open(detObjsFile, "w")
      writeBoxes(params=params,detObjs=detObjRectListNPArray, detObjFileHndl=detObjFileHndl, 
        writeBboxMap=writeBboxMap, labelmap=labelmap)
      detObjFileHndl.close()

  print(' ')
  if(len(detObjRectListNPArray)):
    drawBoxes(params=params,detObjs=detObjRectListNPArray, drawHandle=curFrameDrawHandle,
        writeBboxMap=writeBboxMap,labelmap=labelmap)
  return imageCurFrameAry 
##################################################################################################
def updateWithNearestKeypointpoint(curPosX=0,curPosY=0,tlX=0,
    tlY=0, brX=0, brY=0, p0=[],p1=[],good_old=[],good_new=[]):
  debug_print = False
  #if L1 dist to nearest keypoint is more than this value consider it as
  #failure
  minErrTh = 50
  result = False
  best_p0 = [] 
  best_match_p0 = [] 
  minErr = 1E10
  matchFoundTL = False
  for pointP0, pointP1 in zip(good_old, good_new):
    #consider nearest point which is inside the det objects
    if (pointP0[0] >= tlX) and (pointP0[1] >= tlY) and (pointP0[0] <= brX) and (pointP0[1] <= brY):
      err = abs(pointP0[0]-curPosX) + abs(pointP0[1]-curPosY)
      #print "err: ", err
      if (err < minErr):
        nearest_old = pointP0
        nearest_old_match = pointP1
        minErr = err
        #print "minErr: ", minErr
        matchFoundTL = True

  if matchFoundTL and (minErr < minErrTh):      
    #adjust TL with OF of best match   
    if debug_print:
      print "OF_x:", nearest_old[0] - nearest_old_match[0]  
      print "OF_y:", nearest_old[1] - nearest_old_match[1]  

    #curPosX += nearest_old[0] - nearest_old_match[0]  
    #curPosY += nearest_old[1] - nearest_old_match[1]  

    curPosX += nearest_old_match[0] - nearest_old[0]
    curPosY += nearest_old_match[1] - nearest_old[1] 
    result = True
  else: 
    if debug_print:
      print "minErr: ", minErr 
  return [result, curPosX,curPosY]

def shouldKeepObjAlive(params=[], detObj=[], labelmap=[]):    
  AGE_BASED = True
  reduceScoreTh  =0.05
  AGE_IDX = 6
  SCORE_IDX = 5
  LBL_IDX = 4

  if AGE_BASED:
    #increase age for each year(frame)
    detObj[AGE_IDX] += 1
    keep_obj_alive = (detObj[AGE_IDX] <= params.maxAgeTh)
  else: #score based.score gets reduced with each passing year (frame)
    #reduce score of tracked objects to give more preference to objs in
    #the currnt frame also it will make sure objs will die after few frames
    detObj[SCORE_IDX] = max(0.0, float(detObj[SCORE_IDX]-reduceScoreTh))
    if type(params.confTh) is dict:                                
      label = int(detObj[LBL_IDX]) 
      label_name = str(get_labelname(labelmap,label)[0])
      keep_obj_alive = detObj[SCORE_IDX] > params.confTh[label_name]
    else:    
      keep_obj_alive = detObj[SCORE_IDX] > params.confTh
  return keep_obj_alive     

def findBestOverlap(trackedBox=[], top_conf=[], top_label_indices=[], top_xmin=[], top_ymin=[], top_xmax=[], top_ymax=[]):
  debug_print = False
  x1 = trackedBox[0]
  y1 = trackedBox[1]
  x2 = trackedBox[2]
  y2 = trackedBox[3]
  label = trackedBox[4]
  score = trackedBox[5]
  #if age_based_check:
  #  age = boxes[:,6]
  #else:
  #  age = 0
  
  # compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)

  if debug_print:
    print "=================="
    print "x1:", x1,
    print "y1:", y1,
    print "x2:", x2,
    print "y2:", y2,
    print "label:", label,
    print "score:", score,
    print "area:", area

  max_overlap = 0.0
  for cand_x1, cand_x2, cand_y1, cand_y2, cand_label, cand_score in zip(top_xmin, top_xmax, top_ymin,top_ymax, top_label_indices, top_conf):
    if cand_label == label:
      if debug_print:
        print "cand_x1:", cand_x1,
        print "cand_y1:", cand_y1,
        print "cand_x2:", cand_x2,
        print "cand_y2:", cand_y2,
        print "cand_label:", cand_label,
        print "cand_score:", cand_score

      # find the largest (x, y) coordinates for the start of
      # the bounding box and the smallest (x, y) coordinates
      # for the end of the bounding box
      xx1 = np.maximum(x1, cand_x1)
      yy1 = np.maximum(y1, cand_y1)
      xx2 = np.minimum(x2, cand_x2)
      yy2 = np.minimum(y2, cand_y2)

      # compute the width and height of the bounding box
      w = np.maximum(0, xx2 - xx1 + 1)
      h = np.maximum(0, yy2 - yy1 + 1)

      if debug_print:
        print "overlap w: " , w,
        print "overlap h: " , h,

      # compute the ratio of overlap
      overlap = (w * h) / area

      if debug_print:
        print "overlap: ", overlap,

      if overlap > max_overlap:
        max_overlap = overlap

  if debug_print:
    print "max_overlap: ", max_overlap
    print "=================="
  return max_overlap    


import cv2
def objTracker(gPoolDetObjs=[], curImageFloat=[], curFrameNum=0, scaleX =
    1.0, scaleY=1.0, offsetX=0, offsetY=0, params=[], labelmap=[],
    raw_dets_cur_frm=[]):
  debug_print = False
  #if overlap of tracked obj is at least have this much overlap with moderate
  #confidence detections then keep it alive
  maxOverLapTh = 0.4
  AGE_IDX = 6
  SCORE_IDX = 5
  LBL_IDX = 4
  curImage = curImageFloat.astype('uint8')
  frame_gray = cv2.cvtColor(curImage, cv2.COLOR_BGR2GRAY)

  ##prune low score detections
  #print "local detection.shape ", raw_dets_cur_frm.shape
  ## Parse the outputs.
  det_label = raw_dets_cur_frm[0,0,:,1]
  det_conf = raw_dets_cur_frm[0,0,:,2]
  det_xmin = raw_dets_cur_frm[0,0,:,3]
  det_ymin = raw_dets_cur_frm[0,0,:,4]
  det_xmax = raw_dets_cur_frm[0,0,:,5]
  det_ymax = raw_dets_cur_frm[0,0,:,6]
  #det_cur_mod_conf = []
  #moderateTh = 0.2


  #for idx, det_conf in enumerate(det_confs):
  #  if det_conf > moderateTh:
  #    det_cur_mod_conf.append([det_xmins[idx],det_ymins[idx],det_xmaxs[idx],det_ymaxs[idx],det_labels[idx],det_confs[idx]])
  modConfTh = 0.12
  if type(modConfTh) is dict:
    confThList = [None] * len(det_label_list)
    for i, det_label_cur_obj in enumerate(det_label_list): 
      if(det_label_cur_obj <> -1):
        modConfThList[i] = modConfTh[str(get_labelname(labelmap,det_label_cur_obj)[0])] 
      else:  
        #some thing went wrong. Set conservative th
        modConfThList[i] = 1.0
    top_indices = [i for i, conf in enumerate(det_conf) if(conf > modConfThList[i])]
  else:  
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= modConfTh]
  
  top_conf = det_conf[top_indices]
  top_label_indices = det_label[top_indices].tolist()
  top_labels = get_labelname(labelmap, top_label_indices)
  top_xmin = det_xmin[top_indices]
  top_ymin = det_ymin[top_indices]
  top_xmax = det_xmax[top_indices]
  top_ymax = det_ymax[top_indices]

  #print "cur scale image size: W: ", frame_gray.shape[1], "H:", frame_gray.shape[0]

  top_xmin = top_xmin * frame_gray.shape[1]
  top_ymin = top_ymin * frame_gray.shape[0]
  top_xmax = top_xmax * frame_gray.shape[1]
  top_ymax = top_ymax * frame_gray.shape[0]

  #det_cur_mod_conf= np.array(det_cur_mod_conf)
  #print "after pruning with moderate score",  det_cur_mod_conf.shape
  #print det_cur_mod_conf.shape
  #print "det_cur_mod_conf"
  #print det_cur_mod_conf

  if curFrameNum == 0:
    trackedObjsList = []
    trackedObjs = np.asarray(trackedObjsList)
  else: #initalization in first frame
    #cap = cv2.VideoCapture('slow.flv')
    # Parameters for lucas kanade optical flow
    #lk_params = dict( winSize  = (15,15),
    #                  maxLevel = 5,
    #                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Parameters taken for TI SFM OF code 
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 5,
                      criteria = (cv2.TERM_CRITERIA_EPS +
                      cv2.TERM_CRITERIA_COUNT, 20, 0.01))

    # Create some random colors. ma

    maxCornersForViz = 1000 #100
    color = np.random.randint(0,255,(maxCornersForViz*len(gPoolDetObjs),3))

    #bbox = (int(xmin*scaleX)+offsetX, int(ymin*scaleY)+offsetY, int(xmax*scaleX)+offsetX, int(ymax*scaleY)+offsetY, label, score)
    #Get det objs from full image to current scale by reversing the above
    #process
    if debug_print:
      print "gPoolDetObjs: "  
      print gPoolDetObjs
    objPoolInCurScale = gPoolDetObjs.copy()
    for idx, detObj in enumerate(gPoolDetObjs):
      objPoolInCurScale[idx][0] = (objPoolInCurScale[idx][0] - offsetX) / scaleX
      objPoolInCurScale[idx][1] = (objPoolInCurScale[idx][1] - offsetY) / scaleY
      objPoolInCurScale[idx][2] = (objPoolInCurScale[idx][2] - offsetX) / scaleX
      objPoolInCurScale[idx][3] = (objPoolInCurScale[idx][3] - offsetY) / scaleY

    if debug_print:
      print "objPoolInCurScale: "  
      print objPoolInCurScale

    p0_init_done = False
    p0 = None
    for idx, detObj in enumerate(objPoolInCurScale ):
      if debug_print: 
        print "XLeft-XRight: ", int(detObj[0]), ":", int(detObj[2]),
        print "YTop-YBot   : ", int(detObj[1]),":", int(detObj[3])
      
      #initalize all pixels to zero (picture completely black)
      mask_detObjs = np.zeros(objTracker.old_gray.shape,np.uint8)

      mask_detObjs[int(detObj[1]):int(detObj[3]),int(detObj[0]):int(detObj[2])] = 1
      areaCurObj = (detObj[2]-detObj[0]) *  (detObj[3]-detObj[1])
      if debug_print: 
        print "areaCurObj: ", areaCurObj 
      # params for ShiTomasi corner detection
      maxCornersCurObj = 100 #100
      minDisanceCurObj = int(max(min(areaCurObj/24,16),4))
      feature_params = dict( maxCorners = maxCornersCurObj,  #100
                             qualityLevel = 0.1,             #0.3
                             minDistance = minDisanceCurObj, #7
                             blockSize = 7 )
      p0_curObj = cv2.goodFeaturesToTrack(objTracker.old_gray, mask = mask_detObjs, **feature_params)
      if p0_curObj is None:
        if debug_print: 
          print "no key point found on det objs"
      elif p0_init_done:
        p0 = np.concatenate((p0, p0_curObj), axis=0)
      else:  
        p0_init_done = True
        p0 = p0_curObj

    if debug_print:
      print "gPoolDetObjs.dtype: ", gPoolDetObjs.dtype
      print "gPoolDetObjs.shape: ", gPoolDetObjs.shape
      print "gPoolDetObjs: ", gPoolDetObjs

    good_new = np.array([])
    good_old = np.array([])
    if p0 is not None:
      p1, st, err = cv2.calcOpticalFlowPyrLK(objTracker.old_gray, frame_gray, p0, None, **lk_params)
    
      if debug_print:
        #print "p1.dtype: ", p1.dtype
        #print "p1.shape: ", p1.shape
        print "p1: ", p1
        print "st: ", st
        print "err: ", err

      # Select good points
      good_new = p1[st==1]
      good_old = p0[st==1]

      if debug_print:
        print "good_old ", good_old

    trackedObjsList = []
    #find the nearset corner point to TL and BR and adjust tracked window
    for idx, detObj in enumerate(gPoolDetObjs):
      tlX = detObj[0] 
      tlY = detObj[1] 
      brX = detObj[2] 
      brY = detObj[3]

      #convert cur position from original image coordinate(after pre resize if it
      #is enabled)to current scale
      tlX = (tlX - offsetX) / scaleX
      brX = (brX - offsetX) / scaleX

      tlY = (tlY - offsetY) / scaleY
      brY = (brY - offsetY) / scaleY

      if debug_print:
        print "=========="
        print "obj in cur scale coordinate : ", tlX, ":", tlY, ":", brX, ":", brY
      resultTLx, tlX, tlY = updateWithNearestKeypointpoint(curPosX=tlX,curPosY=tlY,tlX=tlX,
         tlY=tlY, brX=brX, brY=brY, good_old=good_old, good_new=good_new)
      resultBRy, brX, brY = updateWithNearestKeypointpoint(curPosX=brX,curPosY=brY,tlX=tlX,
         tlY=tlY, brX=brX, brY=brY, good_old=good_old, good_new=good_new)
      
      #if both TL and BR points are having good match
      if resultTLx and resultBRy: 
        if debug_print:
          print "obj propogation successfull!"
          print "propogated obj in cur scale coordicates: ", tlX, ":", tlY, ":", brX, ":", brY

        trackedBoxCurScale = [tlX, tlY, brX, brY, float(detObj[LBL_IDX]), detObj[SCORE_IDX], detObj[AGE_IDX]]
        maxOverLap = findBestOverlap(trackedBox=trackedBoxCurScale, top_conf=top_conf, top_label_indices=top_label_indices, 
            top_xmin=top_xmin, top_ymin=top_ymin, top_xmax=top_xmax, top_ymax=top_ymax)

        if (maxOverLap < maxOverLapTh) and debug_print:
          print "maxOverLap: ", maxOverLap 
          print "But object removed due to less overlap with moderate detectios in frame num: ", curFrameNum
      
        #convert from cur scale position to original image coordinate
        tlX = (tlX*scaleX + offsetX) 
        brX = (brX*scaleX + offsetX) 

        tlY = (tlY*scaleY + offsetY)
        brY = (brY*scaleY + offsetY)

        keep_obj_alive = shouldKeepObjAlive(params=params, detObj=detObj, labelmap=labelmap)    
        if keep_obj_alive and (maxOverLap>=maxOverLapTh):
          trackedBox = [tlX, tlY, brX, brY, float(detObj[LBL_IDX]), detObj[SCORE_IDX], detObj[AGE_IDX]]
          trackedObjsList.append(trackedBox) 
      else:  
        if debug_print:
          print "obj propogation failed!!"
      if debug_print:  
        print "=========="

    trackedObjs = np.asarray(trackedObjsList)
    
    if debug_print:
      print "trackedObjs.dtype: ", trackedObjs.dtype
      print "trackedObjs.shape: ", trackedObjs.shape
      print "trackedObjs: ", trackedObjs 

    draw_tracks = True
    if draw_tracks and good_old.size:
      # draw the tracks
      # Create a mask image for drawing purposes
      mask = np.zeros_like(curImage)
      for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        imageForViz = cv2.circle(curImage,(a,b),5,color[i].tolist(),-1)
      imageForViz= cv2.add(imageForViz,mask)
      cv2.imwrite('./debug/debug_{}.png'.format(str(curFrameNum)),imageForViz)
  
  # Now update the previous frame
  objTracker.old_gray = frame_gray.copy()
  
  return  trackedObjs
##################################################################################################
def wrapMulScls(imageCurFrame, transformer, net, params,
    numScales=4, curFrameNum=0, detObjsFile='', writeBboxMap=[]):

  print_frame_info= False
  from google.protobuf import text_format
  from caffe.proto import caffe_pb2
  
  # load labels
  file = open(params.labelmapFile, 'r')
  labelmap = caffe_pb2.LabelMap()
  text_format.Merge(str(file.read()), labelmap)

  if(curFrameNum == 0):
    wrapMulScls.gPoolDetObjs = np.asarray([])

  imageWidth, imageHeight = imageCurFrame.size
  centerX = imageWidth/2
  centerY = imageHeight/2
  aspectRatio =  float(imageWidth)/imageHeight
  
  if(curFrameNum == 0):
    print "centerX centerY aspectRatio: ", centerX, centerY, aspectRatio

  imageCurFrameAry = deepcopy(imageCurFrame)

  curFrameDrawHandle = ImageDraw.Draw(imageCurFrameAry)

  if (params.arType == ARType.AR_H_SAME_AS_DESC_H):  
    # Initialize scale values
    maxScale = float(imageHeight)/params.tileSizeH
  elif (params.arType == ARType.AR_W_SAME_AS_DESC_W):  
    maxScale = float(imageWidth)/params.tileSizeW
  elif (params.arType == ARType.AR_PRESERVE):  
    #maxScale = float(max(imageWidth, imageHeight))/params.tileSizeW
    maxScale = max(float(imageWidth)/params.tileSizeW, float(imageHeight)/params.tileSizeH)
  else:
    print "wrong AR type"
  
  if(curFrameNum==0):
    print "maxScale ", maxScale
  if (numScales!=1) :
    scaleStep = (maxScale-1.0)/(numScales-1)
    #print "scaleStep ", scaleStep
    # 4 should be numSclales. pls check. FIX_ME:SN
    scaleAry = np.arange(4, dtype=np.float);
    scaleAry[0] = 1
  else:
    #if only 1 scale is used resize image to tile W/H
    #maxScale will take care of this
    scaleAry = np.arange(1, dtype=np.float);
    scaleAry[0] = maxScale

  for scaleIdx in range(1,numScales):
    scaleAry[scaleIdx] =  scaleAry[scaleIdx-1] + scaleStep 
    if(curFrameNum == 0):
      print "scaleIdx ", scaleIdx, scaleAry[scaleIdx]

  detBBoxesCurFrame = []
  #detLabelsCurFrame = []
  #detLabelNamesCurFrame = []
  #detScoresCurFrame = []
  for scaleIdx in range(0,numScales):
    curScale = scaleAry[scaleIdx]
    # get the rect for cur scale in full resolution quantities
    if (params.arType == ARType.AR_H_SAME_AS_DESC_H):  
      # use height same as desc height and get width keeping aspect ratio same as orignal image dimension
      curSclHInFullRes = min(int(params.tileSizeH*curScale), imageHeight)
      curSclWInFullRes = min(int(curSclHInFullRes*aspectRatio), imageWidth)
    elif (params.arType == ARType.AR_W_SAME_AS_DESC_W):
      # use W same as descW and get H keeping aspect ratio same as orignal image dimension
      curSclWInFullRes = min(int(params.tileSizeW*curScale), imageWidth)
      curSclHInFullRes = min(int(curSclWInFullRes/aspectRatio), imageHeight)
    elif (params.arType == ARType.AR_PRESERVE):
      # Preserve original AR 
      curSclWInFullRes = min(int(params.tileSizeW*curScale), imageWidth)
      curSclHInFullRes = min(int(params.tileSizeH*curScale), imageHeight)
    else:
      print "Invalid AR Type: Exiting Now!!!", params.arType
      sys.exit(2)
    
    offsetXmin = max(centerX - curSclWInFullRes/2,0)
    offsetYmin = max(centerY - curSclHInFullRes/2,0)

    offsetXmax = min(offsetXmin + curSclWInFullRes, imageWidth-1)
    offsetYmax = min(offsetYmin + curSclHInFullRes, imageHeight-1)

    imageCurScale = imageCurFrame.crop((offsetXmin, offsetYmin, offsetXmax, offsetYmax))
    # resize cur scale to desc size
    imageCurScale = imageCurScale.resize((params.tileSizeW,params.tileSizeH), Image.ANTIALIAS);
    curScaleX = float(curSclWInFullRes) / params.tileSizeW
    curScaleY = float(curSclHInFullRes) / params.tileSizeH

    if(curFrameNum == 0):
      print "scaleX scaleY curSclWInFullRes curSclHInFullRes offsetXmin offsetXmin,", curScaleX, curScaleY, curSclWInFullRes, curSclHInFullRes,  offsetXmin, offsetYmin

    #processOneCrop expects image array in the range[0,1]
    imageCoreIpCurScale = np.asarray(imageCurScale)
    imageCoreIpCurScale = imageCoreIpCurScale/255.0

    #print('offsetXmin', offsetXmin)

    if params.writeBbox:
      #for objective eval write out all boxes
      confTh=0.01
    else:  
      confTh=params.confTh
    
    [imageDetOp,raw_dets_cur_frm] =  processOneCrop(imageCoreIpCurScale, transformer, net, params.labelmapFile, curFrameDrawHandle,
        detBBoxesCurFrame, offsetX=offsetXmin, offsetY=offsetYmin,scaleX=curScaleX,
        scaleY=curScaleY, aspectRatio=aspectRatio, 
        confTh=confTh)
  
  detObjRectListNPArray = np.array(detBBoxesCurFrame)
  np.set_printoptions(precision=3)
  np.set_printoptions(suppress=True)
  if print_frame_info:
    print "======================================="
    print "curFrameDetObjs:" 
    print detObjRectListNPArray
  if params.enObjTracker:
    wrapMulScls.gPoolDetObjs = objTracker(gPoolDetObjs=wrapMulScls.gPoolDetObjs, 
        curImageFloat=imageCoreIpCurScale*255.0, curFrameNum=curFrameNum, scaleX=curScaleX, scaleY=curScaleY, offsetX=offsetXmin,
        offsetY=offsetYmin, params=params, labelmap=labelmap, raw_dets_cur_frm=raw_dets_cur_frm)
    if print_frame_info:
      print "trackedObjs:"
      print wrapMulScls.gPoolDetObjs
    #propogate detected objs from prev frame to current frame
    if len(wrapMulScls.gPoolDetObjs) > 0 and len(detObjRectListNPArray) > 0:
      wrapMulScls.gPoolDetObjs = np.concatenate((detObjRectListNPArray, wrapMulScls.gPoolDetObjs), axis=0)
    else:
      wrapMulScls.gPoolDetObjs = detObjRectListNPArray
  else:
    wrapMulScls.gPoolDetObjs = detObjRectListNPArray

  if print_frame_info:
    print "global pool after trackedObjs merged with curImgDetObjs:"
    print "gPoolDetObjs: "
    print wrapMulScls.gPoolDetObjs

  #write detected boxes before NMS
  if write_boxes_afr_nms == False:
    if len(wrapMulScls.gPoolDetObjs) and params.writeBbox and ((curFrameNum%params.decFreq) == 0):
      detObjFileHndl = open(detObjsFile, "w")
      writeBoxes(params,wrapMulScls.gPoolDetObjs, detObjFileHndl=detObjFileHndl, 
        writeBboxMap=writeBboxMap, labelmap=labelmap)
      detObjFileHndl.close()

  pick = []

  if print_frame_info:
    print " Objs bef,afr NMS, %d " % (len(wrapMulScls.gPoolDetObjs )),
  if(params.enNMS):
    nmsTh = 0.45 #0.5
    #print detObjRectListNPArray 
    wrapMulScls.gPoolDetObjs = non_max_suppression_fast(wrapMulScls.gPoolDetObjs, nmsTh, pick, age_based_check=True)
    wrapMulScls.gPoolDetObjs = np.array(wrapMulScls.gPoolDetObjs)
    if print_frame_info:
      print "type(wrapMulScls.gPoolDetObjs): ", type(wrapMulScls.gPoolDetObjs)
      print "Global pool After NMS"
      print "gPoolDetObjs.dtype: ", wrapMulScls.gPoolDetObjs.dtype
      print "gPoolDetObjs.shape: ", wrapMulScls.gPoolDetObjs.shape
      print "gPoolDetObjs: "
      print wrapMulScls.gPoolDetObjs

  #write detected boxes Afr NMS
  if write_boxes_afr_nms:
    if len(wrapMulScls.gPoolDetObjs) and params.writeBbox and ((curFrameNum%params.decFreq) == 0):
      detObjFileHndl = open(detObjsFile, "w")
      writeBoxes(params=params,detObjs=wrapMulScls.gPoolDetObjs, detObjFileHndl=detObjFileHndl, 
        writeBboxMap=writeBboxMap, pick=pick, labelmap=labelmap)
      detObjFileHndl.close()

  if(len(detObjRectListNPArray)):
    drawBoxes(params=params,detObjs=wrapMulScls.gPoolDetObjs, drawHandle=curFrameDrawHandle, 
        writeBboxMap=writeBboxMap, labelmap=labelmap)

  if print_frame_info:
    print "======================================="
  return imageCurFrameAry 
##########################################################################################
def vizCNN(vizCNNEn=False, net =''):
  if vizCNNEn:
    all_names = [n for n in net._layer_names]
    for layer_name in all_names: 
      print layer_name
      filename = "/user/a0875091/files/work/github/weiliu89/caffe-ssd/VizCNN/S_Q/" + layer_name + '.jpg'
      visualize_weights(net, layer_name , filename=filename, padding=0)
##########################################################################################
def ssd_detect_video(ipFileName='', opFileName='', deployFileName='',
  modelWeights='', numFrames=1000, tileSizeW=512, tileSizeH=512, labelmapFile='', tileScaleMode=1, 
  resizeW=0, resizeH=0, enNMS=True, numScales=4, arType=0, confTh=0.6,
  enCrop=False, cropMinX=0, cropMinY=0,cropMaxX=0,cropMaxY=0, writeBbox=False, 
  tileStepX=0, tileStepY=0, meanPixVec=[104.0,117.0,123.0], ipScale=1.0,
  writeBboxMap=[], decFreq=1, enObjTracker=False, start_frame_num=0, maxAgeTh=8):    
  ###################################################################################  
  enum_multipleTiles = 0
  enum_multipleScales = 1
  class Params:

    def displayArgs(self):
      print "=================================="
      print "ipFileName %s" , self.ipFileName
      print "opFileName %s" , self.opFileName
      print "deployFileName %s" , self.deployFileName
      print "modelWeights %s" , self.modelWeights
      print "tileSizeW %d" , self.tileSizeW
      print "tileSizeH %d" , self.tileSizeH
      print "numFrames %d" , self.numFrames
      print "start_frame_num %d" , self.start_frame_num
      print "labelmapFile %s" , self.labelmapFile
      print "tileScaleMode %d" , self.tileScaleMode
      print "resizeW %d" , self.resizeW 
      print "resizeH %d" , self.resizeH 
      print "enCrop " , self.enCrop
      print "cropMinX" , self.cropMinX
      print "cropMinY" , self.cropMinY
      print "cropMaxX" , self.cropMaxX
      print "cropMaxY" , self.cropMaxY
      print "enNMS " , self.enNMS
      print "numScales " , self.numScales
      print "arType " , self.arType
      print "confTh " , self.confTh
      print "writeBbox" , self.writeBbox
      print "decFreq" , self.decFreq
      print "writeBboxMap" , self.writeBboxMap
      print "tileStepX" , self.tileStepX
      print "tileStepY" , self.tileStepY
      print "meanPixVec" , self.meanPixVec
      print "ipScale" , self.ipScale
      print "enObjTracker" , self.enObjTracker
      print "maxAgeTh" , self.maxAgeTh
      print "=================================="

    def override(self):  
      if(self.tileStepX == 0):
        self.tileStepX = self.tileSizeW 
      if(self.tileStepY == 0):
        self.tileStepY = self.tileSizeH

  ################################################ 
  params=Params()
  #objTrackerInstance=ObjTrackerClass()

  params.ipFileName = ipFileName
  params.opFileName = opFileName
  params.deployFileName = deployFileName 
  params.modelWeights = modelWeights
  params.tileSizeW = tileSizeW
  params.tileSizeH = tileSizeH 
  params.numFrames = numFrames 
  params.start_frame_num = start_frame_num 
  params.labelmapFile = labelmapFile 
  params.resizeW = resizeW 
  params.resizeH = resizeH 
  params.cropMinX = cropMinX 
  params.cropMinY = cropMinY 
  params.cropMaxX = cropMaxX 
  params.cropMaxY = cropMaxY 
  params.tileScaleMode = tileScaleMode  
  params.enCrop = enCrop 
  params.enNMS = enNMS 
  params.numScales = numScales 
  params.arType = arType 
  params.confTh = confTh 
  params.writeBbox = writeBbox 
  params.decFreq = decFreq 
  params.writeBboxMap = writeBboxMap 
  params.tileStepX = tileStepX
  params.tileStepY = tileStepY
  params.meanPixVec = meanPixVec
  params.ipScale = ipScale
  params.enObjTracker = enObjTracker
  params.maxAgeTh = maxAgeTh

  for idx,data in enumerate(meanPixVec):
    params.meanPixVec[idx] = data * ipScale

  params.override()

  opPath, fileName = os.path.split(params.opFileName)
  print ("opPath : ", opPath)
  if not os.path.exists(opPath):
    os.makedirs(opPath)
    print ("opPath creatd", opPath)
  
  sys.stdout = open(opPath+'/console.log', 'a+')
  sys.stderr = open(opPath+'/console_err.log', 'a+')

  params.displayArgs()

  plt.rcParams['figure.figsize'] = (10, 10)
  plt.rcParams['image.interpolation'] = 'nearest'
  plt.rcParams['image.cmap'] = 'gray'

  # Make sure that caffe is on the python path:
  curWorDir = os.getcwd();
  caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
  os.chdir(caffe_root)
  print("caffe_root: ", os.getcwd())
  sys.path.insert(0, 'python')
  import caffe
  caffe.set_device(1)
  caffe.set_mode_gpu()

  print("before caffe.Net")
  print("params.deployFileName", params.deployFileName)
  print("params.modelWeights", params.modelWeights)
  net = caffe.Net(params.deployFileName,    # defines the structure of the model
          params.modelWeights,  # contains the trained weights
          caffe.TEST)   # use test mode (e.g., don't perform dropout)

  print("After caffe.Net")
  vizCNN(vizCNNEn=False, net=net)
  # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  transformer.set_transpose('data', (2, 0, 1))
  transformer.set_mean('data', np.array(meanPixVec)) # mean pixel
  scale_mul = ipScale * 255.0
  #print('scale_mul: ', scale_mul)
  transformer.set_raw_scale('data', scale_mul)  # the reference model operates on images in [0,255] range instead of [0,1]
  transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

  # set net to batch size of 1
  net.blobs['data'].reshape(1,3,params.tileSizeH,params.tileSizeW)

  isVideoIp = True
  if(os.path.splitext(params.ipFileName)[1] ==''):
    isVideoIp = False    

  isVideoOp = True
  if(os.path.splitext(params.opFileName)[1] ==''):
    isVideoOp = False    

  fps = 30
  if isVideoIp:
    vidIp = imageio.get_reader(params.ipFileName,  'ffmpeg')
    fps = vidIp.get_meta_data()['fps']

  if isVideoOp:
    #quality=10.0
    vidOp = imageio.get_writer(params.opFileName,  'ffmpeg', fps=fps)

  if isVideoIp:
    for num,im in enumerate(vidIp):
      if num > (params.numFrames+params.start_frame_num-1):
        break;
      
      #skip init few frames
      if(num<params.start_frame_num): 
        continue;

      curFrameNum=num-params.start_frame_num

      #skip every frame in the batch of decFreq 
      if((curFrameNum%params.decFreq) != 0): 
        continue;

      print num,
      sys.stdout.flush()
      image = vidIp.get_data(num)
      #get_data gives image in "np array" with range [0-255]
      #while wrapperMultipleXXX expects an "image" with range[0-255] 
      imageCurFrame = Image.fromarray((image).astype(np.uint8))
      #print "image block: ", image[0:10,0:10] 

      if params.enCrop:
        imageCurFrame = imageCurFrame.crop((cropMinX,cropMinY,cropMaxX,cropMaxY))

      bareFile, file_extension = os.path.splitext(params.opFileName)
      detObjsFile = '{}_{:06d}.txt'.format(bareFile, num)
      
      if(params.tileScaleMode == enum_multipleTiles):
        combinedCurFrameOp = wrapMulTiles(imageCurFrame, transformer, net, params, 
           curFrameNum=curFrameNum,detObjsFile=detObjsFile, writeBboxMap=writeBboxMap) 
      else:
        combinedCurFrameOp = wrapMulScls(imageCurFrame, transformer, net, params, 
           numScales=params.numScales, curFrameNum=curFrameNum,detObjsFile=detObjsFile,
           writeBboxMap=writeBboxMap)

      if isVideoOp:
        vidOp.append_data(np.asarray(combinedCurFrameOp))
      else:
        image_file_name = "{}.jpg".format(num)
        combinedCurFrameOp.save(os.path.join(params.opFileName,image_file_name))

    os.chdir(curWorDir)
  else:
    # input is image folder
    for root, dirs, filenames in os.walk(params.ipFileName):
      for num,f in enumerate(sorted(filenames)):
        if num > params.numFrames-1:
          break;
        print f,
        imageCurFrame = caffe.io.load_image(os.path.join(root, f))

        #load_image gives image in "np array" with range [0-1]
        #while wrapperMultipleXXX expects an "image" with range[0-255] 
        imageCurFrame = imageCurFrame*255

        imageCurFrame = Image.fromarray(imageCurFrame.astype(np.uint8))
        if params.enCrop:
          imageCurFrame = imageCurFrame.crop((cropMinX,cropMinY,cropMaxX,cropMaxY))

        #print 'imageCurFrame.type: ', imageCurFrame.type 
        #print "image block: ", imageCurFrame[0:10,0:10] 
        
        detObjsFile = os.path.splitext(os.path.join(params.opFileName, f))[0]+'.txt'  
        print 'detObjsFile: ', detObjsFile  

        if(params.tileScaleMode == enum_multipleTiles):
           combinedCurFrameOp = wrapMulTiles(imageCurFrame, transformer, net, params, 
             curFrameNum=num, detObjsFile=detObjsFile, writeBboxMap=writeBboxMap) 
        else:
           combinedCurFrameOp = wrapMulScls(imageCurFrame, transformer, net, params, 
             numScales=params.numScales, curFrameNum=num, detObjsFile=detObjsFile, writeBboxMap=writeBboxMap)

        if isVideoOp:
          vidOp.append_data(np.asarray(combinedCurFrameOp))
        else:
          combinedCurFrameOp.save(os.path.join(params.opFileName,f))
    os.chdir(curWorDir)

  #sys.stdout.close()
  #sys.stderr.close()
  return  
