# import the necessary packages
import numpy as np
debug_print = False
#default: sort_with_score = False i.e. sort with y_bot
sort_with_score = True

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh, pick, age_based_check=False):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    if debug_print:
      print "Start of new frame"

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes    
    #pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    label = boxes[:,4]
    score = boxes[:,5]
    if age_based_check:
      age = boxes[:,6]
    else:
      age = 0

    if debug_print:
      print "boxes:", boxes
      print "x1:", x1
      print "label:", label
      print "score:", score
      print "age:", age

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    if sort_with_score:
      #when age based score is on. Give preference to younger boxes. As long
      #as score is within 0-1.0, this logic will work
      idxs = np.argsort(score-age)
    else:  
      idxs = np.argsort(y2)

    if debug_print:
      print "area: ", area
      print "idxs: ", idxs
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        if debug_print:
          print "========="
          print "picked this bbox: ", i
          print "score: ", score[i]
          print "age: ", age[i]

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        if debug_print:
          print "w: " , w
          print "h: " , h

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        if debug_print:
          print "overlap: ", overlap

        labelTemp = label[idxs[:last]]
        curBoxLabel = label[i] 

        # delete all indexes from the index list that have
        #idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        cond = np.where((overlap > overlapThresh) & (labelTemp == curBoxLabel) )
        idxs = np.delete(idxs, np.concatenate(([last], cond[0])))
        if debug_print:
          print "cond", cond[0]
          print "after deleting few bboxes"
          print "area: ", area
          print "idxs: ", idxs
          print "age: ", age
          print "score: ", score

    # return only the bounding boxes that were picked using the
    # integer data type
    #SN: I don't know logic behind this
    #return boxes[pick].astype("int")
    return boxes[pick]
