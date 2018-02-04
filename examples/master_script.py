from ssd_detect_video import ssd_detect_video
import sys
import os
# run this script from ssd-caffe/examples
#-n numframes
#-t  tile size ( after resize tile will be applicable)
#-r resize input image to this width and height
############################################################################################################################################
EVAL_UTIL = '~/files/work/bitbucket_TI/devkit-datasets/Kitti/devkit_object/cpp/evaluate_object'
############################################################################################################################################

###################################
#Define Test Videos
###################################ssd720x368_ssdJacintoNetV2_iter_9600.caffemodel

lindau={}
lindau['IpPath']='/data/mmcodec_video2_tier3/datasets/ObjectDetect/data/vision-dataset/annotatedVbb/data-TIRoadDrive2/videos/ti/lindau/'
lindau['fileNames'] = ['V110_2015sept_103_VIRB_VIRB0001.MP4', 'V111_2015sept_104_VIRB_VIRB0001.MP4','V105_2015sept_100_VIRB_VIRB0031_0m_10m.MP4', 'V106_2015sept_100_VIRB_VIRB0031_10m_10m.MP4']

lindau_genvbb={}
lindau_genvbb['IpPath']='/data/mmcodec_video2_tier3/datasets/ObjectDetect/data/vision-dataset/annotatedVbb/data-TIRoadDrive2/videos/ti/lindau/'
lindau_genvbb['fileNames'] = ['V103_2015sept_100_VIRB_VIRB0022.MP4', 'V104_2015sept_100_VIRB_VIRB0023.MP4', 'V107_2015sept_101_VIRB_VIRB0032.MP4', 'V108_2015sept_101_VIRB_VIRB0037.MP4', 'V109_2015sept_102_VIRB_VIRB0003.MP4', 'V120_2015sept_105_VIRB_VIRB0001.MP4', 'V121_2015sept_105_VIRB_VIRB0005.MP4', 'V123_2015sept_105_VIRB_VIRB0007.MP4', 'V124_2015sept_105_VIRB_VIRB0008.MP4']
  
munich = {}
munich['IpPath']='/data/mmcodec_video2_tier3/datasets/ObjectDetect/data/vision-dataset/annotatedVbb/data-TIRoadDrive2/videos/ti/munich/'
munich['fileNames'] = ['V008_2015jul_VIRB0008_7m_end.MP4', 'V007_2015jul_VIRB0008_0m_7m.MP4']

#FIX_ME:combine munich_genvbb and munich_genvbb_leftover  
munich_genvbb = {}
munich_genvbb['IpPath']='/data/mmcodec_video2_tier3/datasets/ObjectDetect/data/vision-dataset/annotatedVbb/data-TIRoadDrive2/videos/ti/munich/'
munich_genvbb['fileNames'] = ['V002_2015jul_VIRB0004.MP4', 'V003_2015jul_VIRB0005.MP4',  'V005_2015jul_VIRB0006.MP4', 'V006_2015jul_VIRB0007.MP4', 'V141_2015sept_VIRB0004_17m_end.MP4', 'V154_2015oct_107_VIRB_VIRB0003.MP4', 'V155_2015oct_107_VIRB_VIRB0004.MP4', 'V156_2015oct_107_VIRB_VIRB0005.MP4', 'V157_2015oct_107_VIRB_VIRB0006.MP4', 'V158_2015oct_107_VIRB_VIRB0007.MP4']

munich_genvbb_leftover = {}
munich_genvbb_leftover['IpPath']='/data/mmcodec_video2_tier3/datasets/ObjectDetect/data/vision-dataset/annotatedVbb/data-TIRoadDrive2/videos/ti/munich/'
munich_genvbb_leftover['fileNames'] = ['V150_2015oct_106_VIRB_VIRB0006.MP4','V151_2015oct_106_VIRB_VIRB0007.MP4','V152_2015oct_107_VIRB_VIRB0001.MP4','V153_2015oct_107_VIRB_VIRB0002.MP4']

test_genvbb = {}
test_genvbb['IpPath']='/data/mmcodec_video2_tier3/users/soyeb/AVISynth/'
test_genvbb['fileNames'] = ['V002_2015jul_VIRB0004_300fr.MP4']

V140 = {}
V140['IpPath']='/data/mmcodec_video2_tier3/datasets/ObjectDetect/data/vision-dataset/annotatedVbb/data-TIRoadDrive2/videos/ti/munich/'
V140['fileNames'] = ['V140_2015sept_VIRB0004_0m_17m.MP4']

# Test GTAV Videos (Playing For Data)
gta = {}
gta['IpPath']="/data/mmcodec_video2_tier3/datasets/ObjectDetect/data/vision-dataset/annotatedVbb/data-TIRoadDrive2/videos/other/PlayingForData/"
gta['fileNames'] = ['GTAV_P1.MP4', 'GTAV_P2.MP4']

# Test GTAV Videos (TI captured)
gta_ti_cpatured = {}
gta_ti_cpatured['IpPath']="/data/mmcodec_video2_tier3/users/prashanth/ATD_GTAV/GTA_Videos/"
gta_ti_cpatured['fileNames'] = ['test4.mp4', 'test5.mp4', 'test6.mp4']

#KITTI
kitti = {}
kitti['IpPath'] = "/data/mmcodec_video2_tier3/datasets/ObjectDetect/data/vision-dataset/annotatedVbb/data-TIRoadDrive2/videos/other/kitti_object/training/"
kitti['fileNames'] = ['part1.MP4', 'part2.MP4']

city_test = {}
city_test['IpPath']="/data/mmcodec_video2_tier3/datasets/ObjectDetect/data/vision-dataset/annotatedVbb/data-TIRoadDrive2/videos/other/cityscape/val/"
city_test['fileNames'] = ['frankfurt.MP4', 'lindau.MP4', 'munster.MP4']

city_train = {}
city_train['IpPath']="/data/mmcodec_video2_tier3/datasets/ObjectDetect/data/vision-dataset/annotatedVbb/data-TIRoadDrive2/videos/other/cityscape/train/"
#aachen was part of training set , others were part of test set
city_train['fileNames'] = ['aachen.MP4']

#cityscpes videos 
city_video = {}
city_video['IpPath']="/data/mmcodec_video2_tier3/datasets/ObjectDetect/data/segmentation-dataset/cityscapes/data/leftImg8bit/demoVideo/"
city_video['fileNames'] = ['stuttgart_00.MP4', 'stuttgart_01.MP4', 'stuttgart_02.MP4']

#for objective comp 
city_test_obj = {}
city_test_obj['IpPath']="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_cityscapes/data/train/Videos/"
city_test_obj['fileNames'] = ['cityscape.MP4']

city_512_test_obj = {}
city_512_test_obj['IpPath']="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_cityscapes_512/data/train/Videos/"
city_512_test_obj['fileNames'] = ['city_512_test.MP4']

city_720x368_test_obj = {}
city_720x368_test_obj['IpPath']="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_cityscapes_720x368/data/train/Videos/"
city_720x368_test_obj['fileNames'] = ['city_720x368_test.MP4']

ti_201708_test_obj = {}
ti_201708_test_obj['IpPath']="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/"
ti_201708_test_obj['fileNames'] = ['ti_201708_test.MP4']

voc0712 ={}
voc0712['IpPath']= "/user/a0875091/files/data/datasets/object-detect/other/pascal-voc/combined/VOCdevkit/TIUtils/"
#voc0712['fileNames'] = ['VOC2007test.MP4', 'VOC2007TrainVal.MP4', 'VOC2012TrainVal.MP4']
voc0712['fileNames'] = ['VOC2007test.MP4']

prescan = {}
prescan['IpPath'] = "/data/mmcodec_video2_tier3/users/prashanth/Prescan/"
prescan['fileNames'] = ['CameraSensor_1.avi', 'CameraSensor_2.avi']

ti_demo = {}
ti_demo['IpPath'] = "/data/mmcodec_video2_tier3/datasets/TI/RoadDrive/GermanyVideos/Munich_5th_Oct_Thorsten/107_VIRB/"
ti_demo['fileNames'] = ['VIRB0002.MP4', 'VIRB0003_4400sfr_5000nfr.MP4']

V003 = {}
V003['IpPath'] = "/data/mmcodec_video2_tier3/datasets/TI/RoadDrive/GermanyVideos/Munich_5th_Oct_Thorsten/107_VIRB/"
V003['fileNames'] = ['VIRB0003_4400sfr_5000nfr.MP4']

V002 = {}
V002['IpPath'] = "/data/mmcodec_video2_tier3/datasets/TI/RoadDrive/GermanyVideos/Munich_5th_Oct_Thorsten/107_VIRB/"
V002['fileNames'] = ['VIRB0002.MP4']

fisheye = {}
fisheye['IpPath'] = "/data/mmcodec_video2_tier3/datasets/ObjectDetect/data/other-dataset/fisheye/"
fisheye['fileNames'] = ['DriveA_720x368.MP4']

V106={}
V106['IpPath']='/data/mmcodec_video2_tier3/datasets/ObjectDetect/data/vision-dataset/annotatedVbb/data-TIRoadDrive2/videos/ti/lindau/'
V106['fileNames'] = ['V106_2015sept_100_VIRB_VIRB0031_10m_10m.MP4']

V008={}
V008['IpPath']='/data/mmcodec_video2_tier3/datasets/ObjectDetect/data/vision-dataset/annotatedVbb/data-TIRoadDrive2/videos/ti/munich/'
V008['fileNames'] = ['V008_2015jul_VIRB0008_7m_end.MP4']

#############################################################################
def execute_test(params):
  #city, voc0712, gta_ti_cpatured
  #datasets = [lindau, munich, city_test, voc0712, ti_201708_test_obj, city_video]
  #datasets = [lindau, munich, city_test, ti_demo, V140]
  #datasets = [V003]
  #datasets = [city_test_obj]
  #datasets = [city_512_test_obj, ti_201708_test_obj]
  #datasets = [ti_demo, city_test, city_train]
  datasets = [ti_demo,V008]
  #datasets = [munich_genvbb, lindau_genvbb]
  #datasets = [munich_genvbb_leftover]
  #datasets = [test_genvbb]
  for dataset in datasets:
    if dataset == ti_201708_test_obj: 
     #when writing bbox map category from src->dst
     WRITE_BBOX_MAP=[['car','vehicle'], ['train','vehicle'],['bus','vehicle']]
     category = "TI"
    else: 
     WRITE_BBOX_MAP=[]
     category = "CITY"
  
    print "dataset:", dataset
    for fileName in dataset['fileNames']:
      ipFileName=dataset['IpPath']+fileName
      opFileName=params.OpPath+fileName
      opFileName = opFileName.replace('.MP4', '_detOp.MP4')
      #opFileName = params.OpPath
      print ipFileName, opFileName
      ssd_detect_video(ipFileName=ipFileName,
          opFileName=opFileName, deployFileName=params.Deploy, modelWeights=params.ModelWeights,
          numFrames=params.NumFrames, tileSizeW=params.TileSizeW, tileSizeH=params.TileSizeH,labelmapFile=params.LabelMap, 
          tileScaleMode=params.MUL_TILE_SCL_MD, resizeW=params.ResizeW,resizeH=params.ResizeH,enNMS=params.NMS_FLAG, 
          numScales=params.NUM_SCALES,arType=params.AR_TYPE, confTh=params.CONF_TH, writeBbox=params.WRITE_BBOX, 
          meanPixVec=params.MEAN_PIX_VEC, ipScale=params.IP_SCALE, writeBboxMap=WRITE_BBOX_MAP,
          enCrop=params.enCrop, cropMinX=params.cropMinX, cropMinY=params.cropMinY,
          cropMaxX=params.cropMaxX, cropMaxY=params.cropMaxY, decFreq = params.decFreq, 
          enObjTracker=params.enObjTracker, start_frame_num=params.start_frame_num, maxAgeTh=params.maxAgeTh)
      
      if params.EVAL_OBJ:
        gt_prefix, file_extension = os.path.splitext(fileName)
        gt_prefix = gt_prefix + '_' 
        det_op_prefix = gt_prefix + 'detOp_'
        result = gt_prefix + 'result.txt'
        cmd = "{} {} {} 0 {} {} {} {} > {}{} ".format(EVAL_UTIL, params.OpPath,
           dataset['IpPath'], params.NumFrames, gt_prefix, det_op_prefix, category, params.OpPath, result)
        print "cmd: ", cmd
        os.system(cmd)\

  return
############################################
#common params for all tests
def set_common_params(params):
  #crop params
  #crop is the first operation done if enabled
  params.enCrop=True
  params.cropMinX = 0
  params.cropMinY = 0
  params.cropMaxX = 0
  params.cropMaxY = 0

  if params.enCrop:
    #use positive y_offset if horizon is below the middle of the picture
    y_offset = -80 #40
    crop_w = 1280 #
    crop_h = 720
    image_w = 1920
    image_h = 1080 
    params.cropMinX=(image_w-crop_w)/2
    params.cropMinY=((image_h-crop_h)/2) + y_offset
    params.cropMaxX=params.cropMinX+crop_w
    params.cropMaxY=params.cropMinY+crop_h
    
  #32500 for V140 VBB
  params.NumFrames=3000
  params.start_frame_num=0
  # ResizeW and ResizeH are not used when multiple sclales are used
  # When these options are non zero first crop (if enabled) then resizing will be done before anything else
  params.ResizeW=0
  params.ResizeH=0
  params.NMS_FLAG=1
  params.MUL_TILE_SCL_MD=1  # 0:multiple tiles, 1: multiple scales
  params.NUM_SCALES=1
  params.AR_TYPE=2          # 0:height according to tile height (def), 1:width according to tile width, 2: maintain original AR till it hits limit
  #could be either dict type or int (all cat will have same confTh) 
  #params.CONF_TH=0.6
  #category specific thresholds
  #params.CONF_TH = {'person':0.2,'vehicle':0.6,'trafficsign':0.4,'cyclist':0.2,} # pick all det obj at least with this score (def=0.6)
  #params.TILE_STEP_X=192
  params.WRITE_BBOX = params.EVAL_OBJ
  params.decFreq = 1
  params.enObjTracker=True
  params.maxAgeTh=30
  return

############################################
class Params:
  def displayArgs(self):
    print "===========params instantiated======================="


#############################################################################
#Model="./models/TITrainedOp/20160928_TI/SSD_500x500/store/VGG_20160928_TI_SSD_500x500_iter_19147.caffemodel"
#params.Deploy="./models/TITrainedOp/20160928_TI/SSD_500x500/store/deploy.prototxt"
#params.LabelMap="./data/TI2016/labelmap_ti.prototxt"

#Model="./models/TITrainedOp/20161101_PASCAL_CITYSCAPE/SSD_CITYSCAPE_1024x1024/store/VGG_VOC0712_CITYSCAPE_SSD_CITYSCAPE_1024x1024_iter_15000.caffemodel"
#params.Deploy="./models/TITrainedOp/20161101_PASCAL_CITYSCAPE/SSD_CITYSCAPE_1024x1024/store/deploy.prototxt"
#params.LabelMap="./data/cityscape/labelmap_cityscape.prototxt"

#Best model so far: Pre-trained PASCAL + CITYSCAPES 20000 itr
#params.ModelWeights="./models/TITrainedOp/20161101_PASCAL_CITYSCAPE/SSD_CITYSCAPE_500x500/store/VGG_VOC0712_CITYSCAPE_SSD_CITYSCAPE_500x500_iter_20000.caffemodel"
#params.Deploy="./models/TITrainedOp/20161101_PASCAL_CITYSCAPE/SSD_CITYSCAPE_500x500/store/deploy.prototxt"
#params.LabelMap="./data/cityscape/labelmap_cityscape.prototxt"

#Pre-trained PASCAL given by SSD
#Model="./models/VGGNet/VOC0712_preTrained/SSD_500x500/VGG_VOC0712_SSD_500x500_iter_60000.caffemodel"
#params.Deploy="./models/VGGNet/VOC0712_preTrained/SSD_500x500/deploy.prototxt"
#params.LabelMap="./models/VGGNet/VOC0712_preTrained/SSD_500x500/labelmap_voc.prototxt"

#Pre-trained COCO model given by SSD
#Model="./models/VGGNet/coco_preTrained/SSD_500x500/VGG_coco_SSD_500x500_iter_200000.caffemodel"
#params.Deploy="./models/VGGNet/coco_preTrained/SSD_500x500/deploy.prototxt"
#params.LabelMap="./models/VGGNet/coco_preTrained/SSD_500x500/labelmap_coco.prototxt"

#Pascal traning with min ratio 0.03
#Model="./models/TITrainedOp/VGGNet/VOC0712/SSD_1024x1024/store/VGG_VOC0712_SSD_1024x1024_iter_26537.caffemodel"
#params.Deploy="./models/TITrainedOp/VGGNet/VOC0712/SSD_1024x1024/store/deploy.prototxt"
#params.LabelMap="./models/VGGNet/VOC0712_preTrained/SSD_500x500/labelmap_voc.prototxt"

#Model="./models/TITrainedOp/VGGNet/VOC0712/SSD_1024x1024_minRatio15/VGG_VOC0712_SSD_1024x1024_iter_7076.caffemodel"
#params.Deploy="./models/TITrainedOp/VGGNet/VOC0712/SSD_1024x1024_minRatio15/deploy.prototxt"
#params.LabelMap="./models/VGGNet/VOC0712_preTrained/SSD_500x500/labelmap_voc.prototxt"

# 1024 size with min ratio 0.03 (down from 0.15 original)
#Model="./models/TITrainedOp/20161101_PASCAL_CITYSCAPE/SSD_CITYSCAPE_MINRATIO3_1024x1024/store/VGG_VOC0712_CITYSCAPE_SSD_CITYSCAPE_MINRATIO3_1024x1024_iter_51000.caffemodel"
#params.Deploy="./models/TITrainedOp/20161101_PASCAL_CITYSCAPE/SSD_CITYSCAPE_MINRATIO3_1024x1024/store/deploy.prototxt"
#params.LabelMap="./data/cityscape/labelmap_cityscape.prototxt"

# 1000x500 size with min ratio 0.03 (down from 0.15 original)
#Model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/TITrainedOp/20161101_PASCAL_CITYSCAPE/SSD_PASCAL_CITYSCAPE1000x500/store/VGG_VOC0712_CITYSCAPE_SSD_PASCAL_CITYSCAPE1000x500_iter_21000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/TITrainedOp/20161101_PASCAL_CITYSCAPE/SSD_PASCAL_CITYSCAPE1000x500/store/deploy.prototxt"
#params.LabelMap="./data/cityscape/labelmap_cityscape.prototxt"

# 1000x500 size with ratio 0.03-0.50 (original 0.15-0.95)
#Model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/TITrainedOp/20161101_PASCAL_CITYSCAPE/SSD_PASCAL_CITYSCAPE_RATIO_03_50_1000x500/store/VGG_VOC0712_CITYSCAPE_SSD_PASCAL_CITYSCAPE_RATIO_03_50_1000x500_iter_22000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/TITrainedOp/20161101_PASCAL_CITYSCAPE/SSD_PASCAL_CITYSCAPE_RATIO_03_50_1000x500/store/deploy.prototxt"
#params.LabelMap="./data/cityscape/labelmap_cityscape.prototxt"

# 500x500 size with ratio more aspect ratios
#Model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGGNet/VOC0712/SSD_ExhaustAR_500x500/store/VGG_VOC0712_SSD_ExhaustAR_500x500_iter_21609.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGGNet/VOC0712/SSD_ExhaustAR_500x500/store/deploy.prototxt"
#params.LabelMap="./models/VGGNet/VOC0712_preTrained/SSD_500x500/labelmap_voc.prototxt"

# 500x500 size with Jacinto net (PASCAL), Exustive AR 
#Model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/VOC0712/SSD_BN_500x500_exustiveAR/store/Jacinto_VOC0712_SSD_BN_500x500_iter_66481.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/VOC0712/SSD_BN_500x500_exustiveAR/store/deploy.prototxt"
#params.LabelMap="./models/VGGNet/VOC0712_preTrained/SSD_500x500/labelmap_voc.prototxt"

# 500x500 size with Jacinto net (PASCAL) 
#Model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/VOC0712/SSD_BN_500x500/store/JACINTO_VOC0712_SSD_BN_500x500_iter_85125.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/VOC0712/SSD_BN_500x500/store/deploy.prototxt"
#params.LabelMap="./models/VGGNet/VOC0712_preTrained/SSD_500x500/labelmap_voc.prototxt"

# 500x500 size with Jacinto net (PASCAL->CITY) 
#Model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/CITY/SSD_BN_500x500/store/JACINTO_CITY_SSD_BN_500x500_iter_78204.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/CITY/SSD_BN_500x500/store/deploy.prototxt"
#params.LabelMap="./data/cityscape/labelmap_cityscape.prototxt"

# 500x500 size with Jacinto net (My trained model) (PASCAL) 
#Model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/VOC0712/SSD_BN_MyPreTrained_500x500/store/JACINTO_VOC0712_SSD_BN_MyPreTrained_500x500_iter_83000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/VOC0712/SSD_BN_MyPreTrained_500x500/store/deploy.prototxt"
#params.LabelMap="./models/VGGNet/VOC0712_preTrained/SSD_500x500/labelmap_voc.prototxt"

# 500x500 size with Jacinto net (My trained model) (PASCAL->CITY) 
#Model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/CITY/SSD_BN_MyPreTrained_500x500/store/JACINTO_CITY_SSD_BN_MyPreTrained_500x500_iter_23000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/CITY/SSD_BN_MyPreTrained_500x500/store/deploy.prototxt"
#params.LabelMap="./data/cityscape/labelmap_cityscape.prototxt"

# 500x500 size with Jacinto net (My trained model) (PASCAL->CITY->TI) 
#Model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2016/SSD_BN_MyPreTrained_500x500/store/JACINTO_TI2016_SSD_BN_MyPreTrained_500x500_iter_24843.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2016/SSD_BN_MyPreTrained_500x500/store/deploy.prototxt"
#params.LabelMap="./data/TI2016/labelmap_ti.prototxt"

# 500x500 size with Jacinto net (My trained model) (PASCAL->CITY->TI) 
#Model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2017/SSD_BN_MyPreTrained_500x500/store/JACINTO_TI2017_SSD_BN_MyPreTrained_500x500_iter_32847.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2017/SSD_BN_MyPreTrained_500x500/store/deploy.prototxt"
#params.LabelMap="./data/TI2017/labelmap_ti.prototxt"

#Model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2017/SSD_BN_RemovedIgnored_MyPreTrained_512x512_freeze_True/store/JACINTO_TI2017_SSD_BN_RemovedIgnored_MyPreTrained_512x512_freeze_True_iter_12000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2017/SSD_BN_RemovedIgnored_MyPreTrained_512x512_freeze_True/store/deploy.prototxt"
#params.LabelMap="./data/TI2017/labelmap_ti.prototxt"


#Model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2017_Tiny/SSD_BN_512x512_freeze_True_wideAR_True/JACINTO_TI2017_Tiny_SSD_BN_512x512_freeze_True_wideAR_True_iter_2000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2017_Tiny/SSD_BN_512x512_freeze_True_wideAR_True/deploy.prototxt"
#params.LabelMap="./data/TI2017_Tiny/labelmap_ti.prototxt"

#TI2017_Tiny
#Model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI2017_Tiny/SSD_BN_512x512_freeze_True_wideAR_True/store/VGG_TI2017_Tiny_SSD_BN_512x512_freeze_True_wideAR_True_iter_5000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI2017_Tiny/SSD_BN_512x512_freeze_True_wideAR_True/store/deploy.prototxt"
#params.LabelMap="./data/TI2017_Tiny/labelmap_ti.prototxt"

#TI2017_Tiny_MulTiles_MixedSizes
#Model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI2017_Tiny_MulTiles/SSD_BN_512x512_freeze_True_wideAR_True/store/VGG_TI2017_Tiny_MulTiles_SSD_BN_512x512_freeze_True_wideAR_True_iter_5000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI2017_Tiny_MulTiles/SSD_BN_512x512_freeze_True_wideAR_True/store/deploy.prototxt"
#params.LabelMap="./data/TI2017_Tiny_MulTiles_/labelmap_ti.prototxt"

#TI2017_Tiny_MulTiles
#Model="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2017_Tiny_MulTiles_minSize12x32/SSD_BN_512x512_freeze_True_wideAR_False/store/JACINTO_TI2017_Tiny_MulTiles_minSize12x32_SSD_BN_512x512_freeze_True_wideAR_False_iter_2000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2017_Tiny_MulTiles_minSize12x32/SSD_BN_512x512_freeze_True_wideAR_False/store/deploy.prototxt"
#params.LabelMap="./data/TI2017_Tiny_MulTiles_minSize24x56/labelmap_ti.prototxt"

#TI2017_V105
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2017_V105/SSD_BN_512x512_freeze_True_wideAR_False_minRatio_15/jobs/JACINTO_TI2017_V105_SSD_BN_512x512_freeze_True_wideAR_False_minRatio_15_iter_25715.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2017_V105/SSD_BN_512x512_freeze_True_wideAR_False_minRatio_15/jobs/deploy.prototxt"
#params.LabelMap="./data/TI2017_V105/labelmap_ti.prototxt"

#VGG TI2017_V105
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI2017_V105/SSD_BN_512x512_freeze_True_wideAR_False_minRatio_15/jobs/VGG_TI2017_V105_SSD_BN_512x512_freeze_True_wideAR_False_minRatio_15_iter_25715.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI2017_V105/SSD_BN_512x512_freeze_True_wideAR_False_minRatio_15/jobs/deploy.prototxt"
#params.LabelMap="./data/TI2017_V105/labelmap_ti.prototxt"

#JACINTO TI2017_V105
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2017_V105/SSD_BN_512x512_Pre_IMGNET_VOC0712_freeze_True_wideAR_False_minRatio_15/jobs/JACINTO_TI2017_V105_SSD_BN_512x512_Pre_IMGNET_VOC0712_freeze_True_wideAR_False_minRatio_15_iter_25715.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2017_V105/SSD_BN_512x512_Pre_IMGNET_VOC0712_freeze_True_wideAR_False_minRatio_15/jobs/deploy.prototxt"
#params.LabelMap="./data/TI2017_V105/labelmap_ti.prototxt"

#VGG TI2017_V7_106
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI2017_V7_106/SSD_BN_512x512_Pre_IMGNET_VOC0712_freeze_True_wideAR_False_minRatio_15/jobs/VGG_TI2017_V7_106_SSD_BN_512x512_Pre_IMGNET_VOC0712_freeze_True_wideAR_False_minRatio_15_iter_19000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI2017_V7_106/SSD_BN_512x512_Pre_IMGNET_VOC0712_freeze_True_wideAR_False_minRatio_15/jobs/deploy.prototxt"
#params.LabelMap="./data/TI2017_V7_106/labelmap_ti.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170605_TI_7_106_VGG_retest_del_later/"
##if BN is placed at the beginning then mean should not be explictly subtracted so set to 0
#params.TileSizeW=512
#params.TileSizeH=512
#params.MEAN_PIX_VEC =[104,117,123]

#JACINTO TI2017_V7_106
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2017_V7_106/SSD_BN_512x512_Pre_IMGNET_VOC0712_freeze_True_wideAR_False_minRatio_15/JACINTO_TI2017_V7_106_SSD_BN_512x512_Pre_IMGNET_VOC0712_freeze_True_wideAR_False_minRatio_15_iter_23906.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI2017_V7_106/SSD_BN_512x512_Pre_IMGNET_VOC0712_freeze_True_wideAR_False_minRatio_15/deploy.prototxt"
#params.LabelMap="./data/TI2017_V7_106/labelmap_ti.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170216_JACINTO_trV7_106/"

#VGG- KITTI_368x368_MulTiles
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/KITTI_MulTiles/SSD_BN_368x368_Pre_IMGNET_VOC0712_freeze_True_wideAR_False_minRatio_15/jobs/VGG_KITTI_SSD_BN_368x368_Pre_IMGNET_VOC0712_freeze_True_wideAR_False_minRatio_15_iter_45000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/KITTI_MulTiles/SSD_BN_368x368_Pre_IMGNET_VOC0712_freeze_True_wideAR_False_minRatio_15/jobs/deploy_384x384.prototxt"
#params.LabelMap="./data/KITTI_MulTiles/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170311_KITTI_384x384_3Tiles/"

#VGG- KITTI_1248x384
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/KITTI/SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15/jobs/VGG_KITTI_SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_iter_21000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/KITTI/SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15/jobs/deploy.prototxt"
#params.LabelMap="./data/KITTI/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170311_KITTI/"

#VGG- KITTI_1248x384_Per_Car_Cyclist
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_CUSTOM_freeze_True_wideAR_False_minRatio_15/jobs/VGG_KITTI_Per_Car_Cyclist_SSD_1248x384_Pre_CUSTOM_freeze_True_wideAR_False_minRatio_15_iter_21000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_CUSTOM_freeze_True_wideAR_False_minRatio_15/jobs/deploy.prototxt"
#params.LabelMap="./data/KITTI_Per_Car_Cyclist/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170313_KITTI_Per_Car_Cyclist_Test_Images/"

#VGG- KITTI_1248x384_Per_Car_Cyclist_diffNoGT
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_CUSTOM_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed/jobs/VGG_KITTI_Per_Car_Cyclist_SSD_1248x384_Pre_CUSTOM_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed_iter_21000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_CUSTOM_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed/jobs/deploy.prototxt"
#params.LabelMap="./data/KITTI_Per_Car_Cyclist/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170314_IpTest2_KITTI_Per_Car_Cyclist_diffNoGT_AllTh0_01/"

#Jacintonet - KITTI_1248x384_Per_Car_Cyclist_diffNoGT
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed/jobs/JACINTO_KITTI_Per_Car_Cyclist_SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed_iter_41000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed/jobs/deploy.prototxt"
#params.LabelMap="./data/KITTI_Per_Car_Cyclist/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170323_IpTrnP2_Jacinto_KITTI_Per_Car_Cyclist_diffNoGT_AllTh0_01_again/"
#params.MEAN_PIX_VEC =[104,117,123]

#Jacintonet - KITTI_1248x384_Per_Car_Cyclist_diffNoGT
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JINET/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed/jobs/JINET_KITTI_Per_Car_Cyclist_SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed_iter_41000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JINET/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed/jobs/deploy.prototxt"
#params.LabelMap="./data/KITTI_Per_Car_Cyclist/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170322_IpTrnP2_JINet_KITTI_Per_Car_Cyclist_diffNoGT_AllTh0_01_meanFixed/"
#params.MEAN_PIX_VEC =[0,0,0]

#ENet_v13_woDilation  - KITTI_1248x384_Per_Car_Cyclist_diffNoGT
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/ENET/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed_bvlcBN_v2_detEval_77.95/jobs/ENET_KITTI_Per_Car_Cyclist_SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed_bvlcBN_v2_iter_41000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/ENET/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_diffGTNotUsed_bvlcBN_v2_detEval_77.95/jobs/deploy.prototxt"
#params.LabelMap="./data/KITTI_Per_Car_Cyclist/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170410_IpTrnP2_Enet_woDil_KITTI_Per_Car_Cyclist_AllTh0_01/"
##if BN is placed at the beginning then mean should not be explictly subtracted so set to 0
#params.MEAN_PIX_VEC =[0,0,0]

#EXNet_v13_woDilation  - KITTI_1248x384_Per_Car_Cyclist_diffNoGT
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/ENET/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_bn_bvlc_DilENet_True_exNetLateDS_True_cifar10_False_bnAtStart_True_detEval_74.65/jobs/ENET_KITTI_Per_Car_Cyclist_SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_bn_bvlc_DilENet_True_exNetLateDS_True_cifar10_False_bnAtStart_True_iter_41000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/ENET/KITTI_Per_Car_Cyclist/SSD_1248x384_Pre_IMGNET_freeze_True_wideAR_False_minRatio_15_bn_bvlc_DilENet_True_exNetLateDS_True_cifar10_False_bnAtStart_True_detEval_74.65/jobs/deploy.prototxt"
#params.LabelMap="./data/KITTI_Per_Car_Cyclist/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170410_IpTrnP2_Exnet_Dil_KITTI_Per_Car_Cyclist_AllTh0_01/"
##if BN is placed at the beginning then mean should not be explictly subtracted so set to 0
#params.MEAN_PIX_VEC =[0,0,0]
#params.TileSizeW=1248
#params.TileSizeH=384
#NMS_FLAG=0

#if False:
#  #EXNet_v13_woDilation  - TI_V007_106
#  params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/ENET/TI2017_V7_106_MulTiles/SSD_512x512_Pre_CUSTOM_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_T_exNet_T_bnStart_T_tr_SSD_norm_T/ENET_TI2017_V7_106_MulTiles_SSD_512x512_Pre_CUSTOM_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_T_exNet_T_bnStart_T_tr_SSD_norm_T_iter_24000.caffemodel"
#  params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/ENET/TI2017_V7_106_MulTiles/SSD_512x512_Pre_CUSTOM_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_T_exNet_T_bnStart_T_tr_SSD_norm_T/deploy.prototxt"
#  params.LabelMap="./data/TI2017_V7_106_MulTiles/labelmap_ti.prototxt"
#  params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170511_TI_7_106_Exnet_Dil_temp_del_later/"
#  #if BN is placed at the beginning then mean should not be explictly subtracted so set to 0
#  params.MEAN_PIX_VEC =[0,0,0]
#  params.TileSizeW=512
#  params.TileSizeH=512
#
##VGGNet :SSD_1024x512_Pre_CUSTOM_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_F_exNet_F_bnStart_T_tr_IMG_norm_T_missedMeanSubtraction
#if False:
#  params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_mulTS_1024X512_V007_008_106/SSD_1024x512_Pre_CUSTOM_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_F_exNet_F_bnStart_T_tr_IMG_norm_T_missedMeanSubtraction/VGG_TI_mulTS_1024X512_V007_008_106_SSD_1024x512_Pre_CUSTOM_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_F_exNet_F_bnStart_T_tr_IMG_norm_T_iter_40000.caffemodel"
#  params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_mulTS_1024X512_V007_008_106/SSD_1024x512_Pre_CUSTOM_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_F_exNet_F_bnStart_T_tr_IMG_norm_T_missedMeanSubtraction/deploy_mod.prototxt"
#  params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_mulTS_1024X512_V007_008_106/labelmap.prototxt"
#  params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170602_TI_7_106_VGG_1024x512_NumScl1/"
#  ##if BN is placed at the beginning then mean should not be explictly subtracted so set to 0
#  params.MEAN_PIX_VEC =[0,0,0]
#  params.TileSizeW=1024
#  params.TileSizeH=512

#VGGNet :temp_SSD_1024x512_Pre_CUSTOM_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_T_exNet_T_bnStart_F_tr_SSD_norm_T
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_mulTS_1024X512_V007_008_106/temp_SSD_1024x512_Pre_CUSTOM_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_T_exNet_T_bnStart_F_tr_SSD_norm_T/VGG_TI_mulTS_1024X512_V007_008_106_temp_SSD_1024x512_Pre_CUSTOM_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_T_exNet_T_bnStart_F_tr_SSD_norm_T_iter_24317.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_mulTS_1024X512_V007_008_106/temp_SSD_1024x512_Pre_CUSTOM_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_T_exNet_T_bnStart_F_tr_SSD_norm_T/deploy_mod.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_mulTS_1024X512_V007_008_106/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170613_TI_mulTS_1024X512_V007_008_106_VGG_MulScale4/"
##if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.TileSizeW=1024
#params.TileSizeH=512

##Jacinto :
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI_mulTS_1024X512_V007_008_106/SSD_1024x512_Pre_IMGNET_VOC0712_CITY_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_T_exNet_T_bnStart_T_tr_SSD_norm_T/JACINTO_TI_mulTS_1024X512_V007_008_106_SSD_1024x512_Pre_IMGNET_VOC0712_CITY_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_T_exNet_T_bnStart_T_tr_SSD_norm_T_iter_20000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO/TI_mulTS_1024X512_V007_008_106/SSD_1024x512_Pre_IMGNET_VOC0712_CITY_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_T_exNet_T_bnStart_T_tr_SSD_norm_T/deploy_mod.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_mulTS_1024X512_V007_008_106/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170607_Jacinto_1024x512_TI_mulTS_V007_008_106_MulScale2/"
#params.MEAN_PIX_VEC =[0,0,0]
#params.TileSizeW=1024
#params.TileSizeH=512

##VGG TI2017_V7_106
#if False:
#  params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI2017_V7_106_MulTiles/temp_SSD_512x512_Pre_CUSTOM_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_T_exNet_T_bnStart_F_tr_SSD_norm_T/VGG_TI2017_V7_106_MulTiles_temp_SSD_512x512_Pre_CUSTOM_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_T_exNet_T_bnStart_F_tr_SSD_norm_T_iter_4000.caffemodel"
#  params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI2017_V7_106_MulTiles/temp_SSD_512x512_Pre_CUSTOM_frz_T_wideAR_F_minRatio_15_bn_b_DilENet_T_exNet_T_bnStart_F_tr_SSD_norm_T/deploy.prototxt"
#  params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI2017_V7_106_MulTiles/labelmap_ti.prototxt"
#  params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170602_TI2017_7_106_vgg_del_later/"
#  params.MEAN_PIX_VEC =[104,117,123]
#  params.TileSizeW=512
#  params.TileSizeH=512

#VGGNet :SSD_1024x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_detEval_55_16
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_mulTS_1024X512_V007_008_106/SSD_1024x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_detEval_55_16/VGG_TI_mulTS_1024X512_V007_008_106_SSD_1024x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_detEval_55_16_iter_24000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_mulTS_1024X512_V007_008_106/SSD_1024x512_Pre_IMG_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_detEval_55_16/deploy_mod.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_mulTS_1024X512_V007_008_106/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170620_PSCL_CITY_TI_mulTS_1024X512_V007_008_106_VGG_MS2/"
##if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.TileSizeW=1024
#params.TileSizeH=512

#Coco
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGGNet/pre-trained-models/coco/SSD_512x512/VGG_coco_SSD_512x512_iter_360000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGGNet/pre-trained-models/coco/SSD_512x512/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/coco/labelmap_coco.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170620_VGG_512_Coco_1Scl/"
#params.MEAN_PIX_VEC =[104,117,123]
#params.TileSizeW=512
#params.TileSizeH=512

#VGG_IMG_VOC_CIT_TIMulTS_1024x51 : SSD-2017, detEval 66.5%
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_mulTS_1024X512_V007_008_106/SSD_1024x512_Pre_IMG_VOC_CIT_TI_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_nsq_T/VGG_TI_mulTS_1024X512_V007_008_106_SSD_1024x512_Pre_IMG_VOC_CIT_TI_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_nsq_T_iter_11000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_mulTS_1024X512_V007_008_106/SSD_1024x512_Pre_IMG_VOC_CIT_TI_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_nsq_T/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_mulTS_1024X512_V007_008_106/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170704_VGG_1024x512_IMG_VOC_CIT_TIMulTS/"
##if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.TileSizeW=1024
#params.TileSizeH=512

#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/cityscape/SSD_512x512_Pre_CUS_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_detEval_40.96/VGG_cityscape_temp_SSD_512x512_Pre_CUS_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_iter_19000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/cityscape/SSD_512x512_Pre_CUS_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_detEval_40.96/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/cityscape/labelmap_cityscape.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170704_VGG_512_IMG_VOC_CIT_SS/"
##if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.TileSizeW=512
#params.TileSizeH=512

#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/cityscape/SSD_PASCAL_CITYSCAPE_500x500_detEval_31.39/VGG_VOC0712_CITYSCAPE_SSD_CITYSCAPE_500x500_detEval_31.39_iter_20000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/cityscape/SSD_PASCAL_CITYSCAPE_500x500_detEval_31.39/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/cityscape/labelmap_cityscape.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170706_del_VGG_PSCL_CITY_500_SSD2016_SS/"
##if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.TileSizeW=500
#params.TileSizeH=500

#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_512/SSD_512x512_Pre_IMG_COC_VOC07PlPl12_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_39.43/SSD_512x512_Pre_IMG_COC_VOC07PlPl12_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_39.43_iter_19000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_512/SSD_512x512_Pre_IMG_COC_VOC07PlPl12_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_39.43/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/cityscape/labelmap_cityscape.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170711_VGG_512_IMG_VOC_CIT_39.43_SS/"
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.TileSizeW=512
#params.TileSizeH=512

#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_512/SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_lr0.001_39.03/VGG_CITY_512_SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_lr0.001_39.03_iter_26000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_512/SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_lr0.001_39.03/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/cityscape/labelmap_cityscape.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170803_VGG_512_IMG_VOC_CIT512_39.03_SS_P3_TS4/"
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.TileSizeW=512
#params.TileSizeH=512
#params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)

#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/ENET/CITY_512/SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_T_tr_SSD_norm_F_DilENet_T_exNet_F_dilBF_1_poly/ENET_CITY_512_SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_T_tr_SSD_norm_F_DilENet_T_exNet_F_dilBF_1_poly_iter_31000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/ENET/CITY_512/SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_T_tr_SSD_norm_F_DilENet_T_exNet_F_dilBF_1_poly/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/cityscape/labelmap_cityscape.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170718_ENET_512_IMG_VOC_CIT512_32.27_SS/"
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.TileSizeW=512
#params.TileSizeH=512
#params.CONF_TH = {'person':0.2,'car':0.6,'trafficsign':0.6,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)


#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO_V2/CITY_512/SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_24.67/JACINTO_V2_CITY_512_SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_24.67_iter_36000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO_V2/CITY_512/SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_24.67/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/cityscape/labelmap_cityscape.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170724_JacV2_512_IMG_VOC_CIT_24.67_SS/"
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[128,128,128]
#params.TileSizeW=512
#params.TileSizeH=512
#params.CONF_TH = {'person':0.2,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)

#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/mobile/CITY_512/SSD_300x300_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_F_regHdDwn_close2Chuanqi/V1/mobile_CITY_512_SSD_300x300_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_F_regHdDwn_close2Chuanqi_iter_40000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/mobile/CITY_512/SSD_300x300_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_F_regHdDwn_close2Chuanqi/V1/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/cityscape/labelmap_cityscape.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170804_VGG_300_IMG_VOC_CIT512_21_MS4_P3_TS4/"
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[103.94,116.78,123.68]
#params.IP_SCALE = 0.017
#params.TileSizeW=300
#params.TileSizeH=300
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = 0.6

#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/VOC0712/SSD_512x512_Pre_IMG_frz_F_bn_b_bnStart_F_tr_IMG_norm_T/VGG_VOC0712_SSD_512x512_Pre_IMG_frz_F_bn_b_bnStart_F_tr_IMG_norm_T_iter_120000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/VOC0712/SSD_512x512_Pre_IMG_frz_F_bn_b_bnStart_F_tr_IMG_norm_T/deploy.prototxt"
#params.LabelMap="./models/VGGNet/VOC0712_preTrained/SSD_500x500/labelmap_voc.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170807_VGG_300_IMG_VOC_EVAL/"
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
##params.IP_SCALE = 0.017
#params.TileSizeW=512
#params.TileSizeH=512
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = 0.01

#VGG_IMG_VOC_CITY_512
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_512/SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_lr0.001_39.03/VGG_CITY_512_SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_lr0.001_39.03_iter_26000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_512/SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_lr0.001_39.03/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/cityscape/labelmap_cityscape.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170810_VGG_512_IMG_VOC_CITY512_EVAL/"
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
##params.IP_SCALE = 0.017
#params.TileSizeW=512
#params.TileSizeH=512
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = 0.01

#CITY_1024x512: 57.72%(50.51%)
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/cityscape/SSD_1024x512_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_regHd19_50.72/VGG_cityscape_SSD_1024x512_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_regHd19_50.72_iter_14000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/cityscape/SSD_1024x512_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_regHd19_50.72/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/cityscape/labelmap_cityscape.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170814_VGG_1024x512_IMG_VOC_CITY_57.52_EVAL_TEMP/"
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
##params.IP_SCALE = 0.017
#params.TileSizeW=1024
#params.TileSizeH=512
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = 0.01

#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO_V2/CITY_512/SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_26.61/JACINTO_V2_CITY_512_SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_26.61_iter_35000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO_V2/CITY_512/SSD_512x512_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_26.61/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/cityscape/labelmap_cityscape.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170816_JacV2_512_IMG_VOC_CIT_26.61_SS_EVAL_TEMP/"
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[128,128,128]
#params.TileSizeW=512 
#params.TileSizeH=512 
##params.CONF_TH = {'person':0.2,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.01
##when writing bbox map category from src->dst
#WRITE_BBOX_MAP=[['car','vehicle'], ['train','vehicle'],['bus','vehicle']]


#MobileNet_720x368_40.54%
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/mobile/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_FregHd19_40.54/mobile_CITY_720x368_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_FregHd19_40.54_iter_36000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/mobile/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_FregHd19_40.54/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/CITY_720x368/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170823_mobile_720x368_IMG_VOC_CIT_40.54_SS_crop_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[103.94,116.78,123.68]
#params.IP_SCALE = 0.017
#params.TileSizeW=720 
#params.TileSizeH=368 
#params.CONF_TH = {'person':0.3,'car':0.5,'trafficsign':0.4,'bicycle':0.2,'train':0.5,'bus':0.5,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH=0.01
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)


#VGG_720x368_50.59%
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_FregHd19_50.59/VGG_CITY_720x368_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_FregHd19_50.59_iter_26000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_FregHd19_50.59/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/CITY_720x368/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171205_VGG_720x368_IMG_VOC_CIT_50.59_MS_crop_ofst40_H480_city_video/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#VGG_720x368 TI_201708_720x368 59.18%
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F/VGG_TI_201708_720x368_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_iter_5000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201708_720x368/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170824_VGG_720x368_IMG_VOC_CIT_TI_59.18_SS/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH=0.01
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#VGG_720x368 TI_201708_720x368 59.18%
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368/SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F/VGG_TI_201708_720x368_SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_iter_10600.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201708_720x368/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170906_VGG_720x368_IMG_VOC_CIT_TI_CAT_67.17_SS_genVBB/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH=0.01
#params.EVAL_OBJ = True
#set_common_params(params)
#execute_test(params)

#VGG_720x368 TI_201708_720x368_V2 64.48%
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368_V2/SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.48/VGG_TI_201708_720x368_V2_SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.48_iter_7000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368_V2/SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.48/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201708_720x368_V2/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170911_VGG_720x368_IMG_VOC_CIT_TI_V2_64.48_SS/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH=0.01
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#VGG_720x368 TI_201708_720x368_V2 64.48%
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368_V3/SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.61/VGG_TI_201708_720x368_V3_SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.61_iter_5600.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368_V3/SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.61/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201708_720x368_V3/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170912_VGG_720x368_IMG_VOC_CIT_TI_V3_64.61_SS_fisheye/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH=0.01
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#VGG_720x368 VOC_TI_201708_720x368_V4 64.23%
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368_V4/SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.23/VGG_TI_201708_720x368_V4_SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.23_iter_7000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368_V4/SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.23/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201708_720x368_V4/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170915_VGG_720x368_IMG_VOC_TI_V4_64.23_SS_eval/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH=0.01
#params.EVAL_OBJ = True
#set_common_params(params)
#execute_test(params)

#VGG_720x368 COC_VOC_CITY_AND_TI_201708_720x368_V4 62.85%
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368_CITY/SSD_720x368_Pre_IMG_COC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_62.85/VGG_TI_201708_720x368_CITY_SSD_720x368_Pre_IMG_COC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_62.85_iter_59000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368_CITY/SSD_720x368_Pre_IMG_COC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_62.85/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201708_720x368_V4/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171206_VGG_720x368_IMG_COC_VOC_CITY_AND_TI_62.85_SS_ofst40_H480_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH=0.01
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.36 64.36%
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368_CITY/SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.36/VGG_TI_201708_720x368_CITY_SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_iter_42000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368_CITY/SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.36/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201708_720x368_CITY/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171206_VGG_720x368_IMG_COC_VOC_CITY_AND_TI_64.36_SS_ofst40_H480_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)


#Manu_512x512
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/models/ssd/cityscapes-ssd512x512_jdetnet21v2_2017-09-20_22-43-44_(25.63%)_[corrected_due_to_ignored_28.83%]/initial/cityscapes-ssd512x512_jdetnet21v2_iter_80000.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/models/ssd/cityscapes-ssd512x512_jdetnet21v2_2017-09-20_22-43-44_(25.63%)_[corrected_due_to_ignored_28.83%]/initial/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/CITY_512/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171130_VGG_512x512_IMG_COC_VOC_CITY_AND_TI_V4_62.85_SS/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=512
#params.TileSizeH=512
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#Manu_768x384
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/models/ssd/cityscapes-ssd768x384_jdetnet21v2_2017-09-21_21-24-36_(32.40%)/initial/cityscapes-ssd768x384_jdetnet21v2_iter_60000.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/models/ssd/cityscapes-ssd768x384_jdetnet21v2_2017-09-21_21-24-36_(32.40%)/initial/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/models/ssd/CITY_object_detect_SSD/lmdb/lmdb_768x384/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171205_JacV2_768x384_PSP_IMG_VOC_CITY_32.4_SS_city_video/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=768
#params.TileSizeH=384
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#Manu_768x384
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/models/ssd/cityscapes-ssd768x384_jdetnet21v2_2017-09-21_21-24-36_(32.40%)/initial/cityscapes-ssd768x384_jdetnet21v2_iter_60000.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/models/ssd/cityscapes-ssd768x384_jdetnet21v2_2017-09-21_21-24-36_(32.40%)/initial/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/models/ssd/CITY_object_detect_SSD/lmdb/lmdb_768x384/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171205_JacV2_768x384_PSP_IMG_VOC_CITY_32.4_SS_crop_ofst40_H480_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=768
#params.TileSizeH=384
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#Manu_768x384
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/models/ssd/cityscapes-ssd768x384_jdetnet21v2_2017-09-21_21-24-36_(32.40%)/initial/cityscapes-ssd768x384_jdetnet21v2_iter_60000.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/models/ssd/cityscapes-ssd768x384_jdetnet21v2_2017-09-21_21-24-36_(32.40%)/initial/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/models/ssd/CITY_object_detect_SSD/lmdb/lmdb_768x384/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171209_JacV2_768x384_PSP_IMG_VOC_CITY_32.4_ti_demo_v008/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=768
#params.TileSizeH=384
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)


##VGG_720x368 VOC_TI_201708_720x368_V4 56.26%
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368_V4/SSD_720x368_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F/V1/VGG_TI_201708_720x368_V4_SSD_720x368_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_iter_20000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201708_720x368_V4/SSD_720x368_Pre_IMG_VOC_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F/V1/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201708_720x368_V4/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170914_VGG_720x368_IMG_VOC_TI_V4_56.26_SS/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH=0.01
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#VGG_720x368 CITYSCAPES_TI_CAT 53.14%
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_720x368_TI_CATEGORY/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_53.14/VGG_CITY_720x368_TI_CATEGORY_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_53.14_iter_9900.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_720x368_TI_CATEGORY/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_53.14/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/CITY_720x368_TI_CATEGORY/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170906_VGG_720x368_IMG_VOC_CIT_TI_CAT_53.14_SS_genVBB/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH=0.01
#params.EVAL_OBJ = True
#set_common_params(params)
#execute_test(params)


#Jacinto_V2_720x368, 32.73%
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO_V2/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_32.73/JACINTO_V2_CITY_720x368_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_32.73_iter_36000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO_V2/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_32.73/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/CITY_720x368/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170823_JacV2_720x368_IMG_VOC_CIT_32.73_SS_crop_ti_demo/"
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[128,128,128]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368 
#params.CONF_TH = {'person':0.3,'car':0.5,'trafficsign':0.4,'bicycle':0.2,'train':0.5,'bus':0.5,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.EVAL_OBJ = False
##when writing bbox map category from src->dst
##WRITE_BBOX_MAP=[['car','vehicle'], ['train','vehicle'],['bus','vehicle']]
#set_common_params(params)
#execute_test(params)


#ENET_720x368, 38.79%
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/ENET/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_T_tr_SSD_norm_F_fc_F_DilENet_T_exNet_F_dilBF_1_38.79/ENET_CITY_720x368_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_T_tr_SSD_norm_F_fc_F_DilENet_T_exNet_F_dilBF_1_38.79_iter_36000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/ENET/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_T_tr_SSD_norm_F_fc_F_DilENet_T_exNet_F_dilBF_1_38.79/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/CITY_720x368/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20170823_ENET_720x368_IMG_VOC_CITY_38.79_SS_crop_ti_demo/"
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720
#params.TileSizeH=368
#params.CONF_TH = {'person':0.3,'car':0.5,'trafficsign':0.4,'bicycle':0.2,'train':0.5,'bus':0.5,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.EVAL_OBJ = False
#
###when writing bbox map category from src->dst
#WRITE_BBOX_MAP=[['car','vehicle'], ['train','vehicle'],['bus','vehicle']]
#set_common_params(params)
#execute_test(params)


#Jacinto_V2_720x368, 32.73%
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO_V2/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_32.73/JACINTO_V2_CITY_720x368_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_32.73_iter_36000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO_V2/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_32.73/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/CITY_720x368/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171208_JacV2_720x368_IMG_VOC_CIT_32.73_ti_demo_V008/"
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[128,128,128]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368 
##params.CONF_TH = {'person':0.3,'car':0.5,'trafficsign':0.4,'bicycle':0.2,'train':0.5,'bus':0.5,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = 0.4 
#params.EVAL_OBJ = False
##when writing bbox map category from src->dst
##WRITE_BBOX_MAP=[['car','vehicle'], ['train','vehicle'],['bus','vehicle']]
#set_common_params(params)
#execute_test(params)


#CITY_TI_CAT, 53.14 (city_test), 52.8 (V008)
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_720x368_TI_CATEGORY/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_53.14/VGG_CITY_720x368_TI_CATEGORY_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_53.14_iter_9900.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_720x368_TI_CATEGORY/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_53.14/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/CITY_720x368_TI_CATEGORY/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171208_VGG_720x368_IMG_VOC_CITY_TI_CAT/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#TI_CITY_V1, 53.14 (city_test), 59.81 (V008)
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201712_CITY_720x368_V1/SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_shfl_True/VGG_TI_201712_CITY_720x368_V1_SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_shfl_True_iter_28000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201712_CITY_720x368_V1/SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_shfl_True/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/CITY_720x368_TI_CATEGORY/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171208_VGG_720x368_IMG_VOC_CITY_TI_V1/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET:XYZ:
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO_V2/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_32.73/JACINTO_V2_CITY_720x368_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_32.73_iter_36000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO_V2/CITY_720x368/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_32.73/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/CITY_720x368/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171208_JacV2_720x368_IMG_VOC_CIT_32.73_ti_demo_V008/"
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[128,128,128]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368 
##params.CONF_TH = {'person':0.3,'car':0.5,'trafficsign':0.4,'bicycle':0.2,'train':0.5,'bus':0.5,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = 0.4 
#params.EVAL_OBJ = False
##when writing bbox map category from src->dst
##WRITE_BBOX_MAP=[['car','vehicle'], ['train','vehicle'],['bus','vehicle']]
#set_common_params(params)
#execute_test(params)


#CITY_TI_CAT, 53.14 (city_test), 52.8 (V008)
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_720x368_TI_CATEGORY/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_53.14/VGG_CITY_720x368_TI_CATEGORY_SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_53.14_iter_9900.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/CITY_720x368_TI_CATEGORY/SSD_720x368_Pre_IMG_VOC_CIT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_53.14/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/CITY_720x368_TI_CATEGORY/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171208_VGG_720x368_IMG_VOC_CITY_TI_CAT/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#TI_CITY_V1, 53.14 (city_test), 59.81 (V008)
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201712_CITY_720x368_V1/SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_shfl_True/VGG_TI_201712_CITY_720x368_V1_SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_shfl_True_iter_28000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201712_CITY_720x368_V1/SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_shfl_True/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/CITY_720x368_TI_CATEGORY/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171208_VGG_720x368_IMG_VOC_CITY_TI_V1/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#TI_201712_V1, 64.04% 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201712_720x368_V1/SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.04/VGG_TI_201712_720x368_V1_SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.04_iter_14000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201712_720x368_V1/SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.04/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171211_VGG_720x368_IMG_VOC_CITY_TI201712_V1/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET100: VGG_TI_201712_V1, 64.04% 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201712_720x368_V1/SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.04/VGG_TI_201712_720x368_V1_SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.04_iter_14000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201712_720x368_V1/SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.04/deploy_1024x368.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180212_VGG_1024x368_IMG_VOC_CITY_TI201712_V1_64.04_ofstneg80_H720_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=1024
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET101: JACV2_TI_201712_V1, 47.04% 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO_V2/TI_201712_720x368_V1/SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_shfl_T_47.04/JACINTO_V2_TI_201712_720x368_V1_SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_shfl_T_iter_14000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/JACINTO_V2/TI_201712_720x368_V1/SSD_720x368_Pre_IMG_VOC_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_F_fc_T_fc_T_shfl_T_47.04/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171212_JacV2_720x368_IMG_VOC_CITY_TI201712_V1_47.04_ofst40_H720_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[128,128,128]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)
#

#SET102:Manu_PSP_720x368_TI201712_V1_30%
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_jdetnet21v2_2017-12-11_13-25-56/initial/ti201712-ssd720x368_jdetnet21v2_iter_60000.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_jdetnet21v2_2017-12-11_13-25-56/initial/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171212_JacV2_720x368_PSP_IMG_VOC_CITY_TI201712_30.0_crop_ofst40_H720_ti_demo_v008/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET103: VGG_TI_201712_V1_512x256, 51.27% 
#params = Params()
#params.ModelWeights=
#params.Deploy=
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20171211_VGG_512x256_IMG_VOC_CITY_TI201712_V1_51.27_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET104: JDETNET_TI_201712_V1_720x368, 51.45% 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-11_10-13-46/initial/ti201712-ssd720x368_ssdJacintoNetV2_iter_26000_51.45.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-11_10-13-46/initial/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180117_JDetNet_720x368_IMG_VOC_CITY_TI201712_V1_51.45_ofst40_H720_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET105: JDETNET_TI_201712_V1_720x368, 49.3% @spars_fac_0.78
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-11_10-13-46/sparse_spar_0.75_0.8_itr1000_pre_51.46_accu49.3_fac0.78/ti201712-ssd720x368_ssdJacintoNetV2_iter_4000_accu49.3_fac0.78.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-11_10-13-46/initial/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180118_JDetNet_720x368_IMG_VOC_CITY_TI201712_V1_49.3_sparsed_ofst40_H720_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET106: JDETNET_TI_201712_V1_720x368, 50.49_sparsity_fac_0.8% 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-11_10-13-46/sparse/ti201712-ssd720x368_ssdJacintoNetV2_iter_30000_acc50.49_fac0.8.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-11_10-13-46/initial/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180119_JDetNet_720x368_IMG_VOC_CITY_TI201712_V1_50.49_fac0.8_ofst40_H720_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET107: JDETNET_TI_201712_720x368, accu_49.39_sparsity_fac_0.8% 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-720x368/JDetNet/ssd720x368_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1_important_50.01/sparse_fac0.5_0.8_itr5K_singleGPU_lr1E-3_0_itr24k_lr1E-2_24k_itr60k_fac0.8_accu49.39/ti201712-ssd720x368_ssdJacintoNetV2_iter_42000_49.39.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-720x368/JDetNet/ssd720x368_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1_important_50.01/sparse_fac0.5_0.8_itr5K_singleGPU_lr1E-3_0_itr24k_lr1E-2_24k_itr60k_fac0.8_accu49.39/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180209_JDetNet_720x368_VOC_CITY_TI201712_1Gmac_49.39_fac0.8_ofst40_H720_single_crop_prop_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET108: JDETNET_TI_201712_720x368, accu_50.32
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-17_22-11-56_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/initial/ti201712-ssd720x368_ssdJacintoNetV2_iter_44000_50.32.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-17_22-11-56_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/initial/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180124_JDetNet_720x368_VOC_CITY_TI201712_2.2Gmac_50.32_ofst40_H720_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET109: JDETNET_TI_201712_720x368, accu_48.86_sparsity_fac_0.8% 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-17_22-11-56_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/sparse_fac0.5_0.8_itr5K_singleGPU_lr1E-3_0_itr24k_lr1E-2_24k_itr60k_fac0.8_accu49.39/ti201712-ssd720x368_ssdJacintoNetV2_iter_57000_48.86.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-17_22-11-56_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/sparse_fac0.5_0.8_itr5K_singleGPU_lr1E-3_0_itr24k_lr1E-2_24k_itr60k_fac0.8_accu49.39/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180124_JDetNet_720x368_VOC_CITY_TI201712_1Gmac_48.86_sparse0.8_ofst40_H720_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET110: JDETNET_TI_201712_720x368, accu_49.39_sparsity_fac_0.8% 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-17_22-11-56_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/sparse_fac0.5_0.8_itr5K_singleGPU_lr1E-3_0_itr24k_lr1E-2_24k_itr60k_fac0.8_accu49.39/ti201712-ssd720x368_ssdJacintoNetV2_iter_42000_49.39.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-17_22-11-56_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/sparse_fac0.5_0.8_itr5K_singleGPU_lr1E-3_0_itr24k_lr1E-2_24k_itr60k_fac0.8_accu49.39/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180125_JDetNet_720x368_VOC_CITY_TI201712_1Gmac_49.39_sparse0.8_ofst40_H720_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET111: JDETNET_TI_201712_720x368, accu_50.15_sparsity_fac_0.8% 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-720x368/JDetNet/ssd720x368_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1_important_50.01/sparse/V4_gradually_sparsifying_50.15/ti201712-ssd720x368_ssdJacintoNetV2_iter_27700_accu50.15_fac0.8_v4.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-720x368/JDetNet/ssd720x368_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1_important_50.01/sparse/V4_gradually_sparsifying_50.15/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180209_JDetNet_720x368_VOC_CITY_TI201712_1Gmac_50.15_sparse0.8_ofst40_H720_single_crop_prop_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET112: JDETNET_TI_201712_720x368, accu_49.91_sparsity_fac_0.8% 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-17_22-11-56_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/sparse_fac0.5_0.8_itr2K_singleGPU_lR1E-2_40k_1E-3_50k_1E-4_60k_fac0.8_accu49.91/ti201712-ssd720x368_ssdJacintoNetV2_iter_46000_49.91.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-17_22-11-56_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/sparse_fac0.5_0.8_itr2K_singleGPU_lR1E-2_40k_1E-3_50k_1E-4_60k_fac0.8_accu49.91/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180127_JDetNet_720x368_VOC_CITY_TI201712_1Gmac_49.91_sparse0.8_no_nms_ofst40_H720_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET113: JDETNET_TI_201712_1024x256, accu_52.1% 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-1024x256_ssdJacintoNetV2_2018-01-21_00-21-50_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/initial/ti201712-1024x256_ssdJacintoNetV2_iter_20000_52.1.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-1024x256_ssdJacintoNetV2_2018-01-21_00-21-50_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/initial/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180125_JDetNet_1024x256_VOC_CITY_TI201712_1Gmac_52.1_ofst40_H720_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=1024 
#params.TileSizeH=256
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET114: JDETNET_TI_201712_720x368, accu_49.91_sparsity_fac_0.8% 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-17_22-11-56_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/sparse_fac0.5_0.8_itr2K_singleGPU_lR1E-2_40k_1E-3_50k_1E-4_60k_fac0.8_accu49.91/ti201712-ssd720x368_ssdJacintoNetV2_iter_46000_49.91.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-17_22-11-56_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/sparse_fac0.5_0.8_itr2K_singleGPU_lR1E-2_40k_1E-3_50k_1E-4_60k_fac0.8_accu49.91/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180125_JDetNet_720x368_VOC_CITY_TI201712_1Gmac_49.91_sparse0.8_ofst40_H720_ms2_diffTh_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = {'person':0.5,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

##SET115: JDETNET_TI_201712_720x368, accu_49.91_sparsity_fac_0.8% 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-720x368/JDetNet/ssd720x368_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1_important_50.01/sparse_fac0.5_0.8_itr2K_singleGPU_lR1E-2_40k_1E-3_50k_1E-4_60k_fac0.8_accu49.91/ti201712-ssd720x368_ssdJacintoNetV2_iter_46000_49.91.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-720x368/JDetNet/ssd720x368_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1_important_50.01/sparse_fac0.5_0.8_itr2K_singleGPU_lR1E-2_40k_1E-3_50k_1E-4_60k_fac0.8_accu49.91/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180209_JDetNet_720x368_VOC_CITY_TI201712_1Gmac_49.91_sparse0.8_ofst40_H720_single_crop_prop_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET116: JDETNET_TI201712CombCity_720x368, accu_49.02
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-city-720x368_ssdJacintoNetV2_20180125_1757_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/initial/V1/ti201712-city-720x368_ssdJacintoNetV2_iter_34000_49.02.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-city-720x368_ssdJacintoNetV2_20180125_1757_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/initial/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180126_JDetNet_720x368_VOC_CITY_TI201712CombCity_1Gmac_49.02_ofst40_H720_ti_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.5,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET117: VGG_TI_201712, 64.04% gen VBB 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201712_720x368_V1/SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.04/VGG_TI_201712_720x368_V1_SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.04_iter_14000.caffemodel"
#params.Deploy="/user/a0875091/files/work/github/weiliu89/caffe-ssd/models/VGG/TI_201712_720x368_V1/SSD_720x368_Pre_CIT_TI_CAT_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_F_64.04/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180126_VGG_720x368_IMG_VOC_CITY_TI201712_V1_64.04_test_genVBB/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH=0.4
#params.EVAL_OBJ = True
#params.decFreq = 30
#set_common_params(params)
#execute_test(params)

#SET118: VGG_TI_201712_1024x512, 68.36% gen VBB 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/models/VGG/TI_201712_1024x512/SSD_1024x512_Pre_CUS_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_T_68.36/VGG_TI_201712_1024x512_SSD_1024x512_Pre_CUS_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_T_68.36_iter_12000_68.36.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/models/VGG/TI_201712_1024x512/SSD_1024x512_Pre_CUS_frz_T_bn_b_bnStart_F_tr_SSD_norm_T_fc_T_68.36/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_1024x512/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180128_VGG_1024x512_IMG_VOC_CITY_TI201712_V1_68.36_test_genVBB/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[104,117,123]
#params.IP_SCALE  = 1.0
#params.TileSizeW=1024
#params.TileSizeH=512
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = {'person':0.3,'vehicle':0.6,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH=0.4
#params.EVAL_OBJ = True
#params.decFreq = 30
#set_common_params(params)
#execute_test(params)


#SET119: JDETNET_GT_VGG_720x368, accu_52.51
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti-vgg-720x368/JDetNet/ssd_720x368_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/initial/ti-vgg-720x368_ssdJacintoNetV2_iter_140000_52.69.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti-vgg-720x368/JDetNet/ssd_720x368_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/initial/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180203_JDetNet_720x368_vggGT_1Gmac_52.69_ofst40_H720_ti_demo_debug/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.4,'vehicle':0.5,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET120: JDETNET_720x368, accu_53.7: COCO_CITY_TI201712
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-720x368/JDetNet/20180208_22-53_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/initial/ti201712-720x368_ssdJacintoNetV2_iter_96000_53.70.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-720x368/JDetNet/20180208_22-53_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/initial/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180210_JDetNet_720x368_1Gmac_53.7_ofst40_H720_ti_prop_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.4,'vehicle':0.5,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET121: JDETNET_720x368, accu_50.32: 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-720x368/JDetNet/ssd720x368_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1_important_50.01/initial/ti201712-ssd720x368_ssdJacintoNetV2_iter_44000_50.32.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-720x368/JDetNet/ssd720x368_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1_important_50.01/initial/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180210_JDetNet_720x368_1Gmac_50.32_ofst40_H720_ti_prop_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720 
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.4,'vehicle':0.5,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET122: JDETNET_720x368, accu_55.92: 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti-vgg-720x368-v2/JDetNet/20180211_01-20_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/initial/ti-vgg-720x368-v2_ssdJacintoNetV2_iter_10000_55.92.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti-vgg-720x368-v2/JDetNet/20180211_01-20_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/initial/deploy_1024x368.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180212_JDetNet_1024x368_1Gmac_55.27_ofstneg80_H720_ti_prop_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=1024
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.4,'vehicle':0.5,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET123: JDETNET_720x368, accu_53.77, sparse_0.5: 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti-vgg-720x368-v2/JDetNet/20180211_01-20_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/sparse/ti-vgg-720x368-v2_ssdJacintoNetV2_iter_2000_fac0.5_53.77.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti-vgg-720x368-v2/JDetNet/20180211_01-20_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/sparse/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180213_JDetNet_720x368_1Gmac_53.77_ofstneg80_H720_ti_prop_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.4,'vehicle':0.5,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET124: JDETNET_720x368, accu_53.34, sparse_0.75: 
#params = Params()
#params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti-vgg-720x368-v2/JDetNet/20180211_01-20_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/sparse/ti-vgg-720x368-v2_ssdJacintoNetV2_iter_14000_fac0.75_53.34.caffemodel"
#params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti-vgg-720x368-v2/JDetNet/20180211_01-20_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/sparse/deploy.prototxt"
#params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
#params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180213_JDetNet_720x368_1Gmac_fac0.75_53.34_ofstneg80_H720_ti_prop_demo/"
##dir containing GT annotations in KITTI format
##gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
###if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
#params.MEAN_PIX_VEC =[0,0,0]
#params.IP_SCALE  = 1.0
#params.TileSizeW=720
#params.TileSizeH=368
##params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
##params.CONF_TH = {'person':0.4,'vehicle':0.5,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH=0.4
#params.EVAL_OBJ = False
#set_common_params(params)
#execute_test(params)

#SET125: JDETNET_720x368, accu_53.26, sparse_0.80: 
params = Params()
params.ModelWeights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti-vgg-720x368-v2/JDetNet/20180211_01-20_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/sparse_fac0.5_0.8_53.26/ti-vgg-720x368-v2_ssdJacintoNetV2_iter_38000_53.26.caffemodel"
params.Deploy="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti-vgg-720x368-v2/JDetNet/20180211_01-20_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/sparse_fac0.5_0.8_53.26/deploy.prototxt"
params.LabelMap="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
params.OpPath="/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/detetctedOp/20180220_JDetNet_720x368_1Gmac_fac0.8_53.26_ofstneg80_H720_ti_prop_demo/"
#dir containing GT annotations in KITTI format
#gtDIR = '/data/mmcodec_video2_tier3/users/soyeb/ObjectDetect/ssd/data/tempdata_TI201708/data/train/Videos/'
##if BN is placed at the beginning then mean should not be explictly subtracted. Set to 0 in that case.
params.MEAN_PIX_VEC =[0,0,0]
params.IP_SCALE  = 1.0
params.TileSizeW=720
params.TileSizeH=368
#params.CONF_TH = {'person':0.3,'car':0.6,'trafficsign':0.4,'bicycle':0.2,'train':0.6,'bus':0.6,'motorbike':0.2,} # pick all det obj at least with this score (def=0.6)
#params.CONF_TH = {'person':0.4,'vehicle':0.5,'trafficsign':0.4} # pick all det obj at least with this score (def=0.6)
params.CONF_TH=0.4
params.EVAL_OBJ = False
set_common_params(params)
execute_test(params)


########################################################################################################################################################################
