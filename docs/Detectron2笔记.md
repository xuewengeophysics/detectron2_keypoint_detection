# Detectron2笔记

+ PyTorch
+ Modular, extensible design
+ New models and features
+ New tasks
+ Implementation quality
+ Speed and scalability
+ Detectron2go





二次开发

+ 基于detectron2完成hrnet的关键点检测。参考centermask2。



```
配置文件：configs/
模型文件：hrnet/
```

+ 注意./detectron2/config/defaults.py与./maskrcnn_benchmark/config/defaults.py的区别

```
没有_C.DATALOADER.SIZE_DIVISIBILITY
没有_C.INPUT.TO_BGR255
不是_C.INPUT.PIXEL_MEAN，而是_C.MODEL.PIXEL_MEAN
不是_C.INPUT.PIXEL_STD，而是_C.MODEL.PIXEL_STD
不是_C.MODEL.WEIGHT，而是_C.MODEL.WEIGHTS
不是_C.MODEL.BACKBONE.CONV_BODY，而是_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
没有_C.MODEL.BACKBONE.OUT_CHANNELS
没有_C.MODEL.NECK
没有_C.MODEL.RPN.USE_FPN
没有_C.MODEL.RPN.ANCHOR_STRIDE
没有_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
没有_C.MODEL.RPN.PRE_NMS_TOP_N_TEST
没有_C.MODEL.RPN.POST_NMS_TOP_N_TEST
没有_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST
没有_C.MODEL.HRNET
没有_C.MODEL.ROI_HEADS.USE_FPN
没有_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES
没有_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
没有_C.MODEL.ROI_BOX_HEAD.PREDICTOR
没有_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES
没有_C.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR
没有_C.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR
没有_C.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION
没有_C.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR

修改方法是新增./keypoint/config/defaults.py，并进行相应的修改。

```


### 先把backbone替换成hrnet

+ 参考https://github.com/sxhxliang/detectron2_backbone/blob/master/detectron2_backbone/backbone/hrnet.py

+ 参考centermask2/centermask/modeling/backbone/fpn.py，重写detectron2_keypoint_detection/keypoint/modeling/backbone/hrfpn.py，注意返回的是backbone module, must be a subclass of :class:`Backbone`.
