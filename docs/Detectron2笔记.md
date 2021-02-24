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

+ 要注意HRNetv2P中FPN的改写，参考HRNet./detectron2/modeling/backbone/fpn.py以及HRNet-MaskRCNN-Benchmark/maskrcnn_benchmark/modeling/backbone/hrfpn.py

1. 新增默认参数配置文件./keypoint/config/defaults.py，并针对hrnet进行相应的修改；并在tools/train_net.py导入这个module，导入的时候要注意文件路径。
```
import _init_paths
from keypoint.config import get_cfg
```

```
#_init_paths.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

my_path = osp.join(this_dir, '..')
add_path(my_path)
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
没有_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN，而是_C.MODEL.RPN.PRE_NMS_TOPK_TRAIN
没有_C.MODEL.RPN.PRE_NMS_TOP_N_TEST，而是_C.MODEL.RPN.PRE_NMS_TOPK_TEST
没有_C.MODEL.RPN.POST_NMS_TOP_N_TEST，而是_C.MODEL.RPN.POST_NMS_TOPK_TEST
没有_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST
没有_C.MODEL.HRNET
没有_C.MODEL.ROI_HEADS.USE_FPN
没有_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES
没有_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
没有_C.MODEL.ROI_BOX_HEAD.PREDICTOR
没有_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES
没有_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES
没有_C.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR
没有_C.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR
没有_C.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION
没有_C.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR

修改方法是新增./keypoint/config/defaults.py，并进行相应的修改。

注意：_C.MODEL.RPN.IN_FEATURES = ["res4"]

```


2. 新增参数配置文件./configs/hrnet/e2e_keypoint_rcnn_hrnet_w18_1x.yaml，并针对hrnet进行相应的修改；并在训练模型的时候导入至--config-file。


```
配置文件：configs/
模型文件：hrnet/
```



3. 

### 先把backbone替换成hrnet

+ 参考https://github.com/sxhxliang/detectron2_backbone/blob/master/detectron2_backbone/backbone/hrnet.py

+ 参考centermask2/centermask/modeling/backbone/fpn.py，重写detectron2_keypoint_detection/keypoint/modeling/backbone/hrfpn.py，注意返回的是backbone module, must be a subclass of :class:`Backbone`.


KeyError: "No object named 'build_hrnet_fpn_backbone' found in 'BACKBONE' registry!"
```
Traceback (most recent call last):
  File "tools/train_net.py", line 169, in <module>
    args=(args,),
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/launch.py", line 62, in launch
    main_func(*args)
  File "tools/train_net.py", line 151, in main
    trainer = Trainer(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/defaults.py", line 310, in __init__
    model = self.build_model(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/defaults.py", line 452, in build_model
    model = build_model(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/meta_arch/build.py", line 21, in build_model
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 181, in wrapped
    explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 238, in _get_args_from_config
    ret = from_config_func(*args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/meta_arch/rcnn.py", line 75, in from_config
    backbone = build_backbone(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/backbone/build.py", line 31, in build_backbone
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape)
  File "/opt/Software/miniconda3/envs/det2/lib/python3.6/site-packages/fvcore/common/registry.py", line 72, in get
    "No object named '{}' found in '{}' registry!".format(name, self._name)
KeyError: "No object named 'build_hrnet_fpn_backbone' found in 'BACKBONE' registry!"

```

+ 解决办法：在`hrfpn.py`中添加`from keypoint.modeling.backbone import HRNet`


+ torch.nn.modules.module.ModuleAttributeError: 'HRNet' object has no attribute '_out_feature_channels'

```
Traceback (most recent call last):
  File "tools/train_net.py", line 169, in <module>
    args=(args,),
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/launch.py", line 62, in launch
    main_func(*args)
  File "tools/train_net.py", line 151, in main
    trainer = Trainer(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/defaults.py", line 310, in __init__
    model = self.build_model(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/defaults.py", line 452, in build_model
    model = build_model(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/meta_arch/build.py", line 21, in build_model
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 181, in wrapped
    explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 238, in _get_args_from_config
    ret = from_config_func(*args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/meta_arch/rcnn.py", line 75, in from_config
    backbone = build_backbone(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/backbone/build.py", line 31, in build_backbone
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape)
  File "/opt/SRC/projects/keypoint_detection/detectron2_keypoint_detection/tools/../keypoint/modeling/backbone/hrfpn.py", line 98, in build_hrnet_fpn_backbone
    bottom_up = build_hrnet_backbone(cfg, input_shape)
  File "/opt/SRC/projects/keypoint_detection/detectron2_keypoint_detection/tools/../keypoint/modeling/backbone/hrfpn.py", line 85, in build_hrnet_backbone
    print(backbone.output_shape())
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/backbone/backbone.py", line 52, in output_shape
    for name in self._out_features
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/backbone/backbone.py", line 52, in <dictcomp>
    for name in self._out_features
  File "/opt/Software/miniconda3/envs/det2/lib/python3.6/site-packages/torch/nn/modules/module.py", line 779, in __getattr__
    type(self).__name__, name))
torch.nn.modules.module.ModuleAttributeError: 'HRNet' object has no attribute '_out_feature_channels'

```

+ 解决办法：换成https://github.com/sxhxliang/detectron2_backbone/中的hrnet


+ assert in_features, in_features  AssertionError: []

```
Traceback (most recent call last):
  File "tools/train_net.py", line 169, in <module>
    args=(args,),
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/launch.py", line 62, in launch
    main_func(*args)
  File "tools/train_net.py", line 151, in main
    trainer = Trainer(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/defaults.py", line 310, in __init__
    model = self.build_model(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/defaults.py", line 452, in build_model
    model = build_model(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/meta_arch/build.py", line 21, in build_model
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 181, in wrapped
    explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 238, in _get_args_from_config
    ret = from_config_func(*args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/meta_arch/rcnn.py", line 75, in from_config
    backbone = build_backbone(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/backbone/build.py", line 31, in build_backbone
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape)
  File "/opt/SRC/projects/keypoint_detection/detectron2_keypoint_detection/tools/../keypoint/modeling/backbone/hrfpn.py", line 107, in build_hrnet_fpn_backbone
    fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/backbone/fpn.py", line 50, in __init__
    assert in_features, in_features
AssertionError: []
```

因为默认参数配置文件./detectron2/config/defaults.py中`_C.MODEL.FPN.IN_FEATURES = []`；

+ 解决办法：在参数配置文件./configs/hrnet/e2e_keypoint_rcnn_hrnet_w18_1x.yaml中添加MODEL.FPN.IN_FEATURES：

```
  FPN:
    IN_FEATURES: ["stage4"]
```


+ KeyError: 'res4' ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features]) 

```
Traceback (most recent call last):
  File "tools/train_net.py", line 169, in <module>
    args=(args,),
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/launch.py", line 62, in launch
    main_func(*args)
  File "tools/train_net.py", line 151, in main
    trainer = Trainer(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/defaults.py", line 310, in __init__
    model = self.build_model(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/defaults.py", line 452, in build_model
    model = build_model(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/meta_arch/build.py", line 21, in build_model
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 181, in wrapped
    explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 238, in _get_args_from_config
    ret = from_config_func(*args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/meta_arch/rcnn.py", line 78, in from_config
    "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/proposal_generator/build.py", line 24, in build_proposal_generator
    return PROPOSAL_GENERATOR_REGISTRY.get(name)(cfg, input_shape)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 181, in wrapped
    explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 238, in _get_args_from_config
    ret = from_config_func(*args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/proposal_generator/rpn.py", line 241, in from_config
    ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/proposal_generator/rpn.py", line 241, in <listcomp>
    ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
KeyError: 'res4'

```

+ 解决办法：在参数配置文件./configs/hrnet/e2e_keypoint_rcnn_hrnet_w18_1x.yaml中添加MODEL.RPN.IN_FEATURES：

```
  RPN:
    IN_FEATURES: ["p5"]
```


+ KeyError: 'res4' roi_heads.py 

```
Traceback (most recent call last):
  File "tools/train_net.py", line 169, in <module>
    args=(args,),
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/launch.py", line 62, in launch
    main_func(*args)
  File "tools/train_net.py", line 151, in main
    trainer = Trainer(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/defaults.py", line 310, in __init__
    model = self.build_model(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/defaults.py", line 452, in build_model
    model = build_model(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/meta_arch/build.py", line 21, in build_model
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 181, in wrapped
    explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 238, in _get_args_from_config
    ret = from_config_func(*args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/meta_arch/rcnn.py", line 79, in from_config
    "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/roi_heads/roi_heads.py", line 43, in build_roi_heads
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 181, in wrapped
    explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 238, in _get_args_from_config
    ret = from_config_func(*args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/roi_heads/roi_heads.py", line 391, in from_config
    pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
KeyError: 'res4'

```

+ 解决办法：在参数配置文件./configs/hrnet/e2e_keypoint_rcnn_hrnet_w18_1x.yaml中添加MODEL.RPN.IN_FEATURES：

```
  ROI_HEADS:
    IN_FEATURES: ["p5"]
```


+ AssertionError: assert not cfg.MODEL.KEYPOINT_ON

```
Traceback (most recent call last):
  File "tools/train_net.py", line 169, in <module>
    args=(args,),
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/launch.py", line 62, in launch
    main_func(*args)
  File "tools/train_net.py", line 151, in main
    trainer = Trainer(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/defaults.py", line 310, in __init__
    model = self.build_model(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/defaults.py", line 452, in build_model
    model = build_model(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/meta_arch/build.py", line 21, in build_model
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 181, in wrapped
    explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 238, in _get_args_from_config
    ret = from_config_func(*args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/meta_arch/rcnn.py", line 79, in from_config
    "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/roi_heads/roi_heads.py", line 43, in build_roi_heads
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 181, in wrapped
    explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/config/config.py", line 238, in _get_args_from_config
    ret = from_config_func(*args, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/roi_heads/roi_heads.py", line 395, in from_config
    assert not cfg.MODEL.KEYPOINT_ON
AssertionError
```

```
GeneralizedRCNN(
  (backbone): FPN(
    (fpn_lateral2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_output2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fpn_lateral3): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_output3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fpn_lateral4): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_output4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fpn_lateral5): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_output5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (top_block): LastLevelMaxPool()
    (bottom_up): ResNet(
      (stem): BasicStem(
        (conv1): Conv2d(
          3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
          (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
        )
      )
      (res2): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv1): Conv2d(
            64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv2): Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv3): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv2): Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv3): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv2): Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv3): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
        )
      )
      (res3): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv1): Conv2d(
            256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
        (3): BottleneckBlock(
          (conv1): Conv2d(
            512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
      )
      (res4): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
          (conv1): Conv2d(
            512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (3): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (4): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (5): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
      )
      (res5): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
          (conv1): Conv2d(
            1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv2): Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv3): Conv2d(
            512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv2): Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv3): Conv2d(
            512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv2): Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv3): Conv2d(
            512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
        )
      )
    )
  )
  (proposal_generator): RPN(
    (rpn_head): StandardRPNHead(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (objectness_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
      (anchor_deltas): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
    )
    (anchor_generator): DefaultAnchorGenerator(
      (cell_anchors): BufferList()
    )
  )
  (roi_heads): StandardROIHeads(
    (box_pooler): ROIPooler(
      (level_poolers): ModuleList(
        (0): ROIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=0, aligned=True)
        (1): ROIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=0, aligned=True)
        (2): ROIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=0, aligned=True)
        (3): ROIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=0, aligned=True)
      )
    )
    (box_head): FastRCNNConvFCHead(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (fc1): Linear(in_features=12544, out_features=1024, bias=True)
      (fc_relu1): ReLU()
      (fc2): Linear(in_features=1024, out_features=1024, bias=True)
      (fc_relu2): ReLU()
    )
    (box_predictor): FastRCNNOutputLayers(
      (cls_score): Linear(in_features=1024, out_features=2, bias=True)
      (bbox_pred): Linear(in_features=1024, out_features=4, bias=True)
    )
    (keypoint_pooler): ROIPooler(
      (level_poolers): ModuleList(
        (0): ROIAlign(output_size=(14, 14), spatial_scale=0.25, sampling_ratio=0, aligned=True)
        (1): ROIAlign(output_size=(14, 14), spatial_scale=0.125, sampling_ratio=0, aligned=True)
        (2): ROIAlign(output_size=(14, 14), spatial_scale=0.0625, sampling_ratio=0, aligned=True)
        (3): ROIAlign(output_size=(14, 14), spatial_scale=0.03125, sampling_ratio=0, aligned=True)
      )
    )
    (keypoint_head): KRCNNConvDeconvUpsampleHead(
      (conv_fcn1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_fcn_relu1): ReLU()
      (conv_fcn2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_fcn_relu2): ReLU()
      (conv_fcn3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_fcn_relu3): ReLU()
      (conv_fcn4): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_fcn_relu4): ReLU()
      (conv_fcn5): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_fcn_relu5): ReLU()
      (conv_fcn6): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_fcn_relu6): ReLU()
      (conv_fcn7): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_fcn_relu7): ReLU()
      (conv_fcn8): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_fcn_relu8): ReLU()
      (score_lowres): ConvTranspose2d(512, 17, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    )
  )
)
```

把model的网络结构和各个模块的输出区分开。


+ 'NoneType' object has no attribute 'fill_'

解决办法
```
if m.bias is not None:
    nn.init.constant_(m.bias, 0)
```

+ FloatingPointError: Predicted boxes or scores contain Inf/NaN. Training has diverged.
```
Traceback (most recent call last):
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/train_loop.py", line 134, in train
    self.run_step()
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/defaults.py", line 441, in run_step
    self._trainer.run_step()
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/engine/train_loop.py", line 228, in run_step
    loss_dict = self.model(data)
  File "/opt/Software/miniconda3/envs/det2/lib/python3.6/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/meta_arch/rcnn.py", line 160, in forward
    proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
  File "/opt/Software/miniconda3/envs/det2/lib/python3.6/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/proposal_generator/rpn.py", line 437, in forward
    anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/proposal_generator/rpn.py", line 470, in predict_proposals
    self.training,
  File "/opt/SRC/projects/keypoint_detection/detectron2/detectron2/modeling/proposal_generator/proposal_utils.py", line 104, in find_top_rpn_proposals
    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
FloatingPointError: Predicted boxes or scores contain Inf/NaN. Training has diverged.
```