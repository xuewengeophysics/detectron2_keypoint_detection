# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# FPN（在maskrcnn-benchmark中为NECK）
# ---------------------------------------------------------------------------- #
# _C.MODEL.NECK = CN()
_C.MODEL.FPN.IN_CHANNELS = (32, 64, 128, 256)
# _C.MODEL.NECK.OUT_CHANNELS = 256
# _C.MODEL.NECK.ACTIVATION = False
_C.MODEL.FPN.POOLING = ' AVG'
# _C.MODEL.NECK.SHARING_CONV = False
_C.MODEL.FPN.NUM_OUTS = 5

# ---------------------------------------------------------------------------- #
# HRNET options
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.HRNET = CN()

_C.MODEL.HRNET.OUT_FEATURES = ["stage1", "stage2", "stage3", "stage4"]

# MODEL.HRNET related params
_C.MODEL.HRNET.BASE_CHANNEL = [96, 96, 96, 96]
_C.MODEL.HRNET.CHANNEL_GROWTH = 2
_C.MODEL.HRNET.BLOCK_TYPE = 'BottleneckWithFixedBatchNorm'
_C.MODEL.HRNET.BRANCH_DEPTH = [3, 3, 3, 3]
_C.MODEL.HRNET.NUM_BLOCKS = [6, 4, 4, 4]
_C.MODEL.HRNET.NUM_LAYERS = [3, 3, 3]
_C.MODEL.HRNET.FINAL_CONV_KERNEL = 1


# for bi-directional fusion
# Stage 1
_C.MODEL.HRNET.STAGE1 = CN()
_C.MODEL.HRNET.STAGE1.NUM_MODULES = 1
_C.MODEL.HRNET.STAGE1.NUM_BRANCHES = 1
_C.MODEL.HRNET.STAGE1.NUM_BLOCKS = [3]
_C.MODEL.HRNET.STAGE1.NUM_CHANNELS = [64]
_C.MODEL.HRNET.STAGE1.BLOCK = 'BottleneckWithFixedBatchNorm'
_C.MODEL.HRNET.STAGE1.FUSE_METHOD = 'SUM'
# Stage 2
_C.MODEL.HRNET.STAGE2 = CN()
_C.MODEL.HRNET.STAGE2.NUM_MODULES = 1
_C.MODEL.HRNET.STAGE2.NUM_BRANCHES = 2
_C.MODEL.HRNET.STAGE2.NUM_BLOCKS = [4, 4]
_C.MODEL.HRNET.STAGE2.NUM_CHANNELS = [24, 48]
_C.MODEL.HRNET.STAGE2.BLOCK = 'BottleneckWithFixedBatchNorm'
_C.MODEL.HRNET.STAGE2.FUSE_METHOD = 'SUM'
# Stage 3
_C.MODEL.HRNET.STAGE3 = CN()
_C.MODEL.HRNET.STAGE3.NUM_MODULES = 1
_C.MODEL.HRNET.STAGE3.NUM_BRANCHES = 3
_C.MODEL.HRNET.STAGE3.NUM_BLOCKS = [4, 4, 4]
_C.MODEL.HRNET.STAGE3.NUM_CHANNELS = [24, 48, 92]
_C.MODEL.HRNET.STAGE3.BLOCK = 'BottleneckWithFixedBatchNorm'
_C.MODEL.HRNET.STAGE3.FUSE_METHOD = 'SUM'
# Stage 4
_C.MODEL.HRNET.STAGE4 = CN()
_C.MODEL.HRNET.STAGE4.NUM_MODULES = 1
_C.MODEL.HRNET.STAGE4.NUM_BRANCHES = 4
_C.MODEL.HRNET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
_C.MODEL.HRNET.STAGE4.NUM_CHANNELS = [24, 48, 92, 192]
_C.MODEL.HRNET.STAGE4.BLOCK = 'BottleneckWithFixedBatchNorm'
_C.MODEL.HRNET.STAGE4.FUSE_METHOD = 'SUM'
_C.MODEL.HRNET.STAGE4.MULTI_OUTPUT = True
# Decoder
_C.MODEL.HRNET.DECODER = CN()
_C.MODEL.HRNET.DECODER.BLOCK = 'BottleneckWithFixedBatchNorm'
_C.MODEL.HRNET.DECODER.HEAD_UPSAMPLING = 'BILINEAR'
_C.MODEL.HRNET.DECODER.HEAD_UPSAMPLING_KERNEL = 1