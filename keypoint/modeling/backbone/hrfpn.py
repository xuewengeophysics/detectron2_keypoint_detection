import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.modeling.backbone import Backbone
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

from keypoint.modeling.backbone import HRNet

##__all__是一个字符串list，用来定义模块中对于from XXX import *时要对外导出的符号，即要暴露的接口，
##但它只对from XXX import *起作用，对from XXX import XXX不起作用
__all__ = [
    "HRFPN",
    "build_hrnet_fpn_backbone"
]

class HRFPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, cfg, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum"
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(HRFPN, self).__init__()
        #assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        config = cfg.MODEL.FPN
        self.pooling_type = config.POOLING
        self.num_outs = config.NUM_OUTS  ##等于5，为什么呢？与ResNet-FPN中的P5、P4、P3、P2对应(没有P1)
        self.in_channels = config.IN_CHANNELS  ##4个分支、4个channel，IN_CHANNELS:[18, 36, 72, 144]
        self.out_channels = config.OUT_CHANNELS  ##等于256
        self.num_ins = len(self.in_channels)  ##等于4

        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(self.in_channels),
                      out_channels=self.out_channels,
                      kernel_size=1),
        )
        self.fpn_conv = nn.ModuleList()
        for i in range(self.num_outs):
            self.fpn_conv.append(nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                padding=1
            ))
        if self.pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        ##{'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32, 'p6': 64}
        self._out_feature_strides = {"p{}".format(int(s + 2)): 4 * 2**s for s in range(self.num_outs)}
        ##['p2', 'p3', 'p4', 'p5', 'p6']
        self._out_features = list(self._out_feature_strides.keys())
        ##_C.MODEL.FPN.OUT_CHANNELS = 256
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        strides = [self._out_feature_strides[f] for f in self._out_feature_strides.keys()]
        self._size_divisibility = strides[self.num_ins-1]

        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,  a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        x是图片，x.shape为torch.Size([1, 3, 800, 1216])，经过一些列卷积操作后缩小4倍，变为torch.Size([1, 256, 200, 304])
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        ##bottom_up_features是dict，bottom_up_features.keys()为dict_keys(['res2', 'res3', 'res4', 'res5'])
        ##bottom_up_features是list，代表stage4输出的各个分支的特征，为[b1, b2, b3, b4]，分辨率由高到低，逐层减半
        bottom_up_features = self.bottom_up(x)
        assert len(bottom_up_features) == self.num_ins  ##4个分支、4个channel，IN_CHANNELS:[18, 36, 72, 144]
        outs = [bottom_up_features[0]]  ##分支一的feature map的分辨率
        for i in range(1, self.num_ins):
            outs.append(F.interpolate(bottom_up_features[i], scale_factor=2**i, mode='bilinear'))  ##上采样到分支一的feature map的分辨率
        out = torch.cat(outs, dim=1)  ##在Channel维度进行融合(通道叠加)
        out = self.reduction_conv(out)  ##用1x1卷积进行通道变换(270->256)
        outs = [out]
        for i in range(1, self.num_outs):  ##self.num_outs等于5
            outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))  ##用池化层生成特征金字塔的多层，算上HRNet输出的那层，就是5层
        outputs = []
        for i in range(self.num_outs):
            outputs.append(self.fpn_conv[i](outs[i]))  ##用卷积层提取特征金字塔的各个层的特征
        ##len(outputs)等于5，元素都是tensor，是不是可以对应为P2、P3、P4、P5、P6
        ##outputs[0].shape为torch.Size([1, 256, 248, 184])
        ##outputs[1].shape为torch.Size([1, 256, 124,  92])
        ##outputs[2].shape为torch.Size([1, 256,  62,  46])
        ##outputs[3].shape为torch.Size([1, 256,  31,  23])
        ##outputs[4].shape为torch.Size([1, 256,  15,  11])

        assert len(self._out_features) == len(outputs)
        ##返回的dict.keys()为dict_keys(['p2', 'p3', 'p4', 'p5', 'p6'])
        ##fpn_out['p2'].shape为torch.Size([1, 256, 200, 304])
        ##fpn_out['p3'].shape为torch.Size([1, 256, 100, 152])
        ##fpn_out['p4'].shape为torch.Size([1, 256, 50, 76])
        ##fpn_out['p5'].shape为torch.Size([1, 256, 25, 38])
        ##fpn_out['p6'].shape为torch.Size([1, 256, 13, 19])
        return {f: res for f, res in zip(self._out_features, outputs)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class HRFPN0(nn.Module):

    def __init__(self, cfg):
        super(HRFPN, self).__init__()

        config = cfg.MODEL.NECK
        self.pooling_type = config.POOLING
        self.num_outs = config.NUM_OUTS  ##等于5，为什么呢？
        self.in_channels = config.IN_CHANNELS  ##4个分支、4个channel，IN_CHANNELS:[18, 36, 72, 144]
        self.out_channels = config.OUT_CHANNELS  ##等于256
        self.num_ins = len(self.in_channels)  ##等于4

        assert isinstance(self.in_channels, (list, tuple))

        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(self.in_channels),
                      out_channels=self.out_channels,
                      kernel_size=1),
        )
        self.fpn_conv = nn.ModuleList()
        for i in range(self.num_outs):
            self.fpn_conv.append(nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                padding=1
            ))
        if self.pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,  a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == self.num_ins  ##4个分支、4个channel，IN_CHANNELS:[18, 36, 72, 144]
        outs = [inputs[0]]  ##分支一的feature map的分辨率
        for i in range(1, self.num_ins):
            outs.append(F.interpolate(inputs[i], scale_factor=2**i, mode='bilinear'))  ##上采样到分支一的feature map的分辨率
        out = torch.cat(outs, dim=1)  ##在Channel维度进行融合(通道叠加)
        out = self.reduction_conv(out)  ##用1x1卷积进行通道变换(270->256)
        outs = [out]
        for i in range(1, self.num_outs):  ##self.num_outs等于5
            outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))  ##用池化层生成特征金字塔的多层
        outputs = []
        for i in range(self.num_outs):
            outputs.append(self.fpn_conv[i](outs[i]))  ##用卷积层提取特征金字塔的各个层的特征
        return tuple(outputs)


@BACKBONE_REGISTRY.register()
def build_hrnet_backbone(cfg, input_shape: ShapeSpec):
    
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    backbone = HRNet(cfg)
    return backbone


@BACKBONE_REGISTRY.register()
def build_hrnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = HRNet(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = HRFPN(
        cfg=cfg, 
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    # import ipdb;ipdb.set_trace()
    return backbone