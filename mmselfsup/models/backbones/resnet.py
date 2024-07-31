# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch
import wandb
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from ..utils import ResLayer


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@BACKBONES.register_module()
class ResNet(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Defaults to None.
        base_channels (int): Number of base channels of res layer.
            Defaults to 64.
        num_stages (int): Resnet stages. Defaults to 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Defaults to False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None

    Example:
        >>> from mmselfsup.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3, 4),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False,
                 init_cfg=None,
                 unfreeze_bn=False,
                 save_bn_features=False,
                 unfreeze_last_layer=False,
                 unfreeze_second_to_last_layer=False,
                 unfreeze_third_to_last_layer=False,
                 unfreeze_last_module = False,
                 unfreeze_second_to_last_module = False,
                 unfreeze_third_to_last_module = False):
        super(ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.save_bn_features = save_bn_features
        block_init_cfg = None
        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(
                    type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
            ]
            block = self.arch_settings[depth][0]
            if self.zero_init_residual:
                if block is BasicBlock:
                    block_init_cfg = dict(
                        type='Constant', val=0, override=dict(name='norm2'))
                elif block is Bottleneck:
                    block_init_cfg = dict(
                        type='Constant', val=0, override=dict(name='norm3'))

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.unfreeze_last_layer = unfreeze_last_layer
        self.unfreeze_second_to_last_layer = unfreeze_second_to_last_layer
        self.unfreeze_third_to_last_layer = unfreeze_third_to_last_layer
        self.unfreeze_last_module = unfreeze_last_module
        self.unfreeze_second_to_last_module = unfreeze_second_to_last_module
        self.unfreeze_third_to_last_module = unfreeze_third_to_last_module
        self.unfreeze_bn = unfreeze_bn
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages + 1
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        self.weight_scales = []
        self.weight_biases = []
        self.original_weights = []

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        # self.init_weight_adapters()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def init_weight_adapters(self):
        self.weight_scales = []
        self.weight_biases = []
        self.original_weights = []
        self.wandb_log_step = 0
        print("*** Init adapters ***")

        for name, layer in self.named_modules():

            if 'conv' in name:
                print(f"\t{name}")
                scales = torch.normal(mean=1.0, std=0.01, size=layer.weight.shape[:2]).to(device=torch.device("cuda"))
                scales.requires_grad=True
                
                self.weight_scales.append(nn.Parameter(scales))

                biases = torch.normal(mean=0.0, std=0.01, size=layer.weight.shape[:2]).to(device=torch.device("cuda"))
                biases.requires_grad=True
                
                self.weight_biases.append(nn.Parameter(biases))

                self.original_weights.append(layer.weight)


        self.weight_scales = nn.ParameterList(self.weight_scales)
        self.weight_bias = nn.ParameterList(self.weight_biases)
        # wandb.log({f"conv1 weight adapters {self.wandb_log_step}": wandb.plots.HeatMap([i for i in range(self.weight_scales[0].shape[1])], [i for i in range(self.weight_scales[0].shape[0])], self.weight_scales[0].cpu().detach().numpy(), show_text=False)}, step=self.wandb_log_step)
        self.wandb_log_step +=1
        

    def apply_adapters(self):
        k = 0
        # print("*** Apply adapters ***")
        for name, layer in self.named_modules():
            if 'conv' in name:
                # print(f"\t{name}")
                for param_key in layer._parameters:
                    p = layer._parameters[param_key]
                    if param_key == 'weight':
                        updated = p * self.weight_scales[k][:,:,None,None]# + self.weight_biases[k][:,:,None,None]
                        layer._parameters[param_key] = updated
                # layer.weight = nn.Parameter(layer.weight * self.weight_scales[k][:,:,None,None] + self.weight_biases[k][:,:,None,None])
                # dims = layer.weight.shape
                # for i in range(dims[0]):
                #     for j in range(dims[1]):
                #         layer.weight[i,j,:,:] = layer.weight[i,j,:,:] * self.weight_scales[k][i,j] + self.weight_biases[k][i,j]
                k += 1

        # if self.wandb_log_step % 500 == 0:
            # wandb.log({f"conv1 weight adapters {self.wandb_log_step}": wandb.plots.HeatMap([i for i in range(self.weight_scales[0].shape[1])], [i for i in range(self.weight_scales[0].shape[0])], self.weight_scales[0].cpu().detach().numpy(), show_text=False)}, step=self.wandb_log_step)
        self.wandb_log_step += 1
                
    def restore_original_weights(self):
        k = 0
        for name, layer in self.named_modules():
            if 'conv' in name:
                layer.weight = self.original_weights[k]
                k += 1



    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        # Layer by layer

        if self.unfreeze_last_layer:
            m = getattr(self, 'layer4')
            m = m[-1].conv3
            m.train()
            for param in m.parameters():
                param.requires_grad = True

            m = getattr(self, 'layer4')
            m = m[-1].bn3
            m.train()
            for param in m.parameters():
                param.requires_grad = True
                
        if self.unfreeze_second_to_last_layer:
            m = getattr(self, 'layer4')
            m = m[-1].conv2
            m.train()
            for param in m.parameters():
                param.requires_grad = True
            m = getattr(self, 'layer4')
            m = m[-1].bn2
            m.train()
            for param in m.parameters():
                param.requires_grad = True

        if self.unfreeze_third_to_last_layer:
            m = getattr(self, 'layer4')
            m = m[-1].conv1
            m.train()
            for param in m.parameters():
                param.requires_grad = True
            m = getattr(self, 'layer4')
            m = m[-1].bn1
            m.train()
            for param in m.parameters():
                param.requires_grad = True

        # Module by module

        if self.unfreeze_last_module:
            m = getattr(self, 'layer4')
            m = m[-1]
            m.train()
            for param in m.parameters():
                param.requires_grad = True

        if self.unfreeze_second_to_last_module:
            m = getattr(self, 'layer4')
            m = m[-2]
            m.train()
            for param in m.parameters():
                param.requires_grad = True

        if self.unfreeze_third_to_last_module:
            m = getattr(self, 'layer4')
            m = m[-3]
            m.train()
            for param in m.parameters():
                param.requires_grad = True



    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)  # r50: 64x128x128
        outs = []
        if 0 in self.out_indices:
            outs.append(x)
        x = self.maxpool(x)  # r50: 64x56x56
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i + 1 in self.out_indices:
                outs.append(x)
        # r50: 1-256x56x56; 2-512x28x28; 3-1024x14x14; 4-2048x7x7
        # print(torch.cuda.memory_summary())
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        # print(torch.cuda.memory_allocated())
        self._freeze_stages()
        # print(torch.cuda.memory_allocated())
        if mode and self.unfreeze_bn:
            self.norm1.train()
            for m in [self.norm1,]:
                for param in m.parameters():
                    param.requires_grad = True
            for i in range(1, self.frozen_stages + 1):
                m = getattr(self, f'layer{i}')
                for name, layer in m.named_modules():
                    if 'bn' in name and 'bn1' not in name:
                        # print(f"Disable eval mode for layer {name}")
                        layer.train()
                        for param in layer.parameters():
                            param.requires_grad = True
        # print(torch.cuda.memory_allocated())

        # if mode and self.norm_eval:
        #     for m in self.modules():
        #         # trick: eval have effect on BatchNorm only
        #         if isinstance(m, _BatchNorm):
        #             m.eval()


@BACKBONES.register_module()
class ResNetV1d(ResNet):
    r"""ResNetV1d variant described in `Bag of Tricks
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)
