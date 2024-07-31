# Copyright (c) OpenMMLab. All rights reserved.
from .adapter_hook import AdapterHook
from .byol_hook import BYOLHook

# from .deepcluster_hook import DeepClusterHook

from .densecl_hook import DenseCLHook
from .odc_hook import ODCHook
from .optimizer_hook import DistOptimizerHook, GradAccumFp16OptimizerHook
from .simsiam_hook import SimSiamHook
from .swav_hook import SwAVHook

# __all__ = [
#     'AdapterHook', 'BYOLHook', 'DeepClusterHook', 'DenseCLHook', 'ODCHook',
#     'DistOptimizerHook', 'GradAccumFp16OptimizerHook', 'SimSiamHook',
#     'SwAVHook'
# ]

__all__ = [
    'AdapterHook', 'BYOLHook','DenseCLHook', 'ODCHook',
    'DistOptimizerHook', 'GradAccumFp16OptimizerHook', 'SimSiamHook',
    'SwAVHook'
]