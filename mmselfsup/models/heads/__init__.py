# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .cls_head_weighted import ClsHeadWeighted
from .cls_head_pool import ClsHeadPool
from .contrastive_head import ContrastiveHead
from .latent_pred_head import LatentClsHead, LatentPredictHead
from .multi_cls_head import MultiClsHead
from .swav_head import SwAVHead
from .cls_head_weighted_non_linear import ClsHeadWeightedNonLinear
from .cls_head_facet import ClsHeadFacet

__all__ = [
    'ContrastiveHead', 'ClsHead', 'ClsHeadFacet', 'ClsHeadWeighted', 'LatentPredictHead', 'LatentClsHead',
    'MultiClsHead', 'SwAVHead', 'ClsHeadPool', 'ClsHeadWeightedNonLinear'
]

