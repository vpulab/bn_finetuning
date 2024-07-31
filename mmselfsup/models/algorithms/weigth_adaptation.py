# Copyright (c) OpenMMLab. All rights reserved.
import torch
import wandb

from ..builder import ALGORITHMS, build_backbone, build_head
from .base import BaseModel


@ALGORITHMS.register_module()
class WeightAdaptation(BaseModel):
    """Simple image classification.

    Args:
        backbone (dict): Config dict for module of backbone.
        with_sobel (bool): Whether to apply a Sobel filter.
            Defaults to False.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self, backbone, backbone_1, backbone_2, head=None, init_cfg=None):
        super(WeightAdaptation, self).__init__(init_cfg)

        self.backbone = backbone # placeholder

        self.backbone_1 = build_backbone(backbone_1)
        self.backbone_2 = build_backbone(backbone_2)


        self.backbone_1.load_state_dict(torch.load("/home/kis/Desktop/rhome/kis/code/mmselfsup/pretrained_models/official_weights/mmselfsup_format/swav_backbone.pth"), strict=False)
        self.backbone_2.load_state_dict(torch.load("/home/kis/Desktop/rhome/kis/code/mmselfsup/pretrained_models/official_weights/mmselfsup_format/relative_loc_backbone.pth"), strict=False)
        assert head is not None
        self.head = build_head(head)

    def arithmetic_sum(self, scaling_coeffs=[1.0, 1.0]):
        for (name_1, layer_1), (name_2, layer_2) in zip(self.backbone_1.named_modules(), self.backbone_2.named_modules()):
            if 'conv' in name_1:
                layer_1.weight.data = layer_1.weight * scaling_coeffs[0] + layer_2.weight * scaling_coeffs[1]

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone_1(img)
        return x


    def forward_train(self, img, label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        

        x = self.backbone_1(img)
        outs = self.head(x)
        loss_inputs = (outs, label)
        losses = self.head.loss(*loss_inputs)

        
        return losses

    def forward_test(self, img, **kwargs):
        """Forward computation during test.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        x = self.extract_feat(img)  # tuple
        outs = self.head(x)
        if hasattr(self.backbone, "out_indices"):
            keys = [f'head{i}' for i in self.backbone.out_indices]
        else:
            keys = ['head4']
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def forward_test_loss(self, img, label, **kwargs):
        """Forward computation during test with losses.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor, Tensor]: A dictionary of output features and losses.
        """
        x = self.extract_feat(img)  # tuple
        outs = self.head(x)
        loss_inputs = (outs, label)
        losses = self.head.loss(*loss_inputs)
        keys = [f'head{i}' for i in self.backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors, losses))
