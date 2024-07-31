# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import wandb

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from ..utils import DistillKL
from .base import BaseModel


@ALGORITHMS.register_module()
class SwAVKD(BaseModel):
    """SwAV.

    Implementation of `Unsupervised Learning of Visual Features by Contrasting
    Cluster Assignments <https://arxiv.org/abs/2006.09882>`_.
    The queue is built in `core/hooks/swav_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 backbone_teacher,
                 teacher_checkpoint,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 kd_T=4,
                 beta=0.1,
                 delta=1,
                 **kwargs):
        super(SwAVKD, self).__init__(init_cfg)

        
        self.backbone = build_backbone(backbone)

        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)
        
        # Build the teacher model
        self.teacher_checkpoint = teacher_checkpoint
        self.backbone_teacher = build_backbone(backbone_teacher)
        self.teacher_initialized = False
        # Save the KD parameters 
        self.beta = beta
        self.delta = delta
        # Define the KL criterion
        self.criterion_div = DistillKL(kd_T)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_teacher(self):
        print("\n\n\n\n")
        print(f"Load teacher model checkpoint from {self.teacher_checkpoint}")
        print("\n\n\n\n")
        self.backbone_teacher.load_state_dict(torch.load(self.teacher_checkpoint)['state_dict'], strict=False)
        # print(self.backbone_teacher.state_dict()["layer4.2.conv3.weight"][0][0])
        for p in self.backbone_teacher.parameters():
            p.requires_grad = False
        self.teacher_initialized = True

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: Backbone outputs.
        """
        x = self.backbone(img)
        return x


    def sanity_check(self):
        ckpt = torch.load(self.teacher_checkpoint, map_location="cuda:0")['state_dict']
        student = self.backbone.state_dict()
        teacher = self.backbone_teacher.state_dict()
        for k,v in ckpt.items():
            if k in student:
                print(f"\t (Student) Checking loaded weights for layer {k}...")
                print(v)
                print(student[k])
                print(v==student[k])
                if not torch.equal(v,student[k]):
                    raise ValueError
        for k,v in ckpt.items():
            if k in teacher:
                print(f"\t (Teacher) Checking loaded weights for layer {k}...")
                if not torch.equal(teacher[k],v):
                    raise ValueError

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # self.backbone_teacher.train()
        assert isinstance(img, list)
        if not self.teacher_initialized:
            self.init_teacher()
            self.sanity_check()
        # print(self.backbone.state_dict()["layer4.2.conv3.weight"][0][:5])
        # print(self.backbone_teacher.state_dict()["layer4.2.conv3.weight"][0][:5])
        # import sys
        # sys.exit(0)
        # multi-res forward passes
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([i.shape[-1] for i in img]),
                return_counts=True)[1], 0)
        start_idx = 0
        output = []
        output_teacher = []
        for end_idx in idx_crops:
            crop = torch.cat(img[start_idx:end_idx])
            _out = self.backbone(crop)    
            _out_teacher = self.backbone_teacher(crop)
            output.append(_out)
            output_teacher.append(_out_teacher)
            start_idx = end_idx
        output_neck = self.neck(output)[0]
        
#        torch.save(self.backbone_teacher.state_dict(), "/home/kis/code/mmselfsup/work_dirs/isic_2019_ssl/resnet50_from_imagenet/relative_loc_lr_0.2057_bsize512_200ep_data_split_1_from_imagenet_sup/teacher.pth")
        
        
        
        
        # print(sum([self.criterion_div(torch.Tensor(self.avg_pool(output[i][0]).detach().cpu()), self.avg_pool(output_teacher[i][0]).detach().cpu()) for i in range(len(output))]))
        loss_div = torch.sum(torch.stack([self.criterion_div(self.avg_pool(output[i][0]), self.avg_pool(output_teacher[i][0])) for i in range(len(output))])) / len(output)
        # print(self.head(output_neck))
        loss = self.head(output_neck)
        wandb.log({"train/loss": loss['loss']})
        loss['loss'] = self.beta * loss['loss'] + self.delta * loss_div
        
        wandb.log({"KD loss": loss_div})
        wandb.log({"Total loss (KD + SwAV)": loss['loss']})
        return loss
