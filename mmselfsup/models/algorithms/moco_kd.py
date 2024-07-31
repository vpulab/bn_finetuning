# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import wandb

from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel
from ..utils import DistillKL



@ALGORITHMS.register_module()
class MoCoKD(BaseModel):
    """MoCo.

    Implementation of `Momentum Contrast for Unsupervised Visual
    Representation Learning <https://arxiv.org/abs/1911.05722>`_.
    Part of the code is borrowed from:
    `<https://github.com/facebookresearch/moco/blob/master/moco/builder.py>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        queue_len (int): Number of negative keys maintained in the queue.
            Defaults to 65536.
        feat_dim (int): Dimension of compact feature vectors. Defaults to 128.
        momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.999.
    """

    def __init__(self,
                 backbone,
                 backbone_teacher,
                 teacher_checkpoint,
                 neck=None,
                 head=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 init_cfg=None,
                 kd_T=4,
                 beta=0.1,
                 delta=1,
                 **kwargs):
        super(MoCoKD, self).__init__(init_cfg)
        assert neck is not None
        self.beta = beta
        self.delta = delta
        self.encoder_q = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.encoder_k = nn.Sequential(
            build_backbone(backbone), build_neck(neck))

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.backbone = self.encoder_q[0]
        self.neck = self.encoder_q[1]
        assert head is not None
        self.head = build_head(head)

        self.queue_len = queue_len
        self.momentum = momentum

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # create the queue
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # Build the teacher model
        self.teacher_checkpoint = teacher_checkpoint
        self.backbone_teacher = build_backbone(backbone_teacher)
        print("CLASS INIT")
        print(self.backbone_teacher.state_dict()["layer4.2.conv3.weight"][0][0])
        self.teacher_initialized = False
        # Define the KL criterion
        # self.criterion_div = DistillKL(kd_T)
        self.criterion_div = torch.nn.MSELoss()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_teacher(self):
        print("\n\n\n\n")
        print(f"Load teacher model checkpoint from {self.teacher_checkpoint}")
        print("\n\n\n\n")
        self.backbone_teacher.load_state_dict(torch.load(self.teacher_checkpoint)['state_dict'], strict=False)
        print("TEACHER INIT")
        print(self.backbone_teacher.state_dict()["layer4.2.conv3.weight"][0][0])
        for p in self.backbone_teacher.parameters():
            p.requires_grad = False
        self.teacher_initialized = True


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x


    def sanity_check(self):
        ckpt = torch.load(self.teacher_checkpoint, map_location="cuda:0")['state_dict']
        student = self.encoder_q.state_dict()
        teacher = self.backbone_teacher.state_dict()
        for k,v in ckpt.items():
            if k in student:
                print(f"\t (Student) Checking loaded weights for layer {k}...")
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
        assert isinstance(img, list)


        if not self.teacher_initialized:
            self.init_teacher()
            self.teacher_initialized = True
            self.sanity_check()
	
        # print(self.backbone_teacher.state_dict()["layer4.2.conv3.weight"][0][0])
        # print(self.encoder_q.state_dict()["0.layer4.2.conv3.weight"][0][0:10])
        # import sys 
        # sys.exit(0)
        im_q = img[0]
        im_k = img[1]
        # compute query features
        q = self.encoder_q(im_q)[0]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_key_encoder()

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)[0]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = batch_unshuffle_ddp(k, idx_unshuffle)

        
        x_teacher = torch.cat(tuple([self.avg_pool(output) for output in self.backbone_teacher(im_q)]))
        x_student = torch.cat(tuple([self.avg_pool(output) for output in self.backbone(im_q)]))
        loss_div = self.criterion_div(x_student, x_teacher)
        wandb.log({"train/KD loss": loss_div})
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        losses = self.head(l_pos, l_neg)
        
        wandb.log({"train/loss": losses['loss']})
        losses['loss'] = self.beta * losses['loss'] + self.delta * loss_div
        wandb.log({"train/Total loss (MoCo + KD)": losses['loss']})

        # update the queue
        self._dequeue_and_enqueue(k)

        return losses

    def forward_test(self, img, **kwargs):
        """Forward computation during testing to compute losses.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(img, list)
        im_q = img[0]
        im_k = img[1]
        # compute query features
        q = self.encoder_q(im_q)[0]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)[0]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        losses = self.head(l_pos, l_neg)


        return losses
