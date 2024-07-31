# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import random
from mmcv.runner import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torch import tensor
import torch.nn as nn
from tqdm import tqdm


class LossSampler(Sampler):

    def __init__(self,
                 #model,
                 labels,
                 dataset,
                 batch_size,
                 num_classes=8,
                 refresh_rate=8,
                 subset_size=0.1,
                 shuffle=True,
                 replace=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.subset_size = subset_size
        self.batch_size = batch_size
        #self.model = model
        self.refresh_rate = refresh_rate
        self.instance_labels = labels
        self.current_iter = 0
        self.indices = []
        self.losses = []
        self.loss_dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=1,
                shuffle=False,
                drop_last=False)

    def update_current_losses(self):

        if self.current_iter == 0:
            all_losses = []
            all_labels = []
            
            
            for idx, batch in enumerate(self.loss_dataloader):
                x = self.model.forward_train(batch['img'].cuda(), batch['label'].cuda())
                all_labels += batch['label'].cpu().detach().numpy().tolist()
                all_losses += x['loss'].cpu().detach().numpy().tolist()
                
            
            assert self.instance_labels == all_labels
            self.losses = np.asarray(all_losses)
            num_subset_samples = int(len(self.dataset) * self.subset_size) 
            self.indices = np.argpartition(self.losses, (-1) * num_subset_samples)[(-1) * num_subset_samples:]
        

    def __len__(self):
        return self.batch_size * (int(len(self.dataset) * self.subset_size) // self.batch_size)


    def update_loss_criterion(self):
        samples_per_class = [0 for _ in range(self.num_classes)]
        for instance_index in self.indices:
            label = self.instance_labels[instance_index]
            samples_per_class[label] += 1
        total_training_samples = len(self.indices)
        class_imbalances = tuple([1 - n/total_training_samples for n in samples_per_class])
        return class_imbalances



    def __iter__(self):
        
        if self.current_iter == self.refresh_rate-1:
            self.current_iter = 0
        else:
            self.current_iter += 1

        return iter(self.indices)
