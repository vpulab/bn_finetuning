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


class DynamicRandomSampler(Sampler):

    def __init__(self,
                 # model,
                 labels,
                 dataset,
                 batch_size,
                 num_classes=8,
                 refresh_rate=2,
                 subset_size=0.1,
                 shuffle=True,
                 replace=False):
        # breakpoint()
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.subset_size = subset_size
        self.batch_size = batch_size
        # self.model = model
        self.refresh_rate = refresh_rate
        self.instance_labels = labels
        self.current_iter = 0
        self.num_subset_samples = int(len(self.dataset) * self.subset_size) 
        self.indices = None
        self._update_indices()
        self.class_imbalances = tuple([1 for _ in range(self.num_classes)])


    def __str__(self):
        return "DynamicRandomSampler"


    def __repr__(self):
        return "DynamicRandomSampler"


    def __len__(self):
        return self.batch_size * (int(len(self.dataset) * self.subset_size) // self.batch_size)

    def _update_indices(self):
        self.indices = np.random.randint(len(self.dataset), size=self.num_subset_samples)


    def update_loss_criterion(self, model):

        samples_per_class = np.zeros(self.num_classes, dtype=np.single)
        for instance_index in self.indices:
            label = self.instance_labels[instance_index]
            samples_per_class[label] += 1
        total_training_samples = len(self.indices)

        self.class_imbalances = tuple([1 - n/total_training_samples for n in samples_per_class])
        return self.class_imbalances


    def __iter__(self):
        
        if self.current_iter == 0:
            self._update_indices()
            self.current_iter += 1
        elif self.current_iter == self.refresh_rate-1:
            self.current_iter = 0
        else:
            self.current_iter += 1

        return iter(self.indices)
