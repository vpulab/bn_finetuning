# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import random
from mmcv.runner import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler


class BalancedSampler(Sampler):

    def __init__(self,
                 labels,
                 dataset,
                 num_classes,
                 batch_size,
                 shuffle=True,
                 replace=False):
        self.shuffle = shuffle
        self.replace = replace
        self.unif_sampling_flag = False
        self.num_classes = num_classes
        self.dataset = dataset
        self.batch_size = batch_size
        # print(f"Making class lists")
        self.instance_labels = labels
        self.make_class_lists()
        self.current_iter = 0
        assert batch_size % num_classes == 0
        # print(f"Sampler built")


    def make_class_lists(self):
        self.per_class_indices = dict()
        for num in range(self.num_classes):
            indices = list()
            for i in range(len(self.dataset)):
                instance_label = self.instance_labels[i]
                assert 0 <= instance_label < self.num_classes
                if instance_label == num:
                    indices.append(i)
            if self.shuffle:
                random.shuffle(indices)
            self.per_class_indices[num] = indices


    def __len__(self):
        return self.batch_size * (len(self.dataset) // self.batch_size)


    def __iter__(self):
        if self.shuffle:
            self.reshuffle_indices = self.make_class_lists()

        samples_per_class = self.batch_size // self.num_classes
        batches = len(self.dataset) // self.batch_size
        indices = []

        for _ in range(batches):
            for num in range(self.num_classes):
                list_per_class = self.per_class_indices[num]
                for i in range(samples_per_class):
                    selected_indices = random.choices(list_per_class, k=samples_per_class)
                    indices += selected_indices

        return iter(indices)
