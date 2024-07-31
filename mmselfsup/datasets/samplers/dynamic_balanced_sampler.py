# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import random
import mmcv
from mmcv.runner import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler


class DynamicBalancedSampler(Sampler):

    def __init__(self,
                 labels,
                 dataset,
                 num_classes,
                 batch_size,
                 subset_size=0.1,
                 refresh_rate=8,
                 shuffle=True,
                 replace=False):
        self.shuffle = shuffle
        self.replace = replace
        self.unif_sampling_flag = False
        self.num_classes = num_classes
        self.dataset = dataset
        self.batch_size = batch_size
        self.refresh_rate = refresh_rate
        self.subset_size = subset_size
        self.num_subset_samples = int(len(self.dataset) * self.subset_size) 
        self.indices = np.random.randint(len(self.dataset), size=self.num_subset_samples)
        self.instance_labels = labels
        self.current_iter = 0
        assert batch_size % num_classes == 0


    def __str__(self):
        return "DynamicBalancedSampler"


    def __repr__(self):
        return "DynamicBalancedSampler"


    def make_class_lists(self):
        self.per_class_indices = dict()
        for num in range(self.num_classes):
            indices = list()
            for i in self.indices:
                instance_label = self.instance_labels[i]
                assert 0 <= instance_label < self.num_classes
                if instance_label == num:
                    indices.append(i)
            self.per_class_indices[num] = indices
            mmcv.print_log(f"New sampled lists: {self.per_class_indices}")


    def __len__(self):
        return self.batch_size * (self.num_subset_samples // self.batch_size)

    def _update_indices(self):

        max_attempts = 20

        attempt = 0
        while True:
            if attempt == max_attempts:
                mmcv.print_log("MAX NUMBER OF DATASET SUBSAMPLINGS REACHED. COULDN'T SAMPLE A DATASET WHERE ALL CLASSES ARE PRESENT")
                raise RuntimeError
            class_presence = []
            self.indices = np.random.randint(len(self.dataset), size=self.num_subset_samples)
            for index in self.indices:
                label = self.instance_labels[index]
                if label not in class_presence:
                    class_presence.append(label)
                if len(class_presence) == self.num_classes:
                    return
            attempt += 1 



    def __iter__(self):


        if self.current_iter == 0 and self.refresh_rate != -1:
            self._update_indices()
            self.current_iter += 1
        elif self.current_iter == self.refresh_rate-1:
            self.current_iter = 0
        else:
            self.current_iter += 1

        if self.shuffle:
            self.reshuffle_indices = self.make_class_lists()

        samples_per_class = self.batch_size // self.num_classes
        batches = self.num_subset_samples // self.batch_size
        indices = []

        for _ in range(batches):
            for num in range(self.num_classes):
                list_per_class = self.per_class_indices[num]
                for i in range(samples_per_class):
                    selected_indices = random.choices(list_per_class, k=samples_per_class)
                    indices += selected_indices

        return iter(indices)
