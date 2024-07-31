# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmcv.utils import print_log
from ray import tune

from .base import BaseDataset
from .builder import DATASETS
from .utils import to_numpy


@DATASETS.register_module()
class SingleViewDataset(BaseDataset):
    """The dataset outputs one view of an image, containing some other
    information such as label, idx, etc.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(SingleViewDataset, self).__init__(data_source, pipeline,
                                                prefetch)
        self.gt_labels = self.data_source.get_gt_labels()

    def __getitem__(self, idx):
        label = self.gt_labels[idx]
        img = self.data_source.get_img(idx)
        img = self.pipeline(img)
        if self.prefetch:
            img = torch.from_numpy(to_numpy(img))
        return dict(img=img, label=label, idx=idx)

    def evaluate(self, results, logger=None, topk=(1, 5), per_class=False, n_classes=2):
        """The evaluation function to output accuracy.

        Args:
            results (dict): The key-value pair is the output head name and
                corresponding prediction values.
            logger (logging.Logger | str | None, optional): The defined logger
                to be used. Defaults to None.
            topk (tuple(int)): The output includes topk accuracy.
            per_class (bool): Whether to evaluate the accuracy per-class
        """
        eval_res = {}
        for name, val in results.items():
            val = torch.from_numpy(val)
            target = torch.LongTensor(self.data_source.get_gt_labels())
            assert val.size(0) == target.size(0), (
                f'Inconsistent length for results and labels, '
                f'{val.size(0)} vs {target.size(0)}')

            if not per_class:
                num = val.size(0)
                _, pred = val.topk(max(topk), dim=1, largest=True, sorted=True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))  # [K, N]
                for k in topk:
                    correct_k = correct[:k].contiguous().view(-1).float().sum(
                        0).item()
                    acc = correct_k * 100.0 / num
                    eval_res[f'{name}_top{k}'] = acc
                    if logger is not None and logger != 'silent':
                        print_log(f'{name}_top{k}: {acc:.03f}', logger=logger)
            else:
                mean = 0
                for label in range(n_classes):
                    # print(label)
                    target_indices = np.where(target == label)
                    target_class_labels = torch.LongTensor(target[target_indices])
                    # print(target_class_labels)
                    current_scores = val[target_indices]
                    # print(current_scores)
                    # print_log(
                    #             f"{current_scores.shape}",
                    #             logger=logger)

                    assert current_scores.size(0) == target_class_labels.size(0), \
                        "Inconsistent length for results and labels, {} vs {}".format(
                            current_scores.size(0), target_class_labels.size(0))
                    num = current_scores.size(0)
                    # print(num)
                    _, pred = current_scores.topk(1, dim=1, largest=True, sorted=True)
                    pred = pred.t()
                    correct = pred.eq(target_class_labels.view(1, -1).expand_as(pred))  # KxN
                    for k in topk:
                        correct_k = correct[:k].reshape(-1).float().sum(0).item()  # FIX
                        acc = correct_k * 100.0 / num
                        mean += acc
                        # if logger is not None and logger != 'silent':
                        #     print_log(
                        #         "Accuracy for class with label \'{}\': {:.03f}".format(label, acc),
                        #         logger=logger)
                # if logger is not None and logger != 'silent':
                #     print_log(
                #         "Mean per-class accuracy on: {:.03f}".format(mean/n_classes),
                #         logger=logger)
                eval_res[f'val_per_class_acc'] = mean / n_classes
                # tune.report(mean_acc=mean/n_classes)

                # print(self.runner)
        return eval_res
