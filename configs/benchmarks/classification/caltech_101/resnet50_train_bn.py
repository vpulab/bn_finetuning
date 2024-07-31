_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/caltech_101.py',
    '../_base_/schedules/sgd_coslr-100e.py',
    # '../_base_/schedules/adam_coslr-300e.py',
    '../_base_/default_runtime.py',
]


n_classes = 102

n_per_class_test = (439, 408, 406, 170, 769, 33, 770, 20, 20, 25, 28, 24, 11, 103, 71, 20, 62, 64, 25, 20, 97, 26, 39, 39, 80, 25, 46, 46, 45, 24, 28, 35, 44, 29, 43, 47, 51, 38, 30, 39, 56, 43, 43, 24, 17, 15, 26, 73, 72, 19, 31, 61, 53, 10, 44, 62, 88, 36, 55, 53, 18, 40, 21, 19, 59, 15, 51, 36, 17, 19, 24, 17, 26, 30, 14, 32, 55, 36, 31, 18, 39, 16, 59, 34, 15, 39, 21, 61, 34, 42, 14, 58, 29, 59, 51, 209, 15, 35, 16, 29, 19, 36)
# class_imbalances = tuple([1.0 for n in range(n_classes)])
class_imbalances = tuple([1.0-n/sum(n_per_class_test) for n in n_per_class_test])

model = dict(head=dict(num_classes=n_classes, type='ClsHeadWeightedNonLinear', mlp=False, weight=class_imbalances, dropout_rate=0.0, with_avg_pool=True, in_channels=2048, bn_only=False), backbone=dict(frozen_stages=4, unfreeze_bn=True))

optimizer = dict(type='SGD', lr=0.01)

# img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
runner = dict(type='EpochBasedRunner', max_epochs=70)
evaluation = dict(interval=1, start=1, save_best="auto", topk= (1,), per_class=True, n_classes=n_classes)

# data = dict(imgs_per_gpu=128, train=dict(data_source=dict(ann_file='/home/kis/code/datasets/EuroSat/split_0/train.txt')),
#             val=dict(data_source=dict(ann_file='/home/kis/code/datasets/EuroSat/split_0/val.txt')))


# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=80, max_keep_ckpts=50)
