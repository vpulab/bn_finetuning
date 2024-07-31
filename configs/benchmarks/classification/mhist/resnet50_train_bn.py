_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/mhist.py',
    '../_base_/schedules/sgd_coslr-100e.py',
    # '../_base_/schedules/adam_coslr-300e.py',
    '../_base_/default_runtime.py',
]


n_classes = 2

class_imbalances = tuple([1.0 for n in range(n_classes)])

model = dict(head=dict(num_classes=n_classes, type='ClsHeadWeightedNonLinear', mlp=False, weight=class_imbalances, dropout_rate=0.0, with_avg_pool=True, in_channels=2048, bn_only=False), backbone=dict(frozen_stages=4, unfreeze_bn=True))

optimizer = dict(type='SGD', lr=0.01)

# img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
runner = dict(type='EpochBasedRunner', max_epochs=70)
evaluation = dict(interval=1, start=1, save_best="auto", topk= (1,), per_class=True, n_classes=n_classes)

data = dict(imgs_per_gpu=16)


# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=80, max_keep_ckpts=50)
