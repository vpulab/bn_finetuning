from torch import nn

_base_ = [
    '../_base_/models/resnet18.py',
    '../_base_/datasets/cifar100_imagelist.py',
    '../_base_/schedules/sgd_steplr-200e.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(head=dict(num_classes=100))

# optimizer
optimizer = dict(type='SGD', lr=0.02882, momentum=0.9, weight_decay=5e-4)
evaluation = dict(interval=1, start=0, save_best="auto", topk= (1,), per_class=True, n_classes=100)
data = dict(imgs_per_gpu=100)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=50)
