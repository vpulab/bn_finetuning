_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar10.py',
    '../_base_/schedules/sgd_coslr-100e.py',
    '../_base_/default_runtime.py',
]

# model settings
class_imbalances = tuple([1.0 for n in range(10)])
model = dict(head=dict(num_classes=10, type='ClsHeadWeightedNonLinear', mlp=False, weight=class_imbalances, dropout_rate=0.0, with_avg_pool=True, in_channels=2048, bn_only=False), backbone=dict(frozen_stages=4, unfreeze_bn=False))

# optimizer
optimizer = dict(type='SGD', lr=0.1)

data = dict(imgs_per_gpu=12)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=28)
evaluation = dict(interval=1, start=1, save_best="auto", topk= (1,), per_class=True, n_classes=10)
checkpoint_config = dict(interval=50)
