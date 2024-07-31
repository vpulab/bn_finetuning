_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar100_imagelist.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(head=dict(num_classes=100))

# optimizer
optimizer = dict(type='Adam', lr=0.0027412)
data=dict(imgs_per_gpu=256)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=350)
checkpoint_config = dict(interval=50)
