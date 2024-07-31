_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar100_imagelist.py',
    '../_base_/schedules/sgd_coslr-100e.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(head=dict(num_classes=100))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=350)
checkpoint_config = dict(interval=50)
