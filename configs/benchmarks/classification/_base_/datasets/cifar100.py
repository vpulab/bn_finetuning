# dataset settings
data_source = 'CIFAR100'
dataset_type = 'SingleViewDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomHorizontalFlip'),
]
test_pipeline = []

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])
    test_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    imgs_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/cifar100',
        ),
        pipeline=train_pipeline,
        prefetch=prefetch,
        use_cutmix=False),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/cifar100',
        ),
        pipeline=test_pipeline,
        prefetch=prefetch,
        use_cutmix=False),
    test=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/cifar100',
        ),
        pipeline=test_pipeline,
        prefetch=prefetch,
        use_cutmix=False))
evaluation = dict(interval=10, topk=(1, 5))
