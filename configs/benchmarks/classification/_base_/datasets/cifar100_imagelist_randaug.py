# dataset settings
data_source = 'ImageList'
dataset_type = 'SingleViewDataset'
img_norm_cfg = dict(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
train_pipeline = [
    dict(
        type='RandomAug',
        input_size=32,
        color_jitter=0.4,
        auto_augment='rand-m12-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761))
]
test_pipeline = []

# prefetch
prefetch = False
if not prefetch:
    # train_pipeline.extend(
    #     [dict(type='ToTensor'),
    #      dict(type='Normalize', **img_norm_cfg)])
    test_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data_train_list = '/home/kis/code/datasets/cifar100/train_list_labeled.txt'
data_val_list = '/home/kis/code/datasets/cifar100/test_list_labeled.txt'
data_train_root = data_val_root = '/home/kis/code/datasets/cifar100'

# dataset summary
data = dict(
    imgs_per_gpu=128,
    workers_per_gpu=4,
    drop_last=False,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix=data_train_root,
            ann_file=data_train_list,
        ),
        pipeline=train_pipeline,
        prefetch=prefetch,
        use_cutmix=False),

    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix=data_val_root,
            ann_file=data_val_list,
        ),
        pipeline=test_pipeline,
        prefetch=prefetch,
        use_cutmix=False
        ))
evaluation = dict(interval=10, topk=(1, 5))
