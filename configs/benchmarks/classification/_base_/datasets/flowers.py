# dataset settings
data_source = 'ImageList'
dataset_type = 'SingleViewDataset'
img_norm_cfg = dict(mean=[0.5179, 0.4110, 0.3297], std=[0.2971, 0.2496, 0.2855]) # Flowers normalization

train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    # dict(type='RandomVerticalFlip'),
    # dict(type='Resize', size=256),
    # dict(type='CenterCrop', size=224),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
]

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
data_train_list = '/home/kis/code/datasets/flowers/annotations/trainval_labeled.txt'
data_val_list = '/home/kis/code/datasets/flowers/annotations/test_labeled.txt'
data_train_root = data_val_root = '/home/kis/code/datasets/flowers/images'

# dataset summary
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=2,
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
