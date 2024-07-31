# dataset settings
data_source = 'ImageList'
dataset_type = 'SingleViewDataset'
img_norm_cfg = dict(mean=[0.4460, 0.4311, 0.4319], std=[0.2903, 0.2884, 0.2956]) # Cars normalization

train_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
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
data_train_list = '/home/kis/code/datasets/cars/devkit/train.txt'
data_val_list = '/home/kis/code/datasets/cars/devkit/test.txt'
data_train_root = '/home/kis/code/datasets/cars/cars_train'
data_val_root = '/home/kis/code/datasets/cars/cars_test'

# dataset summary
data = dict(
    imgs_per_gpu=8,
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
