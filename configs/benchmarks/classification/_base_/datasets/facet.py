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
data_root = '/home/kis/Desktop/rhome/kis/datasets/facet/'

data_train_list = data_root + 'train_seed_0.csv'
data_val_list = data_root + 'val_seed_0.csv'
data_test_list = data_root + 'test_val_seed_0.csv'

data_train_root = data_val_root = data_test_root = data_root + 'images_bb'


# dataset summary
data = dict(
    imgs_per_gpu=64,
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
            data_prefix=data_test_root,
            ann_file=data_test_list,
        ),
        pipeline=test_pipeline,
        prefetch=prefetch,
        use_cutmix=False
        ))
evaluation = dict(interval=10, topk=(1, 5))