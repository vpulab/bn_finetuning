# dataset settings
data_source = 'ImageList'
dataset_type = 'SingleViewDataset'
img_norm_cfg = dict(mean=[0.4894, 0.5183, 0.5388], std=[0.2226, 0.2156, 0.2456]) # Aircraft normalization

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
data_train_list = '/home/kis/code/datasets/aircraft/data/mmselfsup_trainval_variant.txt'
data_val_list = '/home/kis/code/datasets/aircraft/data/mmselfsup_test_variant.txt'
data_train_root = data_val_root = '/home/kis/code/datasets/aircraft/data/images_small'

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
