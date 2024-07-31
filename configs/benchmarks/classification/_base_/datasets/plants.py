# dataset settings
data_source = 'ImageList'
dataset_type = 'SingleViewDataset'
img_norm_cfg = dict(mean=[0.4394, 0.4767, 0.3735], std=[0.1970, 0.1746, 0.2135]) # PLANTS NORMALIZATION
#eurosat stats: Mean: tensor([0.3455, 0.3813, 0.4089]); std: tensor([0.2023, 0.1354, 0.1137])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
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
data_train_list = '/home/kis/code/datasets/plant_disease/train_list_correct.txt'
data_val_list = '/home/kis/code/datasets/plant_disease/val_list_correct.txt'
data_train_root = '/home/kis/code/datasets/plant_disease/train'
data_val_root = '/home/kis/code/datasets/plant_disease/test'

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
