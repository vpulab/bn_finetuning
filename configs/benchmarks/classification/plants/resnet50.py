_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/plants.py',
    '../_base_/schedules/sgd_coslr-100e.py',
    # '../_base_/schedules/adam_coslr-300e.py',
    '../_base_/default_runtime.py',
]

# Apply weighted loss for Plant Diseases
n_classes = 38

total_training_samples = 43456
n_instances_per_class = [861, 800, 1468, 1528, 1417, 1838, 1341, 1273, 339, 299, 1316, 798, 684, 762, 930, 1107, 788, 1702, 365, 1202, 220, 800, 288, 1183, 504, 4406, 297, 411, 800, 4072, 4286, 1124, 888, 122, 954, 944, 842, 497]


class_imbalances = tuple([1 - (n/total_training_samples) for n in n_instances_per_class])

model = dict(head=dict(num_classes=n_classes, type='ClsHeadWeightedNonLinear', mlp=True, weight=class_imbalances, dropout_rate=0.0, with_avg_pool=True, in_channels=2048), backbone=dict(frozen_stages=4)) # uncomment to fine-tuning instead of freezing the weights
optimizer = dict(type='SGD', lr=0.01)


runner = dict(type='EpochBasedRunner', max_epochs=28)
evaluation = dict(interval=1, start=1, save_best="auto", topk= (1,), per_class= True, n_classes=n_classes)



data = dict(imgs_per_gpu=256)


# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=80, max_keep_ckpts=50)
