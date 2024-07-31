# optimizer
optimizer = dict(type='SGD', lr=0.3, momentum=0.9, weight_decay=1e-6)

# learning policy
lr_config = dict(policy='CosineRestart', periods=[10]*10, restart_weights=[1]*10, min_lr=1.0e-8, by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
