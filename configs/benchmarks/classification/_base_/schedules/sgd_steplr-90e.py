# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(policy='step',
                 step=[30, 60, 80],
                 warmup='linear',
                 warmup_iters=10,
                 warmup_ratio=0.0001, # cannot be 0
                 warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=90)
