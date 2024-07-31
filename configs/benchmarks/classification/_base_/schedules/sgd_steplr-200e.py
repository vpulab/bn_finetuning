# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-5)

# learning policy
lr_config = dict(policy='step', step=[60, 120, 160])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
