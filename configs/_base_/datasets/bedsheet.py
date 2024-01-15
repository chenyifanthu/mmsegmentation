dataset_type = 'BedsheetDataset'
data_root = '/root/LaundryDataset/data'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale_factor=(0.8, 1.25), keep_ratio=True),
    dict(type='RandomRotate', prob=1.0, degree=180),
    dict(type='RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomCrop', crop_size=(512, 512)),
    dict(type='PhotoMetricDistortion'),
    dict(type='RGB2Gray', prob=0.5),
    dict(type="AdjustGamma", gamma=(0.5, 2.0)),
    # dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # # add loading annotation after ``Resize`` because ground truth
    # # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    # dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
