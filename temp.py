from mmseg.datasets import BedsheetDataset
from mmengine.registry import init_default_scope
import numpy as np
import cv2

init_default_scope('mmseg')

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
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs')
]

dataset = BedsheetDataset('/Users/chenyifan/data/LaundrySeg/data', "train", pipeline=train_pipeline)

data = dataset[0]
print(data)
img = data['inputs'].data.numpy().transpose(1, 2, 0)
img = img.astype(np.uint8)
label = data['data_samples'].gt_sem_seg.data.numpy().astype(np.uint8).squeeze()

label[label == 255] = 0
label[label == 1] = 255
print(img.shape, label.shape)

cv2.imshow('img', img)
cv2.imshow('label', label)
cv2.waitKey(0)