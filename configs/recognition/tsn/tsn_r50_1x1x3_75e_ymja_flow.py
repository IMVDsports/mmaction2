# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        in_channels=10,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=4,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.001))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips=None)
# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/YMJA/rawframes_sampled/'
data_root_val = 'data/YMJA/rawframes_sampled/'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'data/YMJA/ymja_train_split_{split}_rawframes.txt'
ann_file_val = f'data/YMJA/ymja_val_split_{split}_rawframes.txt'
ann_file_test = f'data/YMJA/ymja_test_split_{split}_rawframes.txt'
img_norm_cfg = dict(
    mean=[128, 128], std=[128, 128], to_bgr=False) #TODO what should I set these to?
train_pipeline = [
    dict(type='SampleFrames', clip_len=5, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=5,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=5,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='flow_{}_{:05d}.jpg',
        modality='Flow',
        pipeline=train_pipeline,
        start_index=0),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='flow_{}_{:05d}.jpg',
        modality='Flow',
        pipeline=val_pipeline,
        start_index=0),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='flow_{}_{:05d}.jpg',
        modality='Flow',
        pipeline=test_pipeline,
        start_index=0))
# optimizer
optimizer = dict(
    type='SGD', lr=0.005, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[20, 35])
total_epochs = 50
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/YMJA/tsn_r50_1x1x3_30e_flow/'
load_from = None
resume_from = None
workflow = [('train', 1)]
