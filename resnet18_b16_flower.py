_base_ = [
    '../_base_/models/resnet18.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
    ]

# model
model = dict(
    head = dict(
        num_classes=5,
        topk = (1, )
    )
)

# data
data = dict(
    # batchsize和workers
    samples_per_gpu = 32,
    workers_per_gpu=2,
    # 训练集路径
    train = dict(
        data_prefix = '/root/mmclassification/data/data_split/train',
        ann_file = '/root/mmclassification/data/data_split/train.txt',
        classes = '/root/mmclassification/data/data_split/classes.txt'
    ),
    # 验证集路径
    val = dict(
        data_prefix = '/root/mmclassification/data/data_split/val',
        ann_file = '/root/mmclassification/data/data_split/val.txt',
        classes = '/root/mmclassification/data/data_split/classes.txt'
    ),
)

# 定义评估方法
evaluation = dict(metric_options={'topk':(1,)})

# 优化器
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=100)

# runtime
checkpoint_config = dict(interval=10)
load_from = '/root/mmclassification/checkpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth'