import torchvision
import torch
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
# from ssd.data import VOCDataset
from ssd.data import RoadDamageDataset
from ssd.modeling import backbones, AnchorBoxes, SSD300
from ssd import utils
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, RandomHorizontalFlip, RandomSampleCrop, Normalize, Resize,
    GroundTruthBoxesToAnchors)
from .utils import get_dataset_dir, get_output_dir

train = dict(
    batch_size=32,
    amp=True,  # Automatic mixed precision
    log_interval=20,
    seed=0,
    epochs=20,
    _output_dir=get_output_dir(),
    imshape=(300, 300),
    image_channels=3
)

anchors = L(AnchorBoxes)(
    feature_sizes=[[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[8, 8], [16, 16], [32, 32], [64, 64], [100, 100], [300, 300]],
    min_sizes=[[30, 30], [60, 60], [111, 111], [162, 162], [213, 213], [264, 264], [315, 315]],
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

model = L(SSD300)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=10 + 1  # Add 1 for background
)

# Keep the model, except change the backbone and number of classes
model.feature_extractor = L(backbones.VGG)()
model.num_classes = 4 + 1

train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(ToTensor)(),
    L(RandomHorizontalFlip)(),
    L(Resize)(imshape="${train.imshape}"),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])
val_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
])
gpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(Normalize)(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

optimizer = L(torch.optim.Adam)(
    lr=5e-3, weight_decay=0.0005
)

schedulers = dict(
    linear=L(LinearLR)(start_factor=0.1, end_factor=1, total_iters=500),
    multistep=L(MultiStepLR)(milestones=[70000, 9000], gamma=0.1)
)

data_train = dict(
    # dataset=L(torch.utils.data.ConcatDataset)(datasets=[
    #     L(RoadDamageDataset)(
    #         data_dir=get_dataset_dir("road_damage/Czech/train"), 
    #         # split="train", 
    #         transform=train_cpu_transform, 
    #         keep_difficult=True, 
    #         remove_empty=True
    #     ),
    # ]),
    dataset=L(RoadDamageDataset)(
        data_dir=get_dataset_dir("road_damage/Czech/train"), 
        # split="train", 
        transform=train_cpu_transform, 
        keep_difficult=True, 
        remove_empty=True
    ),
    dataloader=L(torch.utils.data.DataLoader)(
        dataset="${..dataset}", 
        num_workers=4, 
        pin_memory=True, 
        shuffle=True, 
        batch_size="${...train.batch_size}", 
        collate_fn=utils.batch_collate,
        drop_last=True
    ),
    gpu_transform=gpu_transform
)
data_val = dict(
    dataset=L(RoadDamageDataset)(
        data_dir=get_dataset_dir("road_damage/Czech/val"), 
        # split="val", 
        transform=val_cpu_transform, 
        remove_empty=True
    ),
    dataloader=L(torch.utils.data.DataLoader)(
        dataset="${..dataset}", 
        num_workers=4, 
        pin_memory=True, 
        shuffle=False, 
        batch_size="${...train.batch_size}", 
        collate_fn=utils.batch_collate_val
    ),
    gpu_transform=gpu_transform
)

# data_train.dataset = L(torch.utils.data.ConcatDataset)(datasets=[
#     L(RoadDamageDataset)(data_dir=get_dataset_dir("road_damage/Czech/train"), split="train", transform=train_cpu_transform, keep_difficult=True, remove_empty=True),
#     # L(RoadDamageDataset)(data_dir=get_dataset_dir("VOCdevkit/VOC2012"), split="train", transform=train_cpu_transform, keep_difficult=True, remove_empty=True)
# ])
# data_val.dataset = L(RoadDamageDataset)(
#     data_dir=get_dataset_dir("road_damage/Czech/train"), 
#     split="val", 
#     transform=val_cpu_transform, 
#     remove_empty=False)
# data_val.gpu_transform = gpu_transform
# data_train.gpu_transform = gpu_transform

label_map = {idx: cls_name for idx, cls_name in enumerate(RoadDamageDataset.class_names)}