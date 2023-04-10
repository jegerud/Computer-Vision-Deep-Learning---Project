import torchvision
import torch
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision import transforms

from data_utils import RoadDamageDataset
from data_utils.transforms import (ToTensor, Normalize, Resize)#, GroundTruthBoxesToAnchors)
from .utils import get_dataset_dir, get_output_dir
from modelling import AnchorBoxes

country = "Norway"
train_set_dir = f"/Users/kristian/Documents/Trondheim/4. semester/TDT4265/TDT4265_project/data/road_damage/{country}/train"
# train_set_dir = f"/cluster/projects/itea_lille-idi-tdt4265/datasets/rdd2022/RDD2022/{country}/train"
imshape = (300,300),
image_channels = 3,
batch_size = 32
epochs = 5
train_val_split = 0.20
num_classes = 4 + 1

anchors = AnchorBoxes(
    feature_sizes=[[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[8, 8], [16, 16], [32, 32], [64, 64], [100, 100], [300, 300]],
    min_sizes=[[30, 30], [60, 60], [111, 111], [162, 162], [213, 213], [264, 264], [315, 315]],
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    image_shape=(300, 300),
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

train_cpu_transform = torchvision.transforms.Compose([
    ToTensor(),
    Resize(imshape=imshape),
    # RandomSampleCrop(),
    # RandomHorizontalFlip(),
    # Resize(imshape=imshape)
    # GroundTruthBoxesToAnchors(anchors=anchors, iou_threshold=0.5),
])
val_cpu_transform = torchvision.transforms.Compose([
    ToTensor(),
    Resize(imshape=imshape),
    # ToTensor(),
    # Resize(imshape=imshape),
])
gpu_transform = torchvision.transforms.Compose(transforms=[
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Dataset():
    def __init__(self):
        """Dataset for road damage data.
        Args:
            data_dir: the root of the road damage dataset, the directory is split into test and 
            train directory:
                
        """
        # self.data_dir = data_dir
        self.train_val_split = train_val_split
        self.dataset_train = RoadDamageDataset(
            data_dir=get_dataset_dir(train_set_dir), 
            split="train", 
            transform=train_cpu_transform, 
            remove_empty=True
        )

        self.dataset_val = RoadDamageDataset(
            data_dir=get_dataset_dir(train_set_dir), 
            split="val", 
            transform=val_cpu_transform,
            remove_empty=True
        )

        self.dataloader_train = torch.utils.data.DataLoader(
            dataset=self.dataset_train, 
            num_workers=4, 
            pin_memory=True, 
            shuffle=True, 
            batch_size=batch_size, 
            collate_fn=self.dataset_train.batch_collate
            # drop_last=True
        )

        self.dataloader_val = torch.utils.data.DataLoader(
            dataset=self.dataset_val, 
            num_workers=4, 
            pin_memory=True, 
            shuffle=True, 
            batch_size=batch_size, 
            collate_fn=self.dataset_val.batch_collate,
            # drop_last=True
        )

