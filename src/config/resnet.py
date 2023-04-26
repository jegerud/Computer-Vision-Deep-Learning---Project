import torchvision
import torch
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2 
from data_utils import RoadDamageDataset, RoadDamageTestDataset
from data_utils.transforms import (ToTensor, Normalize, Resize, RandomHorizontalFlip)#, GroundTruthBoxesToAnchors)
from .utils import get_dataset_dir, get_output_dir
from modelling import AnchorBoxes

country = "Norway"
# base_dir = "/Users/kristian/Documents/Trondheim/4. semester/TDT4265/TDT4265_project/data/road_damage"
base_dir = "/cluster/projects/itea_lille-idi-tdt4265/datasets/rdd2022/RDD2022"

train_set_dir = f"{base_dir}/Norway/train"
train_set_dir_czech = f"{base_dir}/Czech/train"
train_set_dir_india = f"{base_dir}/India/train"
train_set_dir_japan = f"{base_dir}/Japan/train"
train_set_dir_us = f"{base_dir}/United_States/train"
train_set_dir_china_drone = f"{base_dir}/China_Drone/train"
train_set_dir_china_mbike = f"{base_dir}/China_MotorBike/train"

test_set_dir = f"{base_dir}/Norway/test"

imshape = (640,640),
image_channels = 3
batch_size = 8
test_batch_size = 1
epochs = 80
train_val_split = 0.10
num_classes = 4 + 1

train_cpu_transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_cpu_transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_cpu_transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        self.dataset_norway = RoadDamageDataset(
        # self.dataset_train = RoadDamageDataset(
            data_dir=get_dataset_dir(train_set_dir), 
            split="train", 
            country="Norway",
            transform=train_cpu_transform, 
            remove_empty=True
        )

        self.dataset_us = RoadDamageDataset(
            data_dir=get_dataset_dir(train_set_dir_us), 
            split="train", 
            country="United_States",
            transform=train_cpu_transform, 
            remove_empty=True
        )

        self.dataset_czech = RoadDamageDataset(
            data_dir=get_dataset_dir(train_set_dir_czech), 
            split="train", 
            country="Czech",
            transform=train_cpu_transform, 
            remove_empty=True
        )

        self.dataset_japan = RoadDamageDataset(
            data_dir=get_dataset_dir(train_set_dir_japan), 
            split="train", 
            country="Japan",
            transform=train_cpu_transform, 
            remove_empty=True
        )

        self.dataset_india = RoadDamageDataset(
            data_dir=get_dataset_dir(train_set_dir_india), 
            split="train", 
            country="India",
            transform=train_cpu_transform, 
            remove_empty=True
        )

        self.dataset_china_drone = RoadDamageDataset(
            data_dir=get_dataset_dir(train_set_dir_china_drone), 
            split="train", 
            country="China_Drone",
            transform=train_cpu_transform, 
            remove_empty=True
        )

        self.dataset_china_mbike = RoadDamageDataset(
            data_dir=get_dataset_dir(train_set_dir_china_mbike), 
            split="train", 
            country="China_Mbike",
            transform=train_cpu_transform, 
            remove_empty=True
        )

        self.dataset_train = torch.utils.data.ConcatDataset(
            datasets=[self.dataset_japan, self.dataset_czech, self.dataset_us, self.dataset_india, self.dataset_china_drone, self.dataset_china_mbike]
        )
        # self.dataset_train = torch.utils.data.ConcatDataset(
        #     datasets=[self.dataset_japan, self.dataset_czech, self.dataset_us, self.dataset_norway]
        # )

        self.dataset_val = RoadDamageDataset(
            data_dir=get_dataset_dir(train_set_dir), 
            split="val", 
            country="Norway",
            transform=val_cpu_transform,
            remove_empty=False
        )

        self.dataloader_train = torch.utils.data.DataLoader(
            dataset=self.dataset_train, 
            num_workers=4, 
            pin_memory=True, 
            shuffle=True, 
            batch_size=batch_size, 
            collate_fn=self.dataset_norway.batch_collate
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

label_map = {
    0: '__background__',
    1: 'D00', 
    2: 'D10',
    3: 'D20',
    4: 'D40'
}

class DatasetTest():
    def __init__(self):
        """
        Dataset used for testing the model on the road damage data.

        """
        self.dataset = RoadDamageTestDataset(
            data_dir=get_dataset_dir(test_set_dir), 
            country="Norway",
            transform=test_cpu_transform, 
            remove_empty=True
        )

        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset, 
            num_workers=4, 
            pin_memory=True, 
            shuffle=False, 
            batch_size=test_batch_size, 
            collate_fn=self.dataset.batch_collate_test
        )