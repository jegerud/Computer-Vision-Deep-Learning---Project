import sys
# assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import functools
import os
import json
import time
from timeit import default_timer as timer 
import click
import torch
import torchvision
from pprint import pprint
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import csv
import tqdm
from torchvision import models
from data_utils.utils.generate_train_val import generate_train_val
from config.resnet import Dataset, num_classes, gpu_transform, epochs, batch_size
from utils.torch_utils import to_cuda, move_to
from config.utils import get_dataset_dir
from modelling import ResNet
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights


def test():
    dataset = Dataset()
    # dataset_train = dataset.dataset_train
    # dataset_val = dataset.dataset_val
    dataloader_train = dataset.dataloader_train
    dataloader_val = dataset.dataloader_val

    model_name = 'resnet_it1.pt'
    model = torch.load(model_name)
    model.eval()


if __name__ == "__main__":
    test()