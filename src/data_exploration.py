import sys
assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import click
import torch
from data_utils.utils.preprocess import get_mean_box_area, get_number_of_boxes, get_label_mean_center
from data_utils.utils.generate_train_val import generate_train_val
# from config.resnet import data_train, data_val, train_set_dir, batch_size
from config.resnet import Dataset
from data_utils import RoadDamageDataset
from config.utils import get_dataset_dir

torch.backends.cudnn.benchmark = True

def data_exploration():
    generate_train_val()

    dataset = Dataset()
    dataset_train = dataset.dataset_train
    dataset_val = dataset.dataset_val
    dataloader_train = dataset.dataloader_train
    dataloader_val = dataset.dataloader_val

    # print(f"Data_train: {len(dataloader_train)}")
    subsamples_train = (dataset_train.image_ids)
    means_train = get_mean_box_area(dataloader=dataloader_train, subsamples=subsamples_train)
    counts_train = get_number_of_boxes(dataloader=dataloader_train, subsamples=subsamples_train)
    center_train = get_label_mean_center(dataloader=dataloader_train, subsamples=subsamples_train)
    
    print(f"\nTRAIN: {len(subsamples_train)} samples")
    for key in means_train:
        print(f"\nLabel: {key}")
        print(f"  Counts in set: {counts_train[key]}")
        print(f"  Mean size: {means_train[key]}")
        print(f"  Mean center: {center_train[key]}")

if __name__ == "__main__":
    data_exploration()
