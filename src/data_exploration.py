import sys
assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import click
import torch
import tops
from sklearn.model_selection import train_test_split
from pathlib import Path
from ssd import utils
from ssd.data.utils.preprocess import get_mean_box_area, get_number_of_boxes, get_label_mean_center
from ssd.data.utils.generate_train_val import generate_train_val
from tops.config import instantiate
torch.backends.cudnn.benchmark = True


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def load_data(config_path: Path):
    cfg = utils.load_config(config_path)
    generate_train_val(cfg=cfg)

    tops.init(cfg.output_dir)
    tops.set_AMP(cfg.train.amp)
    tops.set_seed(cfg.train.seed)
    dataloader_train = instantiate(cfg.data_train.dataloader)
    dataloader_val = instantiate(cfg.data_val.dataloader)
    
    # train_cocoGt = dataloader_train.dataset.get_annotations_as_coco()
    # val_cocoGt = dataloader_val.dataset.get_annotations_as_coco()

    subsamples_train = (dataloader_train.dataset.image_ids)
    means_train = get_mean_box_area(dataloader=dataloader_train, subsamples=subsamples_train)
    counts_train = get_number_of_boxes(dataloader=dataloader_train, subsamples=subsamples_train)
    center_train = get_label_mean_center(dataloader=dataloader_train, subsamples=subsamples_train)
    
    print(f"\nTRAIN: {len(subsamples_train)} samples")
    for key in means_train:
        print(f"\nLabel: {key}")
        print(f"  Counts in set: {counts_train[key]}")
        print(f"  Mean size: {means_train[key]}")
        print(f"  Mean center: {center_train[key]}")


    subsamples_val = (dataloader_val.dataset.image_ids)
    means_val = get_mean_box_area(dataloader=dataloader_val, subsamples=subsamples_val)
    counts_val = get_number_of_boxes(dataloader=dataloader_val, subsamples=subsamples_val)
    center_val = get_label_mean_center(dataloader=dataloader_val, subsamples=subsamples_val)
    
    print(f"\n\nVALIDATION: {len(subsamples_val)} samples")
    for key in means_val:
        print(f"\nLabel: {key}")
        print(f"  Counts in set: {counts_val[key]}")
        print(f"  Mean size: {means_val[key]}")
        print(f"  Mean center: {center_val[key]}")


if __name__ == "__main__":
    load_data()
