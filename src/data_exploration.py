import sys
assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import click
import torch
import tops
from pathlib import Path
from ssd import utils
from tops.config import instantiate
torch.backends.cudnn.benchmark = True


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def load_data(config_path: Path):
    cfg = utils.load_config(config_path)

    tops.init(cfg.output_dir)
    tops.set_AMP(cfg.train.amp)
    tops.set_seed(cfg.train.seed)
    dataloader_train = instantiate(cfg.data_train.dataloader)
    dataloader_val = instantiate(cfg.data_val.dataloader)
    
    # train_cocoGt = dataloader_train.dataset.get_annotations_as_coco()
    val_cocoGt = dataloader_val.dataset.get_annotations_as_coco()

    subsamples = (dataloader_train.dataset.image_ids)[0:10]
    for subsample in subsamples:
        print(dataloader_train.dataset._get_annotation(subsample))
    # print(f"Dataloader_val: {len(dataloader_val.dataset.image_ids)}")
    

if __name__ == "__main__":
    load_data()
