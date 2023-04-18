import torch
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
# from tops.config import LazyConfig
from os import PathLike
import csv
import json

def batch_collate(batch):
    elem = batch[0]
    batch_ = {key: default_collate([d[key] for d in batch]) for key in elem}
    return batch_

def batch_collate_val(batch):
    """
        Same as batch_collate, but removes boxes/labels from dataloader
    """
    elem = batch[0]
    ignore_keys = set(("boxes", "labels"))
    batch_ = {key: default_collate([d[key] for d in batch]) for key in elem if key not in ignore_keys}
    return batch_


def class_id_to_name(labels, label_map: list):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy().tolist()
    return [label_map[idx] for idx in labels]


def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}]


def get_checkpoint(model_name):
    with open('losses.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        losses = [float(l) for l in [row for row in csv_reader][0]]
    
    start_epoch = 0
    f = open(f"checkpoints/{model_name}.json")
    data = json.load(f)

    start_epoch = data["current_epoch"]

    return losses, start_epoch