import sys
# assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import functools
import os
import json
import numpy as np
import time
from timeit import default_timer as timer 
import click
import torch
import torchvision
from pprint import pprint
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim import lr_scheduler
import csv
import tqdm
from data_utils.utils.generate_train_val import generate_train_val
from config.resnet import Dataset, num_classes, epochs, batch_size
from utils.torch_utils import to_cuda, get_device
from utils.evaluate import evaluate
from config.resnet import train_set_dir, train_set_dir_czech, train_set_dir_us, train_set_dir_japan
from utils import get_checkpoint
from modelling import ResNet, MobileNet

def train_epoch(
        model, 
        dataloader,
        optimizer,
        # scheduler,
        epoch
    ):
    model.train()
    total_train_loss = []

    for data in tqdm.tqdm(dataloader, f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(get_device()) for image in images)
        targets = [{k: v.to(get_device()) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_train_loss.append(losses.item())
        losses.backward()
        optimizer.step()

        # with torch.set_grad_enabled(True):
            # output = model(images, targets)
            
            # loss = sum(loss for loss in output.values())
            # total_train_loss.append(loss.item())
            # loss.backward()
            # optimizer.step()
    
    train_loss = sum(total_train_loss) / len(dataloader)
    print(f"\ttrain_loss: {train_loss} | total_train_loss: {sum(total_train_loss)}")# | test_loss: {test_loss}")

    return total_train_loss

def validate(
        model, 
        dataloader
    ):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    idx = 0
    with torch.no_grad():
        # with torch.inference_mode():
        for data in tqdm.tqdm(dataloader, f"Validating on evaluation set"):
            images, targets = data
            images = list(image.to(get_device()) for image in images)
            targets = [{k: v.to(get_device()) for k, v in t.items()} for t in targets]
            
            detection = model(images)            
            metric.update(detection, targets)
            idx += 1

    m_ap = metric.compute()
    pprint(m_ap)
    # print(f"- mAP:      {m_ap['map']}")
    # print(f"- mAP 0.5:  {m_ap['map_50']}")
    # print(f"- mAP 0.75: {m_ap['map_75']}")
    # print(f"- mAP per class: {m_ap['map_per_class']}")
    # print(f"  - Class 1: {m_ap['map']}\n")

    return m_ap


def train():
    generate_train_val(train_set_dir=train_set_dir, country= "Norway")
    generate_train_val(train_set_dir=train_set_dir_czech, country= "Czech")
    generate_train_val(train_set_dir=train_set_dir_us, country= "United_States")
    generate_train_val(train_set_dir=train_set_dir_japan, country= "Japan")

    dataset = Dataset()
    dataloader_train = dataset.dataloader_train
    dataloader_val = dataset.dataloader_val

    model_name = 'resnet_ite2'
    losses = []
    start_epoch = 0

    model_obj = ResNet()

    # if (os.path.isfile(f"checkpoints/{model_name}.json")):
    #     losses, start_epoch = get_checkpoint(model_name=model_name)
    #     checkpoint = torch.load("checkpoints/" + model_name + ".pt")
    #     model_obj.model = torch.load("checkpoints/" + model_name + ".pt")
    #     # model_obj.model.load_state_dict(checkpoint['model_state'])
    #     print(f"Loaded {model_name}.pt from checkpoints!")
    
    model = model_obj.model  
    
    print(f"Model:\n{model}")
    model.to(get_device())

    scheduler = lr_scheduler.LinearLR(
        model_obj.optimizer, 
        start_factor=1.0, 
        end_factor=0.004, 
        total_iters=60
    )
    # scheduler = lr_scheduler.ExponentialLR(model_obj.optimizer, gamma=0.99)

    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)

    start_time = timer()
    total_time = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        start_epoch_time = time.time()
        results = train_epoch(
            model=model,
            dataloader=dataloader_train,
            optimizer=model_obj.optimizer,
            epoch=epoch
        )
        scheduler.step()
        losses += results
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time
        print(f"Learning rate: {model_obj.optimizer.param_groups[0]['lr']}")
        
        if epoch % 5 == 0:
            validate(
                model=model,
                dataloader=dataloader_val
            )

        if epoch % 10 == 0:
            torch.save({
                "model_state": model.state_dict(),
                'optimizer_state': model_obj.optimizer.state_dict(),
            }, "checkpoints/" + model_name + "epoch" + ".pt")

    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

    with open('losses.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(losses)

    details = {
        'current_epoch': epochs,
        'model_name': model_name
    }

    directory = os.getcwd()
    directory = directory + '/checkpoints/' + model_name + '.json'

    with open(directory, 'w') as convert_file:
        convert_file.write(json.dumps(details))

    torch.save({
        "model_state": model.state_dict(),
        'optimizer_state': model_obj.optimizer.state_dict(),
    }, "checkpoints/" + model_name + ".pt")
    print(f"Model saved and checkpointed!")


if __name__ == "__main__":
    train()