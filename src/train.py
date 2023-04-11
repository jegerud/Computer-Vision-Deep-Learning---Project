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
# from tqdm import tqdm
import tqdm
from torchvision import models
from data_utils.utils.generate_train_val import generate_train_val
from config.resnet import Dataset, num_classes, gpu_transform, epochs
from utils.torch_utils import to_cuda, move_to
from config.utils import get_dataset_dir
from modelling import ResNet
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

    
def train_epoch(
        model, 
        dataloader,
        optimizer,
        epoch
    ):
    model.train()
    
    total_train_loss = []
    # it = 0
    for img, target in tqdm.tqdm(dataloader, f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        
        img = move_to(img)
        target = move_to(target)
        # batch = gpu_transform(batch)

        with torch.set_grad_enabled(True):
            output = model(img, target)
            
            loss = sum(loss for loss in output.values())
            total_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            metric = MeanAveragePrecision()
            metric.update(output, target)
    
        # if it % 100 == 0:
        #     print(f"\tIteration #{it} loss: {loss}")
        
        # it += 1

    train_loss = sum(total_train_loss) / len(dataloader)
    print(f"Epoch: {epoch+1} | train_loss: {train_loss} | total_train_loss: {sum(total_train_loss)}")# | test_loss: {test_loss}")
    pprint(metric.compute())

    return total_train_loss


def train():
    generate_train_val()

    dataset = Dataset()
    # dataset_train = dataset.dataset_train
    # dataset_val = dataset.dataset_val
    dataloader_train = dataset.dataloader_train
    # dataloader_val = dataset.dataloader_val

    model_obj = ResNet()
    model = model_obj.model

    num_features = model.roi_heads.box_predictor.bbox_pred.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(num_features, num_classes)
    print(f"Model:\n{model}")
    
    model = to_cuda(model)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    start_time = timer()
    total_time = 0

    losses = []

    for epoch in range(0, epochs):
        start_epoch_time = time.time()
        results = train_epoch(
            model=model,
            dataloader=dataloader_train,
            optimizer=model_obj.optimizer,
            epoch=epoch
        )
        losses += results
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

    # print(f"losses: {losses}")

    with open('losses.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(losses)

    model_name = 'resnet_it1.pt'
    details = {
        'current_epoch': epochs,
        'model_name': model_name
    }

    directory = os.getcwd()
    directory = directory + '/checkpoints/checkpoint.json'

    with open(directory, 'w') as convert_file:
        convert_file.write(json.dumps(details))

    torch.save(model, model_name)

if __name__ == "__main__":
    train()