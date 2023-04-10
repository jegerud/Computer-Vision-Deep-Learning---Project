import sys
# assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import functools
import time
from timeit import default_timer as timer 
import click
import torch
import torchvision
# from tqdm import tqdm
import tqdm
from torchvision import models
from data_utils.utils.generate_train_val import generate_train_val
from config.resnet import Dataset, num_classes, gpu_transform, epochs
from utils.torch_utils import to_cuda
from modelling import ResNet
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

    
def train_epoch(
        model, 
        dataloader,
        optimizer,
        epoch
    ):
    # losses = {
    #     "train_loss": [],
    #     "test_loss": [],
    # }
    model.train()

    test_loss = 0
    train_loss= 0

    for X, y in tqdm.tqdm(dataloader, f"Epoch {epoch}"):
        optimizer.zero_grad()
        # batch = to_cuda(batch)
        # print(f"X: {X}")
        # print(f"y: {y}")
        X = to_cuda(X)
        # y = to_cuda(y)
        # batch = gpu_transform(batch)
        print("Before model(X, y)")

        with torch.set_grad_enabled(True):
            output = model(X, y)
            
            loss = sum(loss for loss in output.values())
            train_loss += loss.item()
            # loss.backward()
            # optimizer.step()
        
        # if batch % 200 == 0:
        #     print(f"Iteration #{batch} loss: {loss}")

        # Print out what's happening
        print(f"Epoch: {epoch+1} | train_loss: {train_loss} | test_loss: {test_loss}")

    #     losses["train_loss"].append(train_loss)
    #     # losses["test_loss"].append(test_loss)
    train_loss = train_loss / len(dataloader)

    return train_loss


def train():
    generate_train_val()

    dataset = Dataset()
    dataset_train = dataset.dataset_train
    dataset_val = dataset.dataset_val
    dataloader_train = dataset.dataloader_train
    dataloader_val = dataset.dataloader_val

    # print(f"Dataloader train: {dataloader_train}")
    # print(f"Len dataloader training: {len(dataloader_train)}")

    model_obj = ResNet()
    model = model_obj.model

    # print(type(model))

    num_features = model.roi_heads.box_predictor.bbox_pred.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(num_features, num_classes)
    print(f"Model:\n{model}")
    
    model = to_cuda(model)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # start_time = timer()
    for epoch in range(0, epochs):
        start_epoch_time = time.time()
        results = train_epoch(
            model=model,
            dataloader=dataloader_train,
            optimizer=model_obj.optimizer,
            epoch=epoch
        )
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

    # end_time = timer()
    # print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
    # End the timer and print out how long it took

if __name__ == "__main__":
    train()