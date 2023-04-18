import sys
# assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import torch
import os
import numpy as np
import tqdm
import random
from vizer.draw import draw_boxes
from PIL import Image
from config.resnet import DatasetTest, test_cpu_transform
from utils.torch_utils import to_cuda, get_device
from modelling import ResNet

base_dir = "/Users/kristian/Documents/Trondheim/4. semester/TDT4265/TDT4265_project/data/road_damage"
# base_dir = "/cluster/projects/itea_lille-idi-tdt4265/datasets/rdd2022/RDD2022"

test_set_dir = f"{base_dir}/Norway/test/images/"

def test():
    n_elements = 5
    dataset = DatasetTest()
    dataloader = dataset.dataloader

    model_name = 'resnet_it1epoch'
    model_obj = ResNet()

    print(f"Loading model!")
    checkpoint = torch.load("checkpoints/" + model_name + ".pt", map_location=get_device())
    print(f"Checkpoint loaded!")

    model_obj.model.load_state_dict(checkpoint['model_state'])
    print(f"Loaded {model_name}.pt from checkpoints!")

    model = model_obj.model  
    
    # print(f"Model:\n{model}")
    model.to(get_device())
    model.eval()

    image_list = os.listdir(test_set_dir)

    # images = image_list[:3]
    images = random.sample(image_list, n_elements)
    
    for image_id in images:
        iid = image_id.rsplit('.', 1)[0]
        image_path = test_set_dir + image_id

        orig_img = Image.open(image_path).convert("RGB")

        img = test_cpu_transform(orig_img).unsqueeze_(0)
        img = to_cuda(img)

        detection = model(img)[0]    
        
        boxes = detection['boxes']
        scores = detection['scores']
        labels = detection['labels']

        boxes = boxes.detach().numpy()
        labels = labels.detach().numpy()
        scores = scores.detach().numpy()

        drawn_image = draw_boxes(
            orig_img, boxes, labels, scores, width=10, alpha=2,
        ).astype(np.uint8)
        
        im = Image.fromarray(drawn_image)
        im.save(f"demo/{iid}.png")


if __name__ == "__main__":
    test()