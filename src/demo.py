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
from modelling import ResNet18
from torchvision.ops import nms

# base_dir = "/Users/kristian/Documents/Trondheim/4. semester/TDT4265/TDT4265_project/data/road_damage"
base_dir = "/cluster/projects/itea_lille-idi-tdt4265/datasets/rdd2022/RDD2022"

test_set_dir = f"{base_dir}/Norway/test/images/"

def demo():
    n_elements = 5
    dataset = DatasetTest()
    dataloader = dataset.dataloader

    model_name = 'resnet18_it2'
    model_obj = ResNet18()

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
    image_list.sort()
    
    # images = random.sample(image_list, n_elements)
    images = image_list[:1]
    
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

        indices_t = nms(detection['boxes'], detection['scores'], 0.4)
        indices = indices_t.detach().cpu().numpy()

        boxes = boxes.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()

        boxes = np.take(boxes, indices, axis=0)
        labels = np.take(labels, indices, axis=0)
        scores = np.take(scores, indices, axis=0)

        drawn_image = draw_boxes(
            orig_img, boxes, labels, scores, width=10, alpha=2,
        ).astype(np.uint8)
        
        im = Image.fromarray(drawn_image)
        # im.save(f"demo/{iid}.png")
        im.save(f"demo/demo.png")


if __name__ == "__main__":
    demo()