import sys
# assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import torch
import os
import numpy as np
import tqdm
import cv2
import random
from vizer.draw import draw_boxes
from PIL import Image
from config.resnet import DatasetTest, test_cpu_transform
from utils.torch_utils import to_cuda, get_device
from modelling import ResNet, ResNet18
from utils import get_checkpoint
from torchvision.ops import nms

def video():
    model_name = 'resnet18_it2'
    model_obj = ResNet18()

    # if (os.path.isfile(f"checkpoints/{model_name}.json")):
    #     losses, start_epoch = get_checkpoint(model_name=model_name)
    #     # checkpoint = torch.load("checkpoints/" + model_name + ".pt")
    #     # model_obj.model = torch.load("checkpoints/" + model_name + ".pt")
    checkpoint = torch.load("checkpoints/" + model_name + ".pt", map_location=get_device())
    print(f"Checkpoint loaded!")

    model_obj.model.load_state_dict(checkpoint['model_state'])
    print(f"Loaded {model_name}.pt from checkpoints!")

    model = model_obj.model  
    
    # print(f"Model:\n{model}")
    model.to(get_device())
    model.eval()

    cap = cv2.VideoCapture('video2.mp4')
  
    if (cap.isOpened() == False):
        print("Error opening video file")
        exit()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    size = (frame_width, frame_height)
    
    detections = []

    print(f"Starting video stream!")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            orig_img = Image.fromarray(frame)

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

            detections.append({
                'boxes': boxes,
                'labels': labels,
                'scores': scores
            })
    
        else:
            break
    # print(f"Detections: {detections}")
    cap.release()

    print(f"Starting to write on video!")
    # result = cv2.VideoWriter('filename.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, size)
    result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

    idx = 0

    # fps = FPS().start()
    video = cv2.VideoCapture('video2.mp4')
    if (video.isOpened() == False):
        print("Error opening video file")
        exit()

    frame_idx = 0
    while frame_idx < len(detections):
        ret, frame = video.read()
        print(f"{frame_idx} / {len(detections)}")

        if ret == True: 
            orig = frame.copy()
            orig_img = Image.fromarray(orig)
            
            detection = detections[frame_idx]

            boxes = detection["boxes"]
            labels = detection["labels"]
            scores = detection["scores"]

            drawn_image = draw_boxes(
                orig_img, boxes, labels, scores, width=10, alpha=2,
            ).astype(np.uint8)
            
            result.write(drawn_image)
        
        else:
            break
        
        frame_idx += 1
    
    result.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    video()