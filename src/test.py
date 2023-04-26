import sys
# assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import torch
import os
import numpy as np
import tqdm
import json
from config.resnet import DatasetTest
from utils.torch_utils import to_cuda, get_device
from modelling import ResNet18, ResNet
from torchvision.ops import nms

# base_dir = "/Users/kristian/Documents/Trondheim/4. semester/TDT4265/TDT4265_project/data/road_damage"
base_dir = "/cluster/projects/itea_lille-idi-tdt4265/datasets/rdd2022/RDD2022"

test_set_dir = f"{base_dir}/Norway/test/images/"

def test():
    dataset = DatasetTest()
    dataloader = dataset.dataloader

    model_name = 'resnet_it2'
    model_obj = ResNet()

    checkpoint = torch.load("checkpoints/" + model_name + ".pt", map_location=get_device(), )
    model_obj.model.load_state_dict(checkpoint['model_state'], strict=False)
    print(f"Loaded {model_name}.pt from checkpoints!")

    image_ids = {}
    
    with open('image_ids.txt') as f:
        for line in f.readlines():
            idx = line.rsplit(' ', 1)[0]
            iid = line.rsplit(' ', 1)[1][:-1]
            image_ids[iid] = idx

    # print(f"Test id: {image_ids['Norway_010200.jpg']}")

    model = model_obj.model  

    model.to(get_device())
    model.eval()

    predictions = []
    box_counter = 1

    for data in tqdm.tqdm(dataloader, f"Running prediction on test set"):
        image, image_id = data
        image = to_cuda(image)
        prediction = model(image)
        for idx, pred in enumerate(prediction):
            image_iid = f"{image_id[idx]}.jpg" 
            # np_im = image[idx].detach().cpu().numpy()
            # im_height = np_im.shape[1]
            # im_width = np_im.shape[2]
        
            # print(f"Prediction of {image_iid}")            

            indices_t = nms(pred['boxes'], pred['scores'], 0.4)
            indices = indices_t.detach().cpu().numpy()
            
            boxes = pred['boxes'].detach().cpu().numpy()
            scores = pred['scores'].detach().cpu().numpy()
            labels = pred['labels'].detach().cpu().numpy()
            
            # boxes = np.take(boxes, indices, axis=0)
            # labels = np.take(labels, indices, axis=0)
            # scores = np.take(scores, indices, axis=0)

            # print(f"Idx: {idx} -> {indices}")
            for indice in indices:
                box = boxes[indice]
                xmin = float(box[0])#+float(box[2]))/2.0
                ymin = float(box[1])#+float(box[3]))/2.0
                width = (float(box[2])-float(box[0]))
                height = (float(box[3])-float(box[1]))
                item = {
                    "iscrowd": int(0),
                    "ignore": int(0),
                    "image_id": int(image_ids[image_iid]),
                    # "iid": str(image_id[idx].rsplit('_', 1)[1]),
                    "bbox": [xmin, ymin, width, height],
                    "category_id": int(labels[indice]-1),
                    "id": int(box_counter),
                    "score": float(scores[indice])
                }
                predictions.append(item)
                box_counter += 1

            # for indice, box in zip(indices, boxes):
            #     x = (float(box[0])+float(box[2]))/2.0
            #     y = (float(box[1])+float(box[3]))/2.0
            #     width = (float(box[2])-float(box[0]))
            #     height = (float(box[3])-float(box[1]))
            #     item = {
            #         "image_id": int(image_ids[image_iid]),
            # #         "iid": str(image_id[idx].rsplit('_', 1)[1]),
            #         "bbox": [x, y, width, height],
            #         "category_id": int(labels[indice]),
            #         "id": int(box_counter),
            #         "score": float(scores[indice])
            #     }
            #     predictions.append(item)
            #     box_counter += 1

    directory = os.getcwd()
    directory = directory + '/predictions.json'

    with open(directory, 'w') as convert_file:
        data = json.dumps(predictions, indent=2, ensure_ascii=False)
        convert_file.write(data)
        


if __name__ == "__main__":
    test()