import os
import numpy as np
from configs.yolov5 import train_set_dir
from os import listdir
from os.path import isfile, join


def generate_train_val(cfg):
    directory = os.path.join("ssd/data/utils", "splits")
    file_dir = os.path.join("ssd/data/utils/splits", "split.txt")
    if os.path.isfile(file_dir):
        return
    print(f"Creating split file in directory: {file_dir}")

    image_sets_file = os.path.join("data", train_set_dir, "images")
    ids = read_image_ids(image_sets_file=image_sets_file)
    split = cfg.data_train.train_val_split
    
    np.random.shuffle(ids)
    split_nr = int(len(ids)*(1-split))
    train = ids[:split_nr-1]
    val = ids[split_nr:]

    isExist = os.path.exists(directory)
    if not isExist:
        os.makedirs(directory)
   
    f = open(f"{directory}/split.txt", "w")
    for t in train:
        f.write(f"{t} 1 \n") 
    for v in val:
        f.write(f"{v} -1 \n")
    f.close()
    
    
def read_image_ids(image_sets_file):
    ids = []
    images = [f for f in listdir(image_sets_file) if isfile(join(image_sets_file, f))]
    for image in images:
        iid = image.rsplit('.', 1)[0]
        if len(iid) > 0: 
            ids.append(iid)
    return ids