import os
from os import listdir
from os.path import isfile, join
import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data._utils.collate import default_collate
from PIL import Image
import cv2
from pycocotools.coco import COCO
from utils import utils
from utils.box_utils import bbox_ltrb_to_ltwh

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

class RoadDamageDataset(torch.utils.data.Dataset):
    class_names = ('__background__', 'D00', 'D10', 'D20', 'D40')

    def __init__(self, data_dir, split, country, remove_empty, transform=None, keep_difficult=False):
        """Dataset for road damage data.
        Args:
            data_dir: the root of the road damage dataset, the directory is split into test and 
            train directory:
                
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        image_sets_file = os.path.join("data_utils/utils/splits", country, "split.txt")
        self.image_ids = RoadDamageDataset._read_image_ids(image_sets_file, self.split)
        self.keep_difficult = keep_difficult
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        if remove_empty:
            self.image_ids = [id_ for id_ in self.image_ids if len(self._get_annotation(id_)[0]) > 0]
    
    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.data_dir, "images", "%s.jpg" % image_id)
        image_resized = Image.open(image_path).convert("RGB")
        
        boxes, labels, is_difficult, im_info = self._get_annotation(image_id)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1,4)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor(idx)

        # apply the image transforms
        if self.transform:
            image_resized = self.transform(image_resized)
            target['boxes'] = torch.Tensor(target['boxes']).reshape(-1,4)
       
        return image_resized, target

    def __len__(self):
        return len(self.image_ids)

    def batch_collate(self, batch):
        return tuple(zip(*batch))
                            
    @staticmethod
    def _read_image_ids(image_sets_file, split):
        ids = []
        cat_split = 1 if split == "train" else -1
        f = open(image_sets_file, "r")
        for x in f:
            line = x.rsplit(' ', 1)[0]
            iid, cat = line.rsplit(' ', 1)
            if (int(cat) == int(cat_split)):
                ids.append(iid)

        return ids

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, "annotations", "xmls", "%s.xml" % image_id)
        ann_file = ET.parse(annotation_file)
        objects = ann_file.findall("object")

        size = ann_file.getroot().find("size")
        im_info = dict(
            height=int(size.find("height").text),
            width=int(size.find("width").text)
            )
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.strip()
            if (class_name != 'D10' and class_name != 'D20' and class_name != 'D00' and class_name != 'D40'):
                continue
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = 0
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return boxes, labels, is_difficult, im_info
    
    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "images", "%s.jpg" % image_id)
        # image = Image.open(image_file).convert("RGB")
        
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = image
        image_resized /= 255.0

        image = np.array(image_resized)
        return image

    def get_annotations_as_coco(self) -> COCO:
        """
            Returns bounding box annotations in COCO dataset format
        """
        classes = [
            '__background__', 'D00', 'D10', 'D20', 'D40'
        ]
        coco_anns = {"annotations" : [], "images" : [], "licences" : [{"name": "", "id": 0, "url": ""}], "categories" : []}
        coco_anns["categories"] = [
            {"name": cat, "id": i+1, "supercategory": ""}
            for i, cat in enumerate(classes) 
        ]
        ann_id = 1
        for idx in range(len(self)):
            # image_id = idx
            if (idx % 10 == 0):
                print(f"{idx} / {len(self)}")
            image, target = self.__getitem__(idx)
            boxes_ltrb  = target['boxes']
            boxes_ltwh = bbox_ltrb_to_ltwh(boxes_ltrb)
            height, width = image.shape[:2]
            coco_anns["images"].append({"id": int(target['image_id']), "height": height, "width": width })
            for box, label in zip(boxes_ltwh, target['labels']):
                box = box.tolist()
                area = box[-1] * box[-2]
                coco_anns["annotations"].append({
                    "bbox": box, "area": area, "category_id": int(label),
                    "image_id": int(target['image_id']), "id": ann_id, "iscrowd": 0, "segmentation": []}
                )
                ann_id += 1
        coco_anns["annotations"].sort(key=lambda x: x["image_id"])
        coco_anns["images"].sort(key=lambda x: x["id"])
        coco = COCO()
        coco.dataset = coco_anns
        coco.createIndex()
        coco.getAnnIds()
        return coco


class RoadDamageTestDataset(torch.utils.data.Dataset):
    class_names = ('__background__', 'D00', 'D10', 'D20', 'D40')

    def __init__(self, data_dir, country, remove_empty, transform=None, keep_difficult=False):
        """Dataset for road damage data.
        Args:
            data_dir: the root of the road damage dataset, the directory is split into test and 
            train directory:
                
        """
        self.data_dir = data_dir
        self.transform = transform
        image_sets_file = os.path.join(self.data_dir, "images")
        self.image_ids = self.read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.data_dir, "images", "%s.jpg" % image_id)
        image_resized = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image_resized = self.transform(image_resized)
       
        return image_resized, image_id

    def __len__(self):
        return len(self.image_ids)

    def batch_collate_test(self, batch):
        return tuple(zip(*batch))

    def read_image_ids(self, image_sets_file):
        ids = []
        f = os.listdir(image_sets_file)
        for x in f:
            iid = x.rsplit('.', 1)[0]
            ids.append(iid)

        ids.sort()
        return ids

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "images", "%s.jpg" % image_id)
        # image = Image.open(image_file).convert("RGB")
        
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = image
        image_resized /= 255.0

        image = np.array(image_resized)
        return image
    