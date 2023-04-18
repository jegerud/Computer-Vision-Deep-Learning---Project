import torch
import torchvision
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN

class MobileNet():
    def __init__(self):
        """MobileNet model

        Args:
            none (_type_): 
        """
        self.model = self.create_model()

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        # define the optimizer
        self.optimizer = torch.optim.SGD(self.params, lr=0.002, momentum=0.9, weight_decay=0.0005)
        self.num_classes = 4 + 1

    def create_model(self):
        rpn_anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),))
        backbone = mobilenet_backbone(
                backbone_name='mobilenet_v3_large', 
                weights=MobileNet_V3_Large_Weights, 
                fpn=True,
                trainable_layers=4
            )
        
        model = FasterRCNN(
                backbone=backbone,
                num_classes=5,
                rpn_anchor_generator=rpn_anchor_generator,
                box_nms_thresh=0.3,
                box_detections_per_img=10,
            )
        return model