import torchvision
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
# from config.resnet import num_classes

class ResNet():
    def __init__(self):
        """Resnet model

        Args:
            none (_type_): 
        """
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, trainable_backbone_layers=3)
        self.num_classes = 5

        self.params_to_update = self.model.parameters()
        self.learning_rate = 0.001
        self.momentum = 0.9

        self.optimizer = torch.optim.SGD(
            self.params_to_update, 
            lr=self.learning_rate, 
            momentum=self.momentum, 
            weight_decay=0.0005
        )
