import torchvision
import torch
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead, FasterRCNN
# from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2 


class ResNet():
    def __init__(self):
        """Resnet model

        Args:
            none (_type_): 
        """
        self.model = self.create_model()
        # self.model = fasterrcnn_resnet50_fpn_v2()

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        # define the optimizer
        self.optimizer = torch.optim.SGD(self.params, lr=0.01, momentum=0.9, weight_decay=0.0005)
        self.num_classes = 4 + 1

    def create_model(self):
        """
        Inspiration from: https://pytorch.org/vision/main/_modules/torchvision/models/detection/faster_rcnn.html
        
        Args:
            none (_type_): 
        
        """
        backbone = resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.DEFAULT, trainable_layers=5, norm_layer=torch.nn.BatchNorm2d)
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=3)
        box_head = FastRCNNConvFCHead(
            (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=torch.nn.BatchNorm2d
        )
        model = FasterRCNN(
            backbone,
            num_classes=5,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            box_head=box_head,
            box_nms_thresh=0.3,
            box_detections_per_img=10,
        )
        return model

class ResNet18():
    def __init__(self):
        """Resnet model

        Args:
            none (_type_): 
        """
        self.model = self.create_model()
        # self.model = fasterrcnn_resnet50_fpn_v2()

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        # define the optimizer
        self.optimizer = torch.optim.SGD(self.params, lr=0.011, momentum=0.9, weight_decay=0.0005)
        self.num_classes = 4 + 1

    def create_model(self):
        """
        Inspiration from: https://pytorch.org/vision/main/_modules/torchvision/models/detection/faster_rcnn.html
        
        Args:
            none (_type_): 
        
        """
        backbone = resnet_fpn_backbone('resnet18', weights=ResNet18_Weights.DEFAULT, trainable_layers=5, norm_layer=torch.nn.BatchNorm2d)
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
        box_head = FastRCNNConvFCHead(
            (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=torch.nn.BatchNorm2d
        )
        model = FasterRCNN(
            backbone,
            num_classes=5,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            box_head=box_head,
            box_nms_thresh=0.3,
            box_detections_per_img=10,
        )
        return model

# class ResNet():
#     def __init__(self):
#         """Resnet model

#         Args:
#             none (_type_): 
#         """
#         self.model = self.create_model()

#         self.params = [p for p in self.model.parameters() if p.requires_grad]
#         # define the optimizer
#         self.optimizer = torch.optim.SGD(self.params, lr=0.001, momentum=0.9, weight_decay=0.0005)
#         self.num_classes = 4 + 1

#     def create_model(self):
#         backbone = resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.DEFAULT, trainable_layers=5, norm_layer=torch.nn.BatchNorm2d)
#         anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
#         aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
#         rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
#         rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
#         box_head = FastRCNNConvFCHead(
#             (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=torch.nn.BatchNorm2d
#         )
#         model = FasterRCNN(
#             backbone,
#             num_classes=5,
#             rpn_anchor_generator=rpn_anchor_generator,
#             rpn_head=rpn_head,
#             box_head=box_head,
#             box_nms_thresh=0.3,
#             box_detections_per_img=10,
#         )
#         return model
