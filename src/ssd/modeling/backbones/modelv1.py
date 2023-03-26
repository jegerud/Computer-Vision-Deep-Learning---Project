import torch
from torch import nn
from typing import Tuple, List

class InitLayer(nn.Sequential):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            maxpool_kernel_size=2,
            max_pool_stride=2
            ):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size+2, stride=stride, padding=padding+1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=max_pool_stride),
            nn.Dropout(p=0.10),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=max_pool_stride),
            nn.Dropout(p=0.10),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )


class ConvolutionalLayer(nn.Sequential):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size=3,
            padding_one=1,
            padding_two=1,
            stride_one=1,
            stride_two=2
            ):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride_one, padding=padding_one),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride_two, padding=padding_two),
            nn.LeakyReLU(0.1),
        )


class ModelV1(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        self.model = nn.ModuleList()

        self.layer1 = InitLayer(image_channels, output_channels[0])
        self.model.append(self.layer1)

        self.layer2 = ConvolutionalLayer(output_channels[0], output_channels[1])
        self.model.append(self.layer2)

        self.layer3 = ConvolutionalLayer(output_channels[1], output_channels[2])
        self.model.append(self.layer3)

        self.layer4 = ConvolutionalLayer(output_channels[2], output_channels[3])
        self.model.append(self.layer4)

        self.layer5 = ConvolutionalLayer(output_channels[3], output_channels[4])
        self.model.append(self.layer5)

        self.layer6 = ConvolutionalLayer(output_channels[4], output_channels[5], stride_two=1, padding_two=0)
        self.model.append(self.layer6)

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        
        for layer in self.model:
            x = layer(x)
            out_features.append(x)
        
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

