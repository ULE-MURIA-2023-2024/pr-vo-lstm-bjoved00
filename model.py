import torch
import torch.nn as nn
from torchvision.models import resnet50 as resnet
from torchvision.models import ResNet50_Weights as weights
from typing import Callable


class VisualOdometryModel(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool = False,
        lstm_dropout: float = 0.2
    ) -> None:

        super(VisualOdometryModel, self).__init__()

        # Load pre-trained ResNet model
        self.cnn_model = resnet(weights=weights.DEFAULT)
        resnet_output = list(self.cnn_model.children())[-1].in_features
        self.cnn_model.fc = nn.Identity()

        # Freeze the weights of the ResNet layers
        for param in self.cnn_model.parameters():
            param.requires_grad = False

        # TODO: create the LSTM

        # TODO: create the FC to generate the translation (3) and rotation (4)

    def resnet_transforms(self) -> Callable:
        return weights.DEFAULT.transforms(antialias=True)

    def forward(self, x: torch.TensorType) -> torch.TensorType:

        # CNN feature extraction
        batch_size, seq_length, channels, height, width = x.size()
        features = x.view(batch_size * seq_length, channels, height, width)

        with torch.no_grad():
            features = self.cnn_model(features)

        # TODO: use the LSTM

        # TODO: Get the output of the last time step
        return ...
