import torch
import torch.nn as nn
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_
import enum
from common import getLayerType

class LayerType(enum.Enum):
    Embedding = 1
    Linear = 2
    LayerNorm = 3
    Dropout = 4
    BertLayer = 5
    BertSelfAttention = 6
    BertSelfOutput = 7
    BertIntermediate = 8
    BertOutput = 9
    BertPooler = 10
    Other = 11


class ActivationType(enum.Enum):
    Tanh = 1
    Softmax = 2,
    GeLU = 3,
    ReLU = 4


class MLPlayers(nn.Module):
    type = None
    num_node = None
    activation = None
    is_first_layer = False
    is_last_layer = False
    number_samples = None
    number_features = None

    active_count = None
    inactive_count = None

    next_layer = None

    def __init__(self, name, layer):
        self.name = layer.name
        self.type = getLayerType()
        self.num_node = layer.output_shape[len(layer.output_shape) - 1]
        self.layers = nn.ModuleList(layers)


    def forward(self, input_feature):
        return self.mlp