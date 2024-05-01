import enum
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import GELUActivation

class LayerType(enum.Enum):
    NotRecognize = 0
    Activation = 1
    Linear = 2
    Dropout = 3
    Embedding = 4
    LayerNorm = 5

    @staticmethod
    def get_layer_type(layer):
        if isinstance(layer, nn.Linear):
            return LayerType.Linear
        elif isinstance(layer, nn.LayerNorm):
            return LayerType.LayerNorm
        elif isinstance(layer, nn.Dropout):
            return LayerType.Dropout
        elif isinstance(layer, nn.Embedding):
            return LayerType.Embedding
        else:
            return LayerType.NotRecognize

    @staticmethod
    def get_layer_shape(layer):
        layer_type = LayerType.get_layer_type(layer)
        if layer_type in [LayerType.Activation, LayerType.Dropout]:
            return None
        else:
            if hasattr(layer, "weight"):
                return layer.weight.shape
            elif hasattr(layer, "normalized_shape"):
                return layer.normalized_shape
            else:
                return layer.out_features  # [out_features, in_features]

    @staticmethod
    def get_layer_name(layer):
        if isinstance(layer, nn.Linear):
            return "Linear"
        elif isinstance(layer, nn.LayerNorm):
            return "LayerNorm"
        elif isinstance(layer, nn.Dropout):
            return "Dropout"
        elif isinstance(layer, nn.Embedding):
            return "Embedding"
        else:
            raise "Not recognized type"


class ActivationType(enum.Enum):
    # Non-linear Activations
    Linear = 0  # No activations function
    ReLU = 1
    Tanh = 2
    GELU = 3
    Softmax = 4

    @staticmethod
    def get_act_fn_type(layer):
        if isinstance(layer, GELUActivation):
            return ActivationType.GELU
        elif isinstance(layer, nn.Tanh):
            return ActivationType.Tanh
        else:
            return ActivationType.Linear

    @staticmethod
    def get_act_fn(act_fn_type, input_tensor):
        if act_fn_type == ActivationType.Linear:
            return input_tensor
        elif act_fn_type == ActivationType.ReLU:
            return F.relu(input_tensor)
        elif act_fn_type == ActivationType.Tanh:
            return F.tanh(input_tensor)
        elif act_fn_type == ActivationType.GELU:
            return F.gelu(input_tensor)
        elif act_fn_type == ActivationType.Softmax:
            return F.softmax(input_tensor)