import enum
import torch.nn as nn
from transformers.activations import GELUActivation
from transformers.models.bert.modeling_bert import BertAttention


class LayerType(enum.Enum):
    NotRecognize = 0
    Activation = 1
    Linear = 2
    Dropout = 3
    Embedding = 4
    LayerNorm = 5
    Attention = 6

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
