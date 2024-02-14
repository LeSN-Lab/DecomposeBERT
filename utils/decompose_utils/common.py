import torch
import torch.nn as nn
import torch.functional as F
from utils.decompose_utils.constants import ArchitectureType, LayerType, ActivationType
from transformers.activations import GELUActivation



class ModularLayer(nn.Module):

    def __init__(self, layer, name):
        super(ModularLayer, self).__init__()
        self.name = name
        self.layer = layer
        self.active_node = None
        self.weight, self.bias = get_parameters(layer)
        self.num_node = None
        self.layer_type = None
        self.shape = None
        self.act_fcn = get_act_fcn_type(layer)

        self.padding_idx = None     # if layer type is LayerType.Embedding
        self.eps = None             # if layer type is LayerType.LayerNorm
        self.dropout_rate = None    # if layer type is LayerType.Dropout

        if self.act_fcn is ActivationType.Not:      # if layer is not activation fcn
            self.layer_type = get_layer_type(layer)
            self.shape = get_layer_shape(layer)
        else:                                       # if layer is activation fcn
            self.layer_type = LayerType.Activation
            self.shape = None

        if self.layer_type is not LayerType.NotRecognize:
            if hasattr(layer, 'weight'):
                self.weight = layer.weight
            else:
                self.weight = None

            if hasattr(layer, 'bias'):
                self.bias = layer.bias
            else:
                self.bias = None

            if self.layer_type is LayerType.Embedding:  # if layer type is LayerType.Embedding
                self.padding_idx = layer.padding_idx
            elif self.layer_type is LayerType.LayerNorm: # if layer type is LayerType.LayerNorm
                self.eps = layer.eps
            elif self.layer_type == LayerType.Dropout:  # if layer type is LayerType.Dropout
                self.dropout_rate = layer.p


    # def forward(self, x):
    #     x = self.layer(x)
    #     if self.act_fcn:
    #         act_fn_map = {'relu': nn.ReLU, 'sigmoid': torch.sigmoid, 'gelu': torch.gelu, 'tanh': torch.tanh}
    #         x = act_fn_map[self.act_fcn](x)
    #     elif self.layer_type == LayerType.Embedding:
    #         x = nn.Embedding(x, self.weight)
    # elif self.layer_type == LayerType.LayerNorm:
    # x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias)

    #     return x

    def zero_weight(self):
        if self.weight is not None:
            self.weight.data = torch.zeros_like(self.weight.data)

    def zero_bias(self):
        self.register_buffer('bias', torch.zeros(*self.shape[1:]))

    def get_weight(self):
        if self.weight is not None:
            return nn.Parameter(self.weight)
        else:
            return None

    def get_bias(self):
        if self.weight is not None:
            return nn.Parameter(self.bias)
        else:
            return None


class ModularLayerList(nn.ModuleList):
    def __init__(self, *args):
        super(ModularLayerList, self).__init__(*args)
        self.name = None
        self.num_layer = 0
        self.size = len(self)

    def append(self, module):
        """Append a ModularLayer to the list."""
        super().append(module)
        self.size = len(self)

    def forward(self, x):
        """Optional: Define a custom forward pass if needed."""
        for module in self:
            x = module(x)
        return x


def init_modular_layers(model):

    def recurse_layers(module, name=None):
        if len(list(module.named_children())) == 0:
            return ModularLayer(module, name)
        else:
            sub_layers = ModularLayerList()
            for child_name, child in module.named_children():
                sub_layer = recurse_layers(child, name=child_name)
                sub_layers.append(sub_layer)
            return sub_layers

    layers = recurse_layers(model)
    return layers


def get_architecture_type(model):
    if "berts" in model:
        return ArchitectureType.Bert
    elif "Transformer" in model:
        return ArchitectureType.Transformer
    elif "GPT" in model:
        return ArchitectureType.GPT


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


def get_act_fcn_type(layer):
    if isinstance(layer, GELUActivation):
        return ActivationType.GELU
    elif isinstance(layer, nn.Tanh):
        return ActivationType.Tanh
    else:
        return LayerType.NotRecognize


def get_parameters(layer):
    weight, bias = None, None
    if hasattr(layer, 'weight'):
        weight = layer.weight.detach() if layer.weight is not None else None
    if hasattr(layer, 'bias'):
        bias = layer.bias.detach() if layer.bias is not None else None
    return weight, bias


def get_layer_shape(layer):
    layer_type = get_layer_type(layer)
    if layer_type in [LayerType.Activation, LayerType.Dropout]:
        return None
    else:
        if hasattr(layer, 'weight'):
            return layer.weight.shape
        elif hasattr(layer, 'normalized_shape'):
            return layer.normalized_shape
        else:
            return layer.out_features   # [out_features, in_features]
