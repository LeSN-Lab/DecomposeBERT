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
        self.layer_type = self.set_layer_type()
        self.active_node = None
        self.weight, self.bias = get_parameters(self)
        self.num_node = None
        self.shape = get_layer_shape(self)

        self.act_fcn = None         # if layer type is LayerType.Activation
        self.dropout_rate = None    # if layer type is LayerType.Dropout
        self.padding_idx = None     # if layer type is LayerType.Embedding


        if self.layer_type is not LayerType.Activation:
            if self.layer_type in [LayerType.Linear]:
                self.weight = nn.Parameter(layer.weight)
                if layer.bias is not None:
                    self.bias = nn.Parameter(layer.bias)
                else:
                    self.register_buffer('bias', torch.zeros(*self.shape[1:]))
            elif self.layer_type == LayerType.Embedding:
                self.weight = nn.Parameter(layer.weight)
                self.register_buffer('bias', None)
                self.padding_idx = layer.padding_idx
            elif self.layer_type == LayerType.LayerNorm:
                self.weight = nn.Parameter(layer.weight)
                self.bias = nn.Parameter(layer.bias)
                self.shape = layer.normalized_shape
            elif self.layer_type == LayerType.Dropout:
                self.weight = None
                self.dropout_rate = layer.p
                self.shape = None
        else:
            self.weight = None
            self.shape = None

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
    def set_layer_type(self):
        if isinstance(self, nn.Linear):
            return LayerType.Linear
        elif isinstance(self, nn.LayerNorm):
            return LayerType.LayerNorm
        elif isinstance(self, nn.Dropout):
            return LayerType.Dropout
        elif isinstance(self, nn.Embedding):
            return LayerType.Embedding
        elif isinstance(self, GELUActivation):
            self.act_fcn = ActivationType.GELU
            return LayerType.Activation
        elif isinstance(self, nn.Tanh):
            self.act_fcn = ActivationType.Tanh
            return LayerType.Activation
        else:
            return LayerType.NotRecognize

    def set_weight(self, weight):
        if hasattr(self.layer, 'weight'):
            assert self.weight.shape == weight.shape, f"Weight shapes mismatch: {self.weight.shape} vs {weight.shape}"
            self.weight.data = weight.data

    def set_bias(self, bias):
        assert self.bias.shape == bias.shape, f"Bias shapes mismatch: {self.bias.shape} vs {bias.shape}"
        self.bias.data = bias.data


    def get_weight(self):
        if self.weight is not None:
            return self.weight.detach().clone()
        else:
            return None

    def get_bias(self):
        if hasattr(self, 'bias'):
            return self.bias.detach().clone()
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

def get_parameters(layer):
    self.weight = nn.Parameter(layer.weight)


def get_layer_shape(layer):
    if self.layer_type is LayerType.Activation or self.layer_type is LayerType.Dropout:
        return None
    else:
        if hasattr(self, 'weight'):
            return self.weight.shape
        else:
            return self.out_features