import torch.nn as nn
from utils.model_utils.custom_layers import ModularLayer, ModularLayerList


def init_modular_layers(model):

    def _recurse_layers(module, name=None):
        if len(list(module.named_children())) == 0:
            return ModularLayer(module, name)
        else:
            sub_layers = ModularLayerList()
            sub_layers.name = name
            for child_name, child in module.named_children():
                sub_layer = _recurse_layers(child, name=child_name)
                sub_layers.append(sub_layer)
            return sub_layers

    layers = _recurse_layers(model)
    return layers


def recurse_layers(module, func):  # dfs
    func(module)
    if not isinstance(module, ModularLayer):
        for i, layer in enumerate(module):
            recurse_layers(layer, func)








def get_parameters(layer):
    weight, bias = None, None
    if hasattr(layer, "weight"):
        weight = layer.weight.detach() if layer.weight is not None else None
    if hasattr(layer, "bias"):
        bias = layer.bias.detach() if layer.bias is not None else None
    return weight, bias





