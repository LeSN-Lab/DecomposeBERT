import torch
from utils.decompose_utils.common import *


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

        self.padding_idx = None  # if layer type is LayerType.Embedding
        self.eps = None  # if layer type is LayerType.LayerNorm
        self.dropout_rate = None  # if layer type is LayerType.Dropout

        # if layer type is LayerType.Attention
        self.num_attention_heads = None
        self.attention_head_size = None
        self.all_head_size = None

        self.query = None
        self.key = None
        self.value = None

        if self.act_fcn is ActivationType.Linear:  # if layer is not activation fcn
            self.layer_type = get_layer_type(layer)
            self.shape = get_layer_shape(layer)
        else:  # if layer is activation fcn
            self.layer_type = LayerType.Activation
            self.shape = None

        if self.layer_type is not LayerType.NotRecognize:
            self.weight = getattr(layer, "weight", None)
            self.bias = getattr(layer, "bias", None)

            self.padding_idx = getattr(layer, "padding_idx", None)
            self.eps = getattr(layer, "eps", None)
            self.dropout_rate = getattr(layer, "p", None)

    def forward(self, x):
        out = self.layer(x)

        if self.act_fcn:  # if layer type is activation fcn
            act_fn_map = {"relu": nn.ReLU, "gelu": GELUActivation, "tanh": torch.tanh}
            out = act_fn_map[self.act_fcn](out) if self.act_fcn in act_fn_map else out
        return out

    def zero_weight(self):
        if self.weight is not None:
            nn.init.constant_(self.weight, 0)

    def zero_bias(self):
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)


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
