import torch
import torch.nn as nn
from tqdm import tqdm


class LayerWrapper:
    """
    This class is a wrapper for transformer layer.
    """

    def __init__(self, name, layer, batch_size, input, output):
        self.name = name
        self.layer = layer
        self.shape = layer.weight.shape
        self.batch_size = batch_size
        self.input = input
        self.output = output

    def update(self, fcn):
        fcn(self)


def find_layers(model, layer_types=None, prefix=""):
    if layer_types is None:
        layer_types = [nn.Linear]
    layers_dict = {}

    def recursive_find(module, prefix):
        for name, layer in module.named_children():
            layer_name = f"{prefix}.{name}" if prefix else name
            if isinstance(layer, tuple(layer_types)):
                layers_dict[layer_name] = layer
            else:
                recursive_find(layer, layer_name)

    recursive_find(model, prefix)

    return layers_dict


def prune_magnitude(model, sparsity_ratio=0.4):
    layers = find_layers(model)
    for _, layer in layers.items():
        current_weight = layer.weight.data
        threshold = torch.sort(torch.abs(current_weight).flatten())[0][
            int(current_weight.numel() * sparsity_ratio)
        ]
        mask = torch.abs(current_weight) > threshold
        layer.weight.data.mul_(mask)


def prune_wanda(model, dataloader, model_config, sparsity_ratio=0.4):
    layers = find_layers(model)
    wrappers = {}

    def hook(name):
        def hook_fn(module, input, output):
            batch_size = input[0].size(0)
            wrapper = LayerWrapper(
                name, module, batch_size, input[0].detach(), output.detach()
            )
            wrappers[name] = wrapper
        return hook_fn

    def wanda(wrapper, sparsity_ratio):
        W = wrapper.layer.weight.data
        X = wrapper.input
        s = sparsity_ratio

        # Ensure X and W are compatible
        X_flat = X.view(-1, X.shape[-1])  # Flatten the input to (N * L, C_in)

        if X_flat.shape[1] != W.shape[1]:
            raise ValueError(f"Input dimension {X_flat.shape[1]} does not match weight dimension {W.shape[1]}")

        # Metric calculation as per the pseudocode
        metric = W.abs() * X_flat.norm(p=2, dim=0)

        # Sorting the metric
        _, sorted_idx = torch.sort(metric, dim=1)

        # Pruning indices
        pruned_idx = sorted_idx[:, :int(W.shape[1] * s)]

        # Applying the mask
        mask = torch.ones_like(W, dtype=torch.bool)
        mask.scatter_(dim=1, index=pruned_idx, value=False)
        W *= mask.float()
        wrapper.layer.weight.data = W

    for batch in dataloader:
        input_ids, attn_mask, _, total_sampled = batch
        input_ids = input_ids.to(model_config.device)
        attn_mask = attn_mask.to(model_config.device)

        handle_list = []
        for name, layer in layers.items():
            handle = layer.register_forward_hook(hook(name))
            handle_list.append(handle)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attn_mask)

        for _, wrapper in wrappers.items():
            wrapper.update(lambda w: wanda(w, sparsity_ratio))

        for handle in handle_list:
            handle.remove()
