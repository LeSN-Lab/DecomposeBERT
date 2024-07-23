import torch
import torch.nn as nn
from scipy.stats import norm
from torch.nn.functional import threshold

from utils.helper import safe_std


class LayerWrapper:
    """
    This class is a wrapper for transformer layer.
    """

    def __init__(self, name, layer):
        self.name = name
        self.layer = layer
        self.shape = layer.weight.shape
        self.inputs = []
        self.outputs = []

    def update(self, input, output):
        self.inputs.append(input[0].cpu())
        self.outputs.append(output.cpu())

    def update_batch(self):
        self.inputs = torch.cat(self.inputs, dim=0)
        self.outputs = torch.cat(self.outputs, dim=0)


def find_layers(
    model, layer_types=None, include_layers=None, exclude_layers=None, prefix=""
):
    if layer_types is None:
        layer_types = [nn.Linear]
    if include_layers is None:
        include_layers = []
    if exclude_layers is None:
        exclude_layers = []
    layers_dict = {}

    def recursive_find(module, prefix):
        for name, layer in module.named_children():
            layer_name = f"{prefix}.{name}" if prefix else name
            if any(exclude in layer_name for exclude in exclude_layers):
                continue
            if include_layers and not any(
                include in layer_name for include in include_layers
            ):
                if not any(isinstance(layer, t) for t in layer_types):
                    recursive_find(layer, layer_name)
                continue
            if isinstance(layer, tuple(layer_types)):
                layers_dict[layer_name] = layer
            else:
                recursive_find(layer, layer_name)

    recursive_find(model, prefix)

    return layers_dict


def prune_magnitude(
    model, sparsity_ratio=0.4, include_layers=None, exclude_layers=None
):
    layers = find_layers(
        model, include_layers=include_layers, exclude_layers=exclude_layers
    )
    for _, layer in layers.items():
        current_weight = layer.weight.data
        threshold = torch.sort(torch.abs(current_weight).flatten())[0][
            int(current_weight.numel() * sparsity_ratio)
        ]
        mask = torch.abs(current_weight) < threshold
        layer.weight.data[mask] = 0


def prune_norm_distribution(
    model, sparsity_ratio=0.4, include_layers=None, exclude_layers=None
):
    layers = find_layers(
        model, include_layers=include_layers, exclude_layers=exclude_layers
    )
    for _, layer in layers.items():
        current_weight = layer.weight.data
        mean = torch.mean(current_weight, dim=1, keepdim=True)
        std = torch.std(current_weight, dim=1, keepdim=True)
        z_scores = (current_weight - mean) / std

        lower_z, upper_z = norm.ppf(0.5 - sparsity_ratio / 2), norm.ppf(
            0.5 + sparsity_ratio / 2
        )
        mask = torch.logical_and(z_scores >= lower_z, z_scores < upper_z)

        # Calculate the actual sparsity achieved
        actual_sparsity = 1.0 - torch.sum(mask).item() / mask.numel()

        # Adjust if the actual sparsity exceeds the target sparsity_ratio
        if actual_sparsity > sparsity_ratio:
            target_sparsity = 1.0 - sparsity_ratio
            num_weights_to_keep = int(target_sparsity * mask.numel())
            flattened_weights = torch.abs(current_weight).flatten()
            threshold_value = torch.topk(
                flattened_weights, num_weights_to_keep, largest=True
            )[0][-1]
            mask = torch.abs(current_weight) < threshold_value

        layer.weight.data[mask] = 0


def prune_concern_identification(
    model,
    module,
    dataloader,
    sparsity_ratio=0.4,
    p=1,
    include_layers=None,
    exclude_layers=None,
):
    ref_layers = find_layers(
        model, include_layers=include_layers, exclude_layers=exclude_layers
    )
    target_layers = find_layers(
        module, include_layers=include_layers, exclude_layers=exclude_layers
    )
    device = next(model.parameters()).device

    wrappers = {}
    ref_handle_list = []
    target_handle_list = []

    def get_hook(wrapper):
        def hook(module, input, output):
            wrapper.update(input, output)

        return hook

    for (ref_name, ref_layer), (target_name, target_layer) in zip(
        ref_layers.items(), target_layers.items()
    ):
        ref_wrapper = LayerWrapper(ref_name, ref_layer)
        target_wrapper = LayerWrapper(target_name, target_layer)

        wrappers[ref_name] = {"ref": ref_wrapper, "target": target_wrapper}

        ref_handle = ref_layer.register_forward_hook(get_hook(ref_wrapper))
        target_handle = target_layer.register_forward_hook(get_hook(target_wrapper))

        ref_handle_list.append(ref_handle)
        target_handle_list.append(target_handle)

    for batch in dataloader:
        input_ids, attn_mask, _, _ = batch
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        with torch.no_grad():
            model(input_ids, attention_mask=attn_mask)
            module(input_ids, attention_mask=attn_mask)

    for name, wrapper_pair in wrappers.items():
        wrapper_pair["ref"].update_batch()
        wrapper_pair["target"].update_batch()

        ref_outputs = wrapper_pair["ref"].outputs
        target_outputs = wrapper_pair["target"].outputs

        original_weight = wrapper_pair["ref"].layer.weight.data
        current_weight = wrapper_pair["target"].layer.weight.data
        shape = current_weight.shape

        output_loss = target_outputs - ref_outputs

        sign = torch.sign(output_loss.mean(dim=1))

        if p == 'mean':
            output_loss = torch.mean(output_loss, dim=1)
        elif p == 1:
            output_loss = torch.norm(output_loss, p=1, dim=1) * sign
        elif p == 2:
            output_loss = torch.norm(output_loss, p=2, dim=1) * sign
        elif p == "inf":
            output_loss = torch.norm(output_loss, p=float('inf'), dim=1) * sign
        elif p == "cosine":
            pass
        else:
            raise ValueError("Unsupported norm type")

        weight_score = torch.mean(output_loss, dim=0).reshape(-1, 1).to(device)
        importance_score = current_weight * weight_score
        for i in range(current_weight.size(0)):
            abs_weights = torch.abs(current_weight[i])
            num_weights_to_prune = int(sparsity_ratio * abs_weights.numel())

            if num_weights_to_prune > 0:
                sorted_indices = torch.argsort(abs_weights)
                prune_indices = sorted_indices[:num_weights_to_prune]
                current_weight[i].view(-1)[prune_indices] = 0


    for handle in ref_handle_list + target_handle_list:
        handle.remove()


def recover_tangling_identification(
    model, sparsity_ratio=0.4, include_layers=None, exclude_layers=None
):
    pass


def prune_wanda(
    model, dataloader, sparsity_ratio=0.4, include_layers=None, exclude_layers=None
):
    layers = find_layers(
        model, include_layers=include_layers, exclude_layers=exclude_layers
    )
    device = next(model.parameters()).device

    wrappers = {}
    handle_list = []

    def get_hook(wrapper):
        def hook(module, input, output):
            wrapper.update(input, output)

        return hook

    for name, layer in layers.items():
        wrapper = LayerWrapper(name, layer)
        wrappers[name] = wrapper
        handle = layer.register_forward_hook(get_hook(wrapper))
        handle_list.append(handle)

    for batch in dataloader:
        input_ids, attn_mask, _, _ = batch
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            model(input_ids, attention_mask=attn_mask)

        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            for name, wrapper in wrappers.items():
                W = wrapper.layer.weight.data
                X = wrapper.inputs[0][i]
                metric = X.norm(p=2, dim=0) * torch.abs(W)

                sorted_metric, sorted_idx = torch.sort(metric, dim=1)

                # Pruning indices
                pruned_idx = sorted_idx[:, : int(W.shape[1] * sparsity_ratio)]

                # Creating mask and applying it
                W.scatter_(
                    dim=1,
                    index=pruned_idx,
                    src=torch.zeros_like(pruned_idx, dtype=W.dtype),
                )

    for handle in handle_list:
        handle.remove()
