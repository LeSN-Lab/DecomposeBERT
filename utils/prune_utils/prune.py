from select import select

import torch
import torch.nn as nn
from scipy.stats import norm
from typing import *
from torch import Tensor
from torch.nn import Module

from utils.dataset_utils.sampling import SamplingDataset
import gc


class LayerWrapper:
    """
    This class is a wrapper for transformer layer.
    """

    def __init__(self, name: str, layer: Module):
        self.name: str = name
        self.layer: Module = layer
        self.shape: torch.Size = layer.weight.shape
        self.inputs: Union[List[Tensor], Tensor] = []
        self.outputs: Union[List[Tensor], Tensor] = []
        self.device = layer.weight.device
        self.scaler_row: Tensor = torch.zeros(self.shape[1], device=self.device)
        self.scaler_column: Tensor = torch.zeros(
            self.shape[0], device=self.device
        )
        self.nsamples: int = 0

    def update(self, input: Tensor, output: Tensor) -> None:
        self.inputs.append(input[0].cpu())
        self.outputs.append(output.cpu())

    def update_batch(self) -> None:
        if isinstance(self.inputs, list):
            self.inputs = torch.cat(self.inputs, dim=0)
        if isinstance(self.outputs, list):
            self.outputs = torch.cat(self.outputs, dim=0)
        self.inputs = self.inputs.to(self.device)
        self.outputs = self.outputs.to(self.device)



def find_layers(
    model: Module,
    layer_types: Optional[List[Type[Module]]] = None,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
    prefix: str = "",
) -> Dict[str, Module]:
    if layer_types is None:
        layer_types = [nn.Linear]
    if include_layers is None:
        include_layers = []
    if exclude_layers is None:
        exclude_layers = []
    layers_dict: Dict[str, Module] = {}

    def recursive_find(module: Module, prefix: str) -> None:
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
    model: Module,
    sparsity_ratio: float = 0.6,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
) -> None:
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
    model: Module,
    sparsity_ratio: float = 0.4,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
) -> None:
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
    model: Module,
    module: Module,
    dataloader: SamplingDataset,
    sparsity_ratio: float = 0.6,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
    p: int = 2,
) -> None:
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
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            model(input_ids, attention_mask=attn_mask)
            module(input_ids, attention_mask=attn_mask)

    for handle in ref_handle_list + target_handle_list:
        handle.remove()

    for name, wrapper_pair in wrappers.items():
        wrapper_pair["ref"].update_batch()
        wrapper_pair["target"].update_batch()

        ref_outputs = wrapper_pair["ref"].outputs
        target_outputs = wrapper_pair["target"].outputs

        current_weight = wrapper_pair["target"].layer.weight.data

        output_loss = target_outputs - ref_outputs

        output_loss = output_loss.reshape(
            (-1, output_loss.shape[-1])
        )  # (batch_size * seq_dim, output_dim)

        output_loss = output_loss.t()  # (output_dim, batch_size * seq_dim)
        importance_score = torch.norm(output_loss, p=p, dim=1)  # (output_dim)

        importance_score = torch.abs(current_weight) * importance_score.reshape(
            (-1, 1)
        )  # (output_dim, input_dim) * (output_dim, 1) = (output_dim, input_dim)

        W_mask = torch.zeros_like(importance_score) == 1
        sort_res = torch.sort(importance_score, dim=0, stable=True)
        indices = sort_res[1][: int(importance_score.shape[0] * sparsity_ratio), :]
        W_mask.scatter_(0, indices, True)
        current_weight[W_mask] = 0

        del ref_outputs
        del target_outputs
        del output_loss
        del importance_score
        del wrapper_pair["ref"]
        del wrapper_pair["target"]
        gc.collect()


def recover_tangling_identification(
    model: Module,
    module: Module,
    dataloader: SamplingDataset,
    recovery_ratio: float = 0.4,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
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
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            model(input_ids, attention_mask=attn_mask)
            module(input_ids, attention_mask=attn_mask)

    for handle in ref_handle_list + target_handle_list:
        handle.remove()

    for name, wrapper_pair in wrappers.items():
        wrapper_pair["ref"].update_batch()
        wrapper_pair["target"].update_batch()

        ref_outputs = wrapper_pair["ref"].outputs
        target_outputs = wrapper_pair[
            "target"
        ].outputs  # (batch_size, seq_dim, output_dim)
        inputs = wrapper_pair["target"].inputs  # (batch_size, seq_dim, input_dim)

        original_weight = wrapper_pair["ref"].layer.weight.data
        current_weight = wrapper_pair["target"].layer.weight.data

        output_loss = target_outputs - ref_outputs
        output_loss = output_loss.reshape(
            (-1, output_loss.shape[-1])
        )  # (batch_size * seq_dim, output_dim)
        inputs_flat = inputs.reshape(
            (-1, inputs.shape[-1])
        )  # (batch_size * seq_dim, input_dim

        inverse_inputs = torch.linalg.pinv(
            inputs_flat
        )  # (input_dim, batch_size * seq_dim)
        pseudo_weight_matrix = torch.matmul(
            inverse_inputs, output_loss
        )  # (input_dim, output_dim)

        importance_score = torch.abs(
            pseudo_weight_matrix.T * current_weight
        )  # (output_dim, input_dim)

        flattened_importance_score = importance_score.reshape(-1)
        flattened_original_weight = original_weight.reshape(-1)
        flattened_current_weight = current_weight.reshape(-1)

        # Sort importance scores in descending order
        sort_res = torch.sort(flattened_importance_score, descending=True)
        sorted_indices = sort_res[1]

        # Determine the number of elements to restore based on sparsity ratio
        num_elements_to_restore = int(
            flattened_importance_score.shape[0] * recovery_ratio
        )

        # Identify weights that are not included in the current model
        not_included_mask = flattened_original_weight != flattened_current_weight

        # Get the indices of not included weights from the sorted list
        sorted_not_included_indices = sorted_indices[not_included_mask[sorted_indices]]

        # Select top num_elements_to_restore indices from not included weights
        if len(sorted_not_included_indices) > num_elements_to_restore:
            restore_indices = sorted_not_included_indices[:num_elements_to_restore]
        else:
            restore_indices = sorted_not_included_indices

        # Create mask for restoring weights
        W_mask = torch.zeros_like(flattened_current_weight, dtype=torch.bool)
        W_mask[restore_indices] = True

        # Restore the weights based on the mask
        flattened_current_weight[W_mask] = flattened_original_weight[W_mask]

        # Reshape weights back to their original shape
        current_weight.copy_(flattened_current_weight.view_as(current_weight))

        del ref_outputs
        del target_outputs
        del inputs
        del output_loss
        del inputs_flat
        del inverse_inputs
        del pseudo_weight_matrix
        del importance_score
        del flattened_importance_score
        del wrapper_pair["ref"]
        del wrapper_pair["target"]
        gc.collect()


def prune_wanda(
    model: Module,
    dataloader: SamplingDataset,
    sparsity_ratio: float = 0.4,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
    p: int = 2,
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
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            model(input_ids, attention_mask=attn_mask)

    for handle in handle_list:
        handle.remove()

    for name, wrapper in wrappers.items():
        wrapper.update_batch()
        current_weight = wrapper.layer.weight.data

        X = wrapper.inputs  # (batch_size, seq_dim, input_dim)
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
        tmp = X.shape[0]  # (batch_size)
        if len(X.shape) == 3:
            X = X.reshape((-1, X.shape[-1]))  # (batch_size * seq_dim, input_dim)

        X = X.t()  # (input_dim, batch_size * seq_dim)
        wrapper.scaler_row *= wrapper.nsamples / (wrapper.nsamples + tmp)  # (input_dim)
        wrapper.nsamples += tmp
        wrapper.scaler_row += (
            torch.norm(X, p=p, dim=1) ** 2 / wrapper.nsamples
        )  # (input_dim)

        W_metric = torch.abs(current_weight) * torch.sqrt(
            wrapper.scaler_row.reshape((1, -1))
        )  # (output_dim, input_dim) * (1, input_dim) = (output_dim, input_dim)
        W_mask = torch.zeros_like(W_metric) == 1
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        indices = sort_res[1][:, : int(W_metric.shape[1] * sparsity_ratio)]
        W_mask.scatter_(1, indices, True)
        current_weight[W_mask] = 0


def head_prune():
    pass
