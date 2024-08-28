import torch
import torch.nn as nn
from scipy.stats import norm
from typing import *
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from utils.dataset_utils.sampling import SamplingDataset

from utils.helper import ModelConfig


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
        self.scaler_row: Tensor = torch.zeros(self.shape[1])
        self.scaler_column: Tensor = torch.zeros(self.shape[0])
        self.nsamples: int = 0

    def update(self, input: Tensor, output: Tensor) -> None:
        self.inputs.append(input[0].cpu())
        self.outputs.append(output.cpu())

    def update_batch(self) -> None:
        if isinstance(self.inputs, list):
            self.inputs = torch.cat(self.inputs, dim=0)
        if isinstance(self.outputs, list):
            self.outputs = torch.cat(self.outputs, dim=0)

    def to(self, device):
        self.layer = self.layer.to(device)
        self.inputs = self.inputs.to(device)
        self.outputs = self.outputs.to(device)
        self.scaler_row = self.scaler_row.to(device)

    def free(self):
        self.layer = self.layer.to(torch.device("cpu"))
        self.inputs = []
        self.outputs = []


class Methods:
    def __init__(self, method_type: str, ratio: float) -> None:
        self.method_type = method_type
        self.ratio = ratio

    def wanda(self, layer, inputs, outputs):
        current_weight = layer.weight.data
        X = inputs[0]  # (batch_size, seq_dim, input_dim)
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
        nsamples = X.shape[0]  # (batch_size)
        if len(X.shape) == 3:
            X = X.reshape((-1, X.shape[-1]))  # (batch_size * seq_dim, input_dim)

        X = X.t()  # (input_dim, batch_size * seq_dim)
        scaler_row = torch.norm(X, p=2, dim=1) ** 2 / nsamples

        W_metric = torch.abs(current_weight) * torch.sqrt(
            scaler_row.reshape((1, -1))
        )
        W_mask = torch.zeros_like(W_metric) == 1
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        indices = sort_res[1][:, : int(W_metric.shape[1] * self.ratio)]
        W_mask.scatter_(1, indices, True)
        current_weight[W_mask] = 0

    def ci(self, layer, inputs, outputs):
        current_weight = layer.weight.data
        X = inputs[0]

        batch_size = X.shape[0] // 2

        concern_inputs, non_concern_inputs = (
            X[:batch_size],
            X[batch_size:],
        )  # (batch_size, seq_dim, input_dim)

        calc_norm = lambda tensors, dim: torch.norm(
            tensors.reshape((-1, tensors.shape[-1])), dim=dim
        )

        concern_norm = calc_norm(concern_inputs, dim=0).reshape((1, -1))
        all_norm = calc_norm(X, dim=0).reshape((1, -1))
        non_concern_norm = calc_norm(non_concern_inputs, dim=0).reshape((1, -1))

        cosine_similarity = F.cosine_similarity(
            concern_inputs.reshape((-1, concern_inputs.shape[-1])),
            non_concern_inputs.reshape((-1, non_concern_inputs.shape[-1])),
            dim=0,
        ).reshape(1, -1)

        coefficient = concern_norm - cosine_similarity * torch.abs(
            concern_norm - non_concern_norm
        )
        importance_score = torch.abs(current_weight) * torch.abs(coefficient)

        W_mask = torch.zeros_like(importance_score) == 1
        sort_res = torch.sort(importance_score, dim=-1, stable=True)
        indices = sort_res[1][:, : int(importance_score.shape[1] * self.ratio)]
        W_mask.scatter_(1, indices, True)
        current_weight[W_mask] = 0

    def ti(self, layer, inputs, outputs, ref_layer):
        current_weight = layer.weight.data
        original_weight = ref_layer.weight.data

        X = inputs[0]

        batch_size = X.shape[0] // 2

        concern_inputs, non_concern_inputs = (
            X[:batch_size],
            X[batch_size:],
        )

        calc_norm = lambda tensors, dim: torch.norm(
            tensors.reshape((-1, tensors.shape[-1])), dim=dim
        )

        concern_norm = calc_norm(concern_inputs, dim=0).reshape((1, -1))
        all_norm = calc_norm(X, dim=0).reshape((1, -1))
        non_concern_norm = calc_norm(non_concern_inputs, dim=0).reshape((1, -1))

        cosine_similarity = F.cosine_similarity(
            concern_inputs.reshape((-1, concern_inputs.shape[-1])),
            non_concern_inputs.reshape((-1, non_concern_inputs.shape[-1])),
            dim=0,
        ).reshape(1, -1)

        coefficient = all_norm - cosine_similarity * torch.abs(
            concern_norm - non_concern_norm
        )

        importance_score = torch.abs(current_weight - original_weight) * torch.abs(coefficient)

        flattened_importance_score = importance_score.reshape(-1)
        flattened_original_weight = original_weight.reshape(-1)
        flattened_current_weight = current_weight.reshape(-1)

        # Sort importance scores in descending order
        sort_res = torch.sort(flattened_importance_score, descending=True)
        sorted_indices = sort_res[1]

        # Determine the number of elements to restore based on sparsity ratio
        num_elements_to_restore = int(
            flattened_importance_score.shape[0] * self.ratio
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


def get_hook(method):
    def hook(module, input, output):
        method(module, input, output)

    return hook


def propagate(model, dataloader, device, chunk_size=4):
    all_outputs = []
    chunk_outputs = []

    model = model.to(device)
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            output = model(
                input_ids, attention_mask=attn_mask, output_hidden_states=True
            )
            chunk_outputs.append(output.hidden_states[-1])
            if len(chunk_outputs) == chunk_size:
                all_outputs.append(torch.cat(chunk_outputs))
                chunk_outputs = []
    if chunk_outputs:
        all_outputs.append(torch.cat(chunk_outputs))
    all_outputs = torch.cat(all_outputs).cpu().detach().numpy()
    return all_outputs


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
        layer.weight.data[mask] = 0


def prune_concern_identification(
    model: Module,
    model_config: ModelConfig,
    dominant_concern: SamplingDataset,
    non_dominant_concern: SamplingDataset,
    sparsity_ratio: float = 0.6,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
) -> None:
    layers = find_layers(
        model, include_layers=include_layers, exclude_layers=exclude_layers
    )
    device = model_config.device
    handle_list = []

    method = Methods("ci", sparsity_ratio)
    for name, layer in layers.items():
        handle = layer.register_forward_hook(method.ci)
        handle_list.append(handle)

    dominant_batches = list(dominant_concern)
    non_dominant_batches = list(non_dominant_concern)

    if len(dominant_batches) != len(non_dominant_batches):
        raise ValueError("Batch sizes of dominant_concern and non_dominant_concern does not match.")

    combined_input_ids = torch.cat([batch["input_ids"] for batch in dominant_batches + non_dominant_batches])
    combined_attn_mask = torch.cat([batch["attention_mask"] for batch in dominant_batches + non_dominant_batches])

    combined_dataloader = [{"input_ids": combined_input_ids, "attention_mask": combined_attn_mask}]

    propagate(model, combined_dataloader, device)

    for handle in handle_list:
        handle.remove()


def recover_tangling_identification(
    model: Module,
    module: Module,
    model_config: ModelConfig,
    dominant_concern: SamplingDataset,
    non_dominant_concern: SamplingDataset,
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
    device = model_config.device

    handle_list = []

    method = Methods("ti", recovery_ratio)
    for (ref_name, ref_layer), (target_name, target_layer) in zip(
        ref_layers.items(), target_layers.items()
    ):
        handle = target_layer.register_forward_hook(
            lambda module, input, output: method.ti(module, input, output, ref_layer)
        )
        handle_list.append(handle)

    dominant_batches = list(dominant_concern)
    non_dominant_batches = list(non_dominant_concern)

    if len(dominant_batches) != len(non_dominant_batches):
        raise ValueError("Batch sizes of dominant_concern does not match.")

    combined_input_ids = torch.cat([batch["input_ids"] for batch in dominant_batches + non_dominant_batches])
    combined_attn_mask = torch.cat([batch["attention_mask"] for batch in dominant_batches + non_dominant_batches])

    combined_dataloader = [{"input_ids": combined_input_ids, "attention_mask": combined_attn_mask}]

    propagate(model, combined_dataloader, device)

    for handle in handle_list:
        handle.remove()


def prune_wanda(
    model: Module,
    model_config: ModelConfig,
    dataloader: SamplingDataset,
    sparsity_ratio: float = 0.4,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
):
    layers = find_layers(
        model, include_layers=include_layers, exclude_layers=exclude_layers
    )
    device = model_config.device
    handle_list = []

    method = Methods("wanda", sparsity_ratio)
    for name, layer in layers.items():
        handle = layer.register_forward_hook(method.wanda)
        handle_list.append(handle)
    propagate(model, dataloader, device)

    for handle in handle_list:
        handle.remove()
