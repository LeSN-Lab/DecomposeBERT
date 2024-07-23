import torch
import torch.nn as nn
from scipy.stats import norm
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
        self.inputs.append(input)
        self.outputs.append(output)

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

        loss = target_outputs - ref_outputs
        if loss.dim() == 2:
            loss = loss.unsqueeze(1)
        if p == 1:
            output_loss = torch.sum(torch.abs(loss), dim=1)
        elif p == 2:
            output_loss = torch.sqrt(torch.sum(loss**2, dim=1))
        elif p == "inf":
            output_loss = torch.max(torch.abs(loss), dim=1)[0]
        positive_loss_mask = (
            torch.all(output_loss > 0, dim=0).unsqueeze(1).expand(-1, shape[1])
        )
        original_weight_std = safe_std(original_weight, dim=1, keepdim=True)
        current_weight_std = safe_std(
            current_weight,
            epsilon=original_weight_std,
            unbiased=True,
            dim=1,
            keepdim=True,
        )

        padded_positive = torch.where(
            current_weight > 0, current_weight, torch.tensor(float("nan"))
        )
        padded_negative = torch.where(
            current_weight < 0, current_weight, torch.tensor(float("nan"))
        )
        positive_mean = torch.nanmean(padded_positive, dim=1, keepdim=True)
        negative_mean = torch.nanmean(padded_negative, dim=1, keepdim=True)

        positive_std = safe_std(
            current_weight,
            epsilon=current_weight_std,
            unbiased=True,
            dim=1,
            keepdim=True,
        )
        negative_std = safe_std(
            current_weight,
            epsilon=current_weight_std,
            unbiased=True,
            dim=1,
            keepdim=True,
        )

        positive_scores = (padded_positive - positive_mean) / positive_std
        negative_scores = (padded_negative - negative_mean) / negative_std

        positive_median = torch.nanmedian(padded_positive, dim=1, keepdim=True)
        negative_median = torch.nanmedian(padded_negative, dim=1, keepdim=True)
        lower_z, upper_z = norm.ppf(0.1), norm.ppf(0.3)

        positive_remove_mask = torch.where(
            positive_mean < positive_median.values,
            positive_scores <= lower_z,
            torch.logical_and(positive_scores >= lower_z, positive_scores < upper_z),
        )

        negative_remove_mask = torch.where(
            negative_mean < negative_median.values,
            torch.logical_and(negative_scores < -lower_z, negative_scores >= -upper_z),
            negative_scores >= -upper_z,
        )

        remove_mask = torch.where(
            positive_loss_mask, positive_remove_mask, negative_remove_mask
        )

        total_elements = current_weight.numel()
        num_elements_to_prune = int(total_elements * sparsity_ratio)
        current_pruned_elements = torch.sum(remove_mask).item()

        if current_pruned_elements < num_elements_to_prune:
            # Not enough elements to prune, relax the threshold
            while current_pruned_elements < num_elements_to_prune:
                lower_z += 0.1
                upper_z += 0.1
                positive_remove_mask = torch.where(
                    positive_mean < positive_median.values,
                    positive_scores <= lower_z,
                    torch.logical_and(
                        positive_scores >= lower_z, positive_scores < upper_z
                    ),
                )
                negative_remove_mask = torch.where(
                    negative_mean < negative_median.values,
                    torch.logical_and(
                        negative_scores < -lower_z, negative_scores >= -upper_z
                    ),
                    negative_scores >= -upper_z,
                )
                remove_mask = torch.where(
                    ~positive_loss_mask, positive_remove_mask, negative_remove_mask
                )
                current_pruned_elements = torch.sum(remove_mask).item()

        current_weight[remove_mask] = 0

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
