import numpy as np
import torch
import copy
from functools import partial
from tqdm.auto import tqdm
from utils.model_utils.evaluate import evaluate_model
import torch.nn as nn
from transformers.pytorch_utils import find_pruneable_heads_and_indices
from typing import *


def entropy(p):
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)


def layer_max(arr):
    max_index = np.argmax(arr, axis=1)
    return max_index


def compute_heads_importance(model, model_config, dataloader):
    multihead_outputs_list = []
    per_class_importance_list = [
        torch.zeros(12, 12).to(model_config.device)
        for _ in range(model_config.num_labels)
    ]
    per_class_token_list = [0.0 for _ in range(model_config.num_labels)]

    def register_hooks(model):
        handles = []

        def hook_fn(module, input, output, layer_index):
            attention_value, attention_scores = output
            attention_value.requires_grad_(True)
            attention_value.retain_grad()
            multihead_outputs_list.append(attention_value)

        for layer_index, layer in enumerate(model.bert.encoder.layer):
            handle = layer.attention.self.register_forward_hook(
                partial(hook_fn, layer_index=layer_index)
            )
            handles.append(handle)
        return handles

    def remove_hooks(handles):
        for handle in handles:
            handle.remove()

    # Prepare our tensors
    handles = register_hooks(model)
    n_layers, n_heads = (
        model.bert.config.num_hidden_layers,
        model.bert.config.num_attention_heads,
    )
    head_importance = torch.zeros(n_layers, n_heads).to(model_config.device)
    each_pred_head_importance = torch.zeros(n_layers, n_heads).to(model_config.device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(model_config.device)
    preds = None
    labels = None
    tot_tokens = 0.0

    for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
        input_ids = batch["input_ids"].to(model_config.device)
        label_ids = batch["labels"].to(model_config.device)

        model = model.to(model_config.device)
        actual_batch_size = input_ids.size(0)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(input_ids, output_attentions=True)
        all_attentions = outputs[1]
        logits = outputs[0]

        # Update head attention entropy
        for layer, attn in enumerate(all_attentions):
            masked_entropy = entropy(attn.detach())
            attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

        # Update head importance scores with regards to our loss
        # First, backpropagate to populate the gradients

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model_config.num_labels), label_ids.view(-1))

        loss.backward()

        # Second, compute importance scores according to http://arxiv.org/abs/1905.10650
        for layer, mh_layer_output in enumerate(multihead_outputs_list):
            # print(layer)
            mh_layer_output_store = mh_layer_output
            reshaped_mh_layer_output = mh_layer_output_store.view(
                actual_batch_size, 512, 12, 64
            )
            reshaped_mh_layer_output = reshaped_mh_layer_output.permute(0, 2, 1, 3)

            mh_layer_output_grad = mh_layer_output.grad
            reshaped_mh_layer_output_grad = mh_layer_output_grad.view(
                actual_batch_size, 512, 12, 64
            )
            reshaped_mh_layer_output_grad = reshaped_mh_layer_output_grad.permute(
                0, 2, 1, 3
            )
            dot = torch.einsum(
                "bhli,bhli->bhl",
                [reshaped_mh_layer_output_grad, reshaped_mh_layer_output],
            )
            each_head_importance = dot.abs().sum(-1).sum(0).detach()
            head_importance[layer] += each_head_importance
            each_pred_head_importance[layer] += each_head_importance
        temp_each_pred_head_importance = copy.deepcopy(each_pred_head_importance)
        each_pred_head_importance.zero_()
        multihead_outputs_list.clear()

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label_ids.detach().cpu().numpy(), axis=0)
        predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)

        for prediction in predictions:
            per_class_importance_list[prediction] += temp_each_pred_head_importance
            per_class_token = (input_ids != 0).float().sum().item()
            per_class_token_list[prediction] += per_class_token
            tot_tokens += per_class_token

    # Normalize
    attn_entropy /= tot_tokens
    head_importance /= tot_tokens
    for i in range(model_config.num_labels):
        per_class_importance_list[i] /= per_class_token_list[i]

    # Layerwise importance normalization

    exponent = 2
    norm_by_layer = torch.pow(
        torch.pow(head_importance, exponent).sum(-1), 1 / exponent
    )
    head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20
    for i in range(model_config.num_labels):
        norm_by_layer = torch.pow(
            torch.pow(per_class_importance_list[i], exponent).sum(-1), 1 / exponent
        )
        per_class_importance_list[i] /= norm_by_layer.unsqueeze(-1) + 1e-20

    head_importance = (head_importance - head_importance.min()) / (
        head_importance.max() - head_importance.min()
    )
    for i in range(model_config.num_labels):
        per_class_importance_list[i] = (
            per_class_importance_list[i] - per_class_importance_list[i].min()
        ) / (per_class_importance_list[i].max() - per_class_importance_list[i].min())
    remove_hooks(handles)
    for i in range(model_config.num_labels):
        per_class_importance_list[i] = per_class_importance_list[i].cpu().numpy()

    return attn_entropy, head_importance, preds, labels, per_class_importance_list


def print_prune_head_list(prune_list, trial):
    print(f"total prune number : {len(prune_list)*trial}")
    print(f"prune head list")
    print(prune_list)


def calculate_prune_head(arr, i):
    flattened_with_indices = [(value, index) for index, value in np.ndenumerate(arr)]

    sorted_by_value = sorted(flattened_with_indices, key=lambda x: x[0])
    bottom_12 = sorted_by_value[12 * i : 12 * (i + 1)]

    bottom_12_indices = [index for _, index in bottom_12]

    return bottom_12_indices


def prune_head(model, prune_list):
    for layer_index, head_index in prune_list:
        prune_heads(model.bert.encoder.layer[layer_index].attention, ([head_index]))
    return model


def total_preprocess_prunehead(arr):
    max_layer = layer_max(arr)
    for layer in range(12):
        head = max_layer[layer]
        arr[layer][head] = 100
    return arr


def head_importance_prunning(
    model, model_config, dataloader, num_steps, per_class_head_importance_list
):
    for class_index in range(model_config.num_labels):
        temp_model = copy.deepcopy(model)
        for num in range(num_steps):
            print(f"Class {class_index} {(num + 1) * 12} prunning")
            prune_list = calculate_prune_head(
                per_class_head_importance_list[class_index], num
            )
            print_prune_head_list(prune_list, num + 1)
            temp_model = prune_head(temp_model, prune_list)
        evaluate_model(temp_model, model_config, dataloader)


def total_head_importance_prunning(
    model, model_config, dataloader, num_steps, temp_head_importance_score
):
    temp_model = copy.deepcopy(model)
    for num in range(num_steps):
        print(f"Total {(num+1)*12} prunning")
        prune_list = calculate_prune_head(temp_head_importance_score, num)
        print_prune_head_list(prune_list, num + 1)
        temp_model = prune_head(temp_model, prune_list)
    evaluate_model(temp_model, model_config, dataloader)


def prune_heads(layer, heads):
    if len(heads) == 0:
        return
    heads, index = find_pruneable_heads_and_indices(
        heads,
        layer.self.num_attention_heads,
        layer.self.attention_head_size,
        layer.pruned_heads,
    )

    # Zero out weights in linear layers instead of pruning
    layer.self.query = zero_out_head_weights(
        layer.self.query, heads, layer.self.attention_head_size
    )
    layer.self.key = zero_out_head_weights(
        layer.self.key, heads, layer.self.attention_head_size
    )
    layer.self.value = zero_out_head_weights(
        layer.self.value, heads, layer.self.attention_head_size
    )
    layer.output.dense = zero_out_head_weights(
        layer.output.dense, heads, layer.self.attention_head_size, dim=1
    )


def zero_out_head_weights(
    layer: nn.Linear, heads: Set[int], head_size: int, dim: int = 0
) -> nn.Linear:
    """
    Zero out the weights of the specified heads in the linear layer.

    Args:
        layer (`torch.nn.Linear`): The layer to modify.
        heads (`Set[int]`): The indices of heads to zero out.
        head_size (`int`): The size of each head.
        dim (`int`, *optional*, defaults to 0): The dimension on which to zero out the weights.

    Returns:
        `torch.nn.Linear`: The modified layer with weights of specified heads zeroed out.
    """
    for head in heads:
        start_index = head * head_size
        end_index = (head + 1) * head_size
        if dim == 0:
            layer.weight.data[start_index:end_index] = 0
        elif dim == 1:
            layer.weight.data[:, start_index:end_index] = 0

    return layer
