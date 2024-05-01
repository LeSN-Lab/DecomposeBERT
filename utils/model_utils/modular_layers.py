import torch
import torch.nn as nn

def get_extended_attention_mask(attention_mask):
    """
    Converts a 2D attention mask into a 3D tensor for BERT-style attention.

    Args:
        attention_mask (torch.Tensor): 2D attention mask.

    Returns:
        torch.Tensor: 3D attention mask.
    """
    # Expand 2D mask to 3D

    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor(attention_mask)

    if attention_mask.dim() == 2:
        # Create a 3D attention mask from a 2D tensor mask
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
        return extended_attention_mask
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape) or attention_mask (shape {attention_mask.shape})"
        )


def find_prunable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the indices of heads to prune taking `already_pruned_heads`
        into account and the indices of rows/columns to keep in the layer weight.
    """
    mask = torch.ones(n_heads, head_size)  # Create weight mask for all heads
    heads = set(heads) - already_pruned_heads  # Remove already pruned heads
    for head in heads:
        # Adjust head index considering previously pruned ones
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0  # Set mask to 0 for heads to be pruned
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


def prune_linear_layer(layer, index, dim=0):
    """
    Prune a linear layer to keep only entries in index.

    Args:
        layer (Layer): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        Layer: The pruned layer as a new layer with gradients enabled.
    """
    layer = layer.layer
    index = index.to(layer.weight.device)
    # Extract and clone relevant weight and bias portions
    weight = layer.weight.index_select(dim, index).clone().detach()
    bias = None
    if layer.bias is not None:
        if dim == 1:
            bias = layer.bias.clone().detach()
        else:
            bias = layer.bias[index].clone().detach()
    # Create new layer with adjusted size and copy weights/bias
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(
        layer.weight.device
    )
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(weight.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(bias.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

def transpose_for_scores(x, num_heads, head_size):
    """
    Transposes input for attention scores calculation.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Transposed tensor.
    """
    new_x_shape = x.size()[:-1] + (
        num_heads,
        head_size,
    )
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)

def set_parameters(module, weight, bias):
    module.weight = torch.nn.Parameter(weight)
    module.bias = torch.nn.Parameter(bias)