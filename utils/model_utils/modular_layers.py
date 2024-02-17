import copy

import torch
import torch.nn as nn

from utils.type_utils.layer_type import ActivationType, LayerType


class Layer(nn.Module):

    def __init__(self, name, layer):
        super(Layer, self).__init__()
        self.name = name
        self.layer = layer
        self.weight, self.bias = None, None
        self.layer_type = None
        self.shape = None
        self.act_fcn = ActivationType.get_act_fcn_type(layer)

        self.padding_idx = None  # if layer type is LayerType.Embedding
        self.eps = None  # if layer type is LayerType.LayerNorm
        self.dropout_rate = None  # if layer type is LayerType.Dropout

        if self.act_fcn is ActivationType.Linear:  # if layer is not activation fcn
            self.layer_type = LayerType.get_layer_type(layer)
            self.shape = LayerType.get_layer_shape(layer)
        else:  # if layer is activation fcn
            self.layer_type = LayerType.Activation
            self.shape = None

        if self.layer_type is not LayerType.Activation:
            self.weight, self.bias = get_parameters(layer)

            self.padding_idx = getattr(layer, "padding_idx", None)
            self.eps = getattr(layer, "eps", None)
            self.dropout_rate = getattr(layer, "p", None)

    def forward(self, x):
        out = self.layer(x)

        if self.act_fcn:  # if layer type is activation fcn
            act_fn_map = {
                "relu": nn.ReLU,
                "gelu": nn.GELU,
                "tanh": nn.Tanh,
            }
            out = act_fn_map[self.act_fcn](out) if self.act_fcn in act_fn_map else out
        return out


class ModularLayer(nn.ModuleList):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.layers = nn.ModuleList()

    def append(self, layer):
        self.layers.append(layer)

    def append_layers(self, layers):
        for name, layer in layers.named_children():
            self.append(Layer(name, layer))

    def get_size(self):
        return len(self.layers)

    def forward(self, x):
        out = self.layer(x)
        return out

    def get(self, name):
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def get_copy(self, name):
        layer = self.get(name)
        return copy.deepcopy(layer)

    def get_type(self, name):
        layer = self.get(name)
        return layer.layer_type


class EmbeddingModule(ModularLayer):
    def __init__(self, layers, config):
        super().__init__("embeddings")
        self.config = config
        self.append_layers(layers)
        self.word_embeddings = self.get("word_embeddings")
        self.position_embeddings = self.get("position_embeddings")
        self.token_type_embeddings = self.get("token_type_embeddings")
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # Initialize position embeddings
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        """
        :param input_ids: tensors incluing token index of the input sequences
        :param token_type_ids: tensors including token types
        :param position_ids: tensors including position
        :return: embedded tensors
        """
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        for layer in self.layers:
            if layer.name not in ["word_embeddings", "position_embeddings", "token_type_embeddings"]:
                embeddings = layer(embeddings)
        return embeddings


class EncoderModule(ModularLayer):

    def __init__(self, blocks, config):
        super().__init__("encoder")
        self.config = config
        for _, block in enumerate(blocks):
            self.append(EncoderBlock(block, config))

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_state = ()
        all_self_attentions = ()
        for i, layer in enumerate(self.layers):
            all_hidden_state = all_hidden_state + (hidden_states,)

            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class EncoderBlock(ModularLayer):
    def __init__(self, layers, config):
        super().__init__("encoder_block")
        self.config = config
        self.attention = AttentionModule(layers.attention, config)
        self.feed_forward1 = FeedforwardModule(layers.intermediate, config)
        self.feed_forward2 = OutputModule(layers.output, config, "feedforward2")


    def forward(self, hidden_states):
        pass

    def feed_forward_chunk(self, attention_output):
        out = self.feed_forward2(self.feed_forward1(attention_output), attention_output)
        return out

class AttentionModule(ModularLayer):
    def __init__(self, layers, config):
        super().__init__("attention")
        self.self_attention = SelfAttentionModule(layers.self, config)
        self.output = OutputModule(layers.output, config, "output")
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self_attention.num_attention_heads, self.self_attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self_attention.prune(index, heads)
        self.pruned_heads = self.pruned_heads.union(heads)

class OutputModule(ModularLayer):
    def __init__(self, layers, config, name="output"):
        super().__init__(name)
        self.config = config
        self.append_layers(layers)
        self.dense = self.get("dense")
        self.LayerNorm = self.get("LayerNorm")
        self.dropout = self.get("dropout")

    def prune(self, index):
        self.dense = prune_linear_layer(self.dense, index)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class SelfAttentionModule(ModularLayer):
    def __init__(self, layers, config):
        super().__init__("self")
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.append_layers(layers)
        self.query = self.get("query")
        self.key = self.get("key")
        self.value = self.get("value")


    def prune(self, index, heads):
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads

class FeedforwardModule(ModularLayer):
    def __init__(self, layers, config):
        super().__init__("feedforward1")
        self.config = config
        self.append_layers(layers)


class PoolerModule(ModularLayer):
    def __init__(self, layers, config):
        super().__init__("pooler")
        self.config = config
        self.append_layers(layers)

class ModularClassificationBERT(ModularLayer):
    def __init__(self, model, config):
        super().__init__("bert")
        self.num_labels = config.num_labels
        self.config = config

        self.embeddings = EmbeddingModule(model.bert.embeddings, config)
        self.encoder = EncoderModule(model.bert.encoder.layer, config)
        self.pooler = PoolerModule(model.bert.pooler, config)
        self.dropout = Layer("dropout", model.dropout)
        self.classifier = Layer("classifier", model.classifier)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, input_embeds, labels, output_attentions, output_hidden_states):



def get_parameters(layer):
    weight, bias = None, None
    if hasattr(layer, "weight"):
        weight = layer.weight.detach() if layer.weight is not None else None
    if hasattr(layer, "bias"):
        bias = layer.bias.detach() if layer.bias is not None else None
    return weight, bias

def find_pruneable_heads_and_indices(
    heads, n_heads, head_size, already_pruned_heads
):
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
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index

def prune_linear_layer(layer, index, dim = 0):
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer