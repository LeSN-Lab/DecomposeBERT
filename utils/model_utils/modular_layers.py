import copy
import math

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from utils.type_utils.layer_type import ActivationType, LayerType
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)


class Layer(nn.Module):
    """
    Wrapper a single layer.

    Attributes:
        name (str): Name of the layer.
        layer (torch.nn.Module): Actual layer module.
        layer_type (LayerType): Type of the layer.
        shape (tuple): Input and output shape of the layer.
        act_fn (ActivationType): Activation function used in the layer.
        weight (torch.nn.Parameter): Weight parameter of the layer (if applicable).
        bias (torch.nn.Parameter): Bias parameter of the layer (if applicable).

    """

    def __init__(self, name, layer):
        super().__init__()
        self.name = name
        self.layer = layer
        self.layer_type = None
        self.shape = None
        self.act_fn = None
        self.weight, self.bias = None, None
        self.init_states()

    def init_states(self):
        self.layer_type = None
        self.shape = None
        self.act_fn = ActivationType.get_act_fn_type(self.layer)

        if self.act_fn == ActivationType.Linear:
            self.layer_type = LayerType.get_layer_type(self.layer)
            self.shape = LayerType.get_layer_shape(self.layer)
        else:
            self.layer_type = LayerType.Activation
            self.shape = None

        if self.layer_type not in [
            LayerType.Activation,
            LayerType.Dropout,
            LayerType.LayerNorm,
            LayerType.Embedding,
        ]:
            self.weight, self.bias = self.get_parameters()

    def get_parameters(self):
        """
        Extracts weight and bias parameters from a given layer, if available.

        Returns:
            tuple: A tuple containing weight and bias parameters (or None if not available).
        """
        if self.layer_type == LayerType.Activation:
            return None, None

        weight = (
            self.layer.weight.detach()
            if hasattr(self.layer, "weight") and self.layer.weight is not None
            else None
        )
        bias = (
            self.layer.bias.detach()
            if hasattr(self.layer, "bias") and self.layer.bias is not None
            else None
        )
        return weight, bias

    def set_parameters(self):
        if self.layer_type not in [
            LayerType.Activation,
            LayerType.Dropout,
            LayerType.LayerNorm,
            LayerType.Embedding,
        ]:
            self.layer.weight = torch.nn.Parameter(self.weight)
            if self.bias is not None:
                self.layer.bias = torch.nn.Parameter(self.bias)
    def get(self, name):
        if self.name == name:
            return self
        else:
            return None

    def forward(self, x):
        """
        Forward pass through the wrapped layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.layer_type != LayerType.Activation:
            return self.layer(x)


class ModularLayer(nn.Module):
    """
    A container for managing a sequence of layers, allowing for modular construction of neural networks.

    Attributes:
        name (str): Name of the modular layer.
    """

    def __init__(self, name):
        super(ModularLayer, self).__init__()
        self.name = name
        self.layers = nn.ModuleList()

    def append(self, layer):
        """
        Appends a new layer to the modular layer.
        """
        self.layers.append(layer)

    def append_layers(self, layers):
        """
        Appends multiple layers from another `nn.Module` object.

        Args:
            layers (nn.Module): The module containing layers to be added.
        """
        for name, layer in layers.named_children():
            self.append(Layer(name, layer))

    def get(self, name):
        """
        Retrieves a layer by its name.

        Args:
            name (str): The name of the layer to retrieve.

        Returns:
            Layer: The layer object if found, otherwise None.
        """
        for layer in self.layers:
            if layer.name != name:
                sub_layer = layer.get(name)
                if sub_layer is not None:
                    return sub_layer
            else:
                return layer
        return None

    def get_layer(self, name):
        layer = self.get(name)
        return layer if layer else None

    def get_type(self, name):
        """
        Retrieves the layer_type of a layer by its name.

        Args:
            name (str): The name of the layer.

        Returns:
            LayerType: The type of the layer if found, otherwise None.
        """
        layer = self.get(name)
        return layer.layer_type if layer else None


class EmbeddingModule(ModularLayer):
    """
    A modular layer specifically designed for handling various embedding operations.

    Attributes:
        config (object): Configuration.
        position_embeddings (Layer): The positional embedding layer.
        token_type_embeddings (Layer): The token type embedding layer.
        position_embedding_type (str): Type of position embeddings ("absolute").
    """

    def __init__(self, layers, config):
        super().__init__("embeddings")
        self.config = config
        self.append_layers(layers)  # Add embedding layers to this module

        # Automatically retrieve and assign embedding layers
        self.word_embeddings = self.get("word_embeddings")
        self.position_embeddings = self.get("position_embeddings")
        self.token_type_embeddings = self.get("token_type_embeddings")
        self.LayerNorm = self.get("LayerNorm")
        self.dropout = self.get("dropout")

        # Determine position embedding type
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

        # Initialize position IDs for efficient lookup
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids):
        """
        Performs embedding operations on input token indices.

        Args:
            input_ids (torch.Tensor): Tensor containing token indices of input sequences.

        Returns:
            torch.Tensor: The embedded representation of the input sequences.
        """
        input_shape = input_ids.size()
        seq_length = input_shape[1]  # Length of input sequences

        # Prepare position IDs for the specific sequence length
        position_ids = self.position_ids[:, 0:seq_length].to(input_ids.device)

        # Create token type IDs (all zeros for now)
        token_type_ids = torch.zeros(
            input_shape, dtype=torch.long, device=input_ids.device
        )

        # Perform word embeddings and token type embeddings
        inputs_embeds = self.word_embeddings.layer(input_ids)

        token_type_embeddings = self.token_type_embeddings.layer(token_type_ids)

        position_embeddings = self.position_embeddings.layer(position_ids)

        # Combine embeddings
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        # Apply any additional embedding layers (e.g., LayerNorm)
        for layer in self.layers:
            if layer.name not in [
                "word_embeddings",
                "position_embeddings",
                "token_type_embeddings",
            ]:
                embeddings = layer.layer(embeddings)

        return embeddings


class SelfAttentionModule(ModularLayer):
    """
    A modular layer responsible for multi-head self-attention operations.

    Attributes:
        config (object): Configuration.
        num_attention_heads (int): Number of attention heads to use.
        attention_head_size (int): Size of each attention head.
        all_head_size (int): Total size of concatenated attention heads.
        query (Layer): Linear layer for generating query representations.
        key (Layer): Linear layer for generating key representations.
        value (Layer): Linear layer for generating value representations.
        dropout (Layer): Dropout layer for regularization.
        position_embedding_type (str): Type of position embeddings ("absolute").
    """

    def __init__(self, layers, config):
        super().__init__("self")
        self.config = config
        # Check for compatibility between hidden size and number of attention heads
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.append_layers(layers)  # Add self-attention layers to this module
        self.query = self.get("query")
        self.key = self.get("key")
        self.value = self.get("value")
        self.dropout = self.get("dropout")
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

    def transpose_for_scores(self, x):
        """
        Transposes input for attention scores calculation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transposed tensor.
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask):
        """
        Performs multi-head self-attention on the input.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            attention_mask (torch.Tensor, optional): Attention mask.
            head_mask (torch.Tensor, optional): Mask for attention heads.

        Returns:
            tuple: (torch.Tensor, torch.Tensor) representing the context layer and attention probabilities.
        """

        mixed_query_layer = self.query.layer(hidden_states)

        key_layer = self.transpose_for_scores(self.key.layer(hidden_states))
        value_layer = self.transpose_for_scores(self.value.layer(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask and normalize scores
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # Apply dropout and head mask
        attention_probs = self.dropout.layer(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Calculate context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, attention_probs

    def prune(self, heads, index):
        """
        Prunes specific attention heads from the module.

        Args:
            heads (`list[int]`): list of the indices of heads to prune.
            index (`torch.LongTensor`): The indices to keep in the layer.
        """

        # Prune each of the query, key, and value linear layers
        self.query.layer = prune_linear_layer(self.query.layer, index)
        self.key.layer = prune_linear_layer(self.key.layer, index)
        self.value.layer = prune_linear_layer(self.value.layer, index)
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads


class OutputModule(ModularLayer):
    """
    A modular layer responsible for post-attention processing and output projection.

    Attributes:
        config (object): Configuration
        dense (Layer): Dense linear layer for output projection.
        LayerNorm (Layer): Layer normalization layer for stabilization.
        dropout (Layer): Dropout layer for regularization.
    """

    def __init__(self, layers, config, name="output"):
        super().__init__(name)
        self.config = config
        self.append_layers(layers)
        self.dense = self.get("dense")
        self.LayerNorm = self.get("LayerNorm")
        self.dropout = self.get("dropout")

    def prune(self, index):
        """
        Prune a linear layer to keep only entries in index.
        Used to remove heads

        Args:
            index (`torch.LongTensor`): The indices to keep in the layer.
        """
        self.dense = prune_linear_layer(self.dense.layer, index, dim=1)

    def forward(self, hidden_states, input_tensor):
        """
        Performs output processing and projection.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            input_tensor (torch.Tensor): Input tensor to be added after normalization.

        Returns:
            torch.Tensor: Processed and projected output.
        """

        hidden_states = self.dense.layer(hidden_states)
        hidden_states = self.dropout.layer(hidden_states)
        hidden_states = self.LayerNorm.layer(hidden_states + input_tensor)

        return hidden_states


class AttentionModule(ModularLayer):
    """
    A modular layer responsible for self-attention and output projection.

    Attributes:
        config (object): Configuration
        self_attention (SelfAttentionModule): Self-attention submodule.
        output (OutputModule): Output projection submodule.
        pruned_heads (set[int]): Set of already pruned attention head indices.
    """

    def __init__(self, layers, config):
        super().__init__("attention")
        self.config = config
        self.self_attention = SelfAttentionModule(layers.self, config)
        self.output = OutputModule(layers.output, config, "output")
        self.pruned_heads = set()

    def forward(self, hidden_states, attention_mask, head_mask):
        """
        Performs self-attention and output projection.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            attention_mask (torch.Tensor): Attention mask.
            head_mask (torch.Tensor): Attention head mask (optional).

        Returns:
            tuple: (torch.Tensor,) + tuple:
                - First element is the projected output.
                - Remaining elements are additional outputs from self-attention.
        """

        self_outputs = self.self_attention(hidden_states, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs  # output tensor, other output tensors

    def prune_heads(self, heads):
        """
        Prunes specific attention heads from the module.

        Args:
            heads (list[int]): Indices of the heads to be pruned.
        """
        if not heads:
            return

        # Filter and find indices of prunable heads considering already pruned ones
        heads, index = find_prunable_heads_and_indices(
            heads,
            self.self_attention.num_attention_heads,
            self.self_attention.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers in self-attention module and update pruned heads set
        self.self_attention.prune(heads, index)
        self.pruned_heads = self.pruned_heads.union(heads)


class FeedforwardModule(ModularLayer):
    """
    A modular layer responsible for a sequence of feed-forward layers.

    Attributes:
        config (object): Configuration
    """

    def __init__(self, layers, config):
        super().__init__("feedforward1")
        self.config = config
        self.append_layers(layers)

        self.dense = self.get("dense")
        self.intermediate_act_fn = self.get("intermediate_act_fn")

    def forward(self, input_tensor):
        """
        Passes the input through each layer in the sequence.

        Args:
            input_tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after processing through all layers.
        """
        output_tensor = self.dense.layer(input_tensor)
        output_tensor = self.intermediate_act_fn.layer(output_tensor)

        return output_tensor


class EncoderBlock(ModularLayer):
    """
    A single block in the encoder stack, performing self-attention and feed-forward processing.

    Attributes:
        config (object): Configuration.
        attention (AttentionModule): The self-attention submodule.
        feed_forward1 (FeedforwardModule): The first feed-forward submodule.
        feed_forward2 (OutputModule): The output projection submodule.
    """

    def __init__(self, layers, config):
        super().__init__("encoder_block")
        self.config = config
        self.attention = AttentionModule(layers.attention, config)
        self.feed_forward1 = FeedforwardModule(layers.intermediate, config)
        self.feed_forward2 = OutputModule(layers.output, config, "feedforward2")

    def forward(self, hidden_states, attention_mask, head_mask):
        """
        Processes the input with self-attention, feed-forward layers, and residual connections.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            attention_mask (torch.Tensor): Attention mask.
            head_mask (torch.Tensor, optional): Attention head mask.

        Returns:
            tuple: (torch.Tensor,) + tuple:
                - First element is the processed output.
                - Remaining elements are additional outputs from self-attention.
        """

        # Self-attention with optional head mask for selective attention
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]  # Extract main output
        other_outputs = attention_outputs[1:]  # Keep other attention outputs

        # First feed-forward pass
        intermediate_output = self.feed_forward1(attention_output)
        # Second feed-forward pass with output projection
        layer_output = self.feed_forward2(
            intermediate_output, attention_output
        )  # Add residual

        # Combine processed output with original states for skip connections
        outputs = (layer_output,) + other_outputs
        return outputs


class EncoderModule(ModularLayer):
    """
    A modular layer that stacks multiple encoder blocks for sequential processing.

    Attributes:
        config (object): Configuration settings for the encoder module.
        encoder_blocks (list[EncoderBlock]): List of encoder blocks.
    """

    def __init__(self, blocks, config):
        super().__init__("encoder")
        self.config = config
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(block, config) for block in blocks]
        )

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        """
        Processes the input through the stacked encoder blocks.

        Args:
            input_tensor (torch.Tensor): Input tensor to be encoded.
            attention_mask (torch.Tensor, optional): Attention mask.
            head_mask (torch.Tensor, optional): Attention head mask.

        Returns:
            BaseModelOutputWithPoolingAndCrossAttentions: A structured output containing:
                - last_hidden_state: The final output of the encoder.
                - hidden_states: A list of hidden states at each block.
                - attentions: A list of attention outputs at each block.
        """
        for i, encoder_block in enumerate(self.encoder_blocks):
            # Optionally apply head mask for selective attention
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # Process through the encoder block
            block_outputs = encoder_block(
                input_tensor,
                attention_mask,
                layer_head_mask,
            )
            input_tensor = block_outputs[0]  # Updated input for next block

        return input_tensor


class PoolerModule(ModularLayer):
    """
    A modular layer that applies pooling operations to the input.

    Attributes:
        config (object): Configuration.
    """

    def __init__(self, layers, config):
        super().__init__("pooler")
        self.config = config
        self.append_layers(layers)
        self.dense = self.get("dense")
        self.activation = self.get("activation")

    def forward(self, hidden_states):
        """
        Passes the input through the pooling layers.

        Args:
            hidden_states (torch.Tensor): Input tensor to be pooled.

        Returns:
            torch.Tensor: The pooled output.
        """
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense.layer(first_token_tensor)
        pooled_output = self.activation.layer(pooled_output)
        return pooled_output


class ModularClassificationBERT(ModularLayer):
    """
    A modular BERT model for text classification tasks.

    Attributes:
        num_labels (int): Number of classification labels.
        config (object): Configuration.
        embeddings (EmbeddingModule): Embeddings module.
        encoder (EncoderModule): Encoder module with stacked encoder blocks.
        pooler (PoolerModule): Pooler module for final representation.
        dropout (Layer): Dropout layer for regularization.
        classifier (Layer): Output classification layer.

    """

    def __init__(self, model, config):
        super().__init__("bert")
        self.num_labels = config.num_labels
        self.config = config
        copied_model = copy.deepcopy(model)
        self.embeddings = EmbeddingModule(copied_model.bert.embeddings, config)
        self.encoder = EncoderModule(copied_model.bert.encoder.layer, config)
        self.pooler = PoolerModule(copied_model.bert.pooler, config)

        self.dropout = Layer("dropout", copied_model.dropout)
        self.classifier = Layer("classifier", copied_model.classifier)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        """
        Performs the forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels
            head_mask (torch.Tensor, optional): Attention head mask.

        Returns:
            BaseModelOutputWithPoolingAndCrossAttentions: Structured output containing:
                - last_hidden_state: Final output from the encoder.
                - pooler_output: Pooled output if a pooler is used.
                - hidden_states: List of hidden states at each encoder block.
                - attentions: List of attention outputs at each encoder block.

        """
        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # Create attention mask if not provided
        attention_mask = get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Get embeddings (word, positional)
        embedding_output = self.embeddings(input_ids=input_ids)
        # Pass through encoder with attention and head masks
        encoder_outputs = self.encoder(
            input_tensor=embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        sequence_output = encoder_outputs

        # Extract sequence output and apply pooling if available
        pooled_output = self.pooler(sequence_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

    def register_hook(self, hook):
        self.classifier.register_forward_hook(hook)
        self.pooler.register_forward_hook(hook)

    def _prune_heads(self, heads_to_prune):
        """
        Prunes specific attention heads in the encoder layers.

        Args:
            heads_to_prune (dict): A dictionary mapping layer indices to lists of head indices to prune.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder[layer].attention.prune_heads(heads)

    @staticmethod
    def get_head_mask(head_mask, num_hidden_layers):
        """
        Prepare the head mask if needed.

        Args:
            head_mask (torch.Tensor, optional): The mask indicating if we should keep the heads or not
            (1.0 for keep, 0.0 for discard).

            num_hidden_layers (int): Number of hidden layers in the model.

        Returns:
            list[torch.Tensor, optional]: List of head masks for each hidden layer.
        """
        if head_mask is not None:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask


def get_extended_attention_mask(attention_mask, input_shape):
    """
    Converts a 2D attention mask into a 3D tensor for BERT-style attention.

    Args:
        attention_mask (torch.Tensor): 2D attention mask.
        input_shape (tuple): Shape of the input tensor.

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
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
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
