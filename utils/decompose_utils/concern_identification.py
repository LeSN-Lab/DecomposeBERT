import torch
from utils.type_utils.data_type import safe_std
from utils.model_utils.modular_layers import set_parameters
from scipy.stats import norm


class ConcernIdentificationBert:
    def __init__(self, model, p=0.6):
        self.source_model = model
        self.p = p

    def propagate(self, module, original_input_tensor):
        # propagate input tensor to the module
        original_output_tensor, current_output_tensor = self.propagate_embeddings(
            self.source_model.bert.embeddings,
            module.bert.embeddings,
            original_input_tensor.clone(),
            original_input_tensor.clone(),
        )
        original_output_tensor, current_output_tensor = self.propagate_encoder(
            self.source_model.bert.encoder,
            module.bert.encoder,
            original_output_tensor,
            current_output_tensor,
        )
        original_output_tensor, current_output_tensor = self.propagate_pooler(
            self.source_model.bert.pooler,
            module.bert.pooler,
            original_output_tensor,
            current_output_tensor,
        )
        original_output_tensor, current_output_tensor = self.propagate_classifier(
            self.source_model.classifier,
            module.classifier,
            original_output_tensor,
            current_output_tensor,
        )
        return original_output_tensor, current_output_tensor

    def propagate_embeddings(
        self, ref_model, module, original_input_tensor, current_input_tensor
    ):
        original_output_tensor = ref_model(original_input_tensor)
        current_output_tensor = module(current_input_tensor)
        return original_output_tensor, current_output_tensor

    def propagate_encoder(
        self, ref_model, module, original_input_tensor, current_input_tensor
    ):
        for i, encoder_block in enumerate(module.layer):
            original_output_tensor, current_output_tensor = (
                self.propagate_encoder_block(
                    ref_model.layer[i],
                    encoder_block,
                    original_input_tensor,
                    current_input_tensor,
                )
            )
            original_input_tensor = original_output_tensor[0]
            current_input_tensor = current_output_tensor[0]
        return original_input_tensor, current_input_tensor

    def propagate_encoder_block(
        self, ref_model, module, original_input_tensor, current_input_tensor
    ):
        original_attn_output = ref_model.attention(original_input_tensor, None, None)
        current_attn_output = module.attention(current_input_tensor, None, None)
        original_output_tensor = ref_model.intermediate.dense(original_attn_output[0])

        def ff1_hook(module, input, output):
            current_weight, current_bias = module.weight, module.bias

            if torch.sum(current_weight != 0) > torch.numel(current_weight) * self.p:
                self.pruning(
                    ref_model.intermediate.dense,
                    module,
                    original_output_tensor[:, 0, :],
                    output[:, 0, :],
                )

        handle = module.intermediate.dense.register_forward_hook(ff1_hook)
        module.intermediate.dense(current_attn_output[0])
        handle.remove()

        original_intermediate_output = ref_model.intermediate(original_attn_output[0])
        current_intermediate_output = module.intermediate(current_attn_output[0])

        original_output_tensor = ref_model.output.dense(original_intermediate_output)

        def ff2_hook(module, input, output):
            current_weight, current_bias = module.weight, module.bias

            if torch.sum(current_weight != 0) > torch.numel(current_weight) * self.p:
                self.pruning(
                    ref_model.output.dense,
                    module,
                    original_output_tensor[:, 0, :],
                    output[:, 0, :],
                )

        handle = module.output.dense.register_forward_hook(ff2_hook)
        module.output.dense(current_intermediate_output)
        handle.remove()
        original_output = ref_model.output(
            original_intermediate_output, original_attn_output[0]
        )
        current_output = module.output(
            current_intermediate_output, current_attn_output[0]
        )

        return (original_output,), (current_output,)

    def propagate_pooler(
        self, ref_model, module, original_input_tensor, current_input_tensor
    ):
        original_first_token_tensor = original_input_tensor[:, 0]
        current_first_token_tensor = current_input_tensor[:, 0]
        original_output_tensor = ref_model.dense(original_first_token_tensor)

        def pooler_hook(module, input, output):
            # Get the original output from model
            current_weight, current_bias = module.weight, module.bias

            if torch.sum(current_weight != 0) > torch.numel(current_weight) * self.p:
                self.pruning(ref_model.dense, module, original_output_tensor, output)

        handle = module.dense.register_forward_hook(pooler_hook)
        module.dense(current_first_token_tensor)
        handle.remove()
        original_output_tensor = ref_model(original_input_tensor)
        current_output_tensor = module(current_input_tensor)

        return original_output_tensor, current_output_tensor

    def propagate_classifier(
        self, ref_model, module, original_input_tensor, current_input_tensor
    ):
        original_output_tensor = ref_model(original_input_tensor)

        def classifier_hook(module, input, output):
            # Get the original output from model
            current_weight, current_bias = module.weight, module.bias

            if torch.sum(current_weight != 0) > torch.numel(current_weight) * self.p:
                self.pruning(ref_model, module, original_output_tensor, output)

        handle = module.register_forward_hook(classifier_hook)
        current_output_tensor = module(current_input_tensor)
        handle.remove()
        return original_output_tensor, current_output_tensor

    def pruning(self, ref_model, module, original_output, output):
        # Get the shapes and original parameters (weights and biases) of the module

        current_weight, current_bias = module.weight.clone(), module.bias.clone()
        original_weight, original_bias = (
            ref_model.weight.clone(),
            ref_model.bias.clone(),
        )
        shape = current_weight.shape

        output_loss = output - original_output
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
            ~positive_loss_mask, positive_remove_mask, negative_remove_mask
        )

        current_weight[remove_mask] = 0

        all_zeros = ~current_weight.any(dim=1)
        current_bias[all_zeros] = 0

        set_parameters(module, current_weight, current_bias)
