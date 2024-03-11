import torch
from utils.model_utils.modular_layers import get_extended_attention_mask, transpose_for_scores

class ConcernIdentificationBert:
    def __init__(self):
        self.positive_sample = True
        self.flag = False
        self.attn_probs = None
        self.num_attention_heads = 12
        self.attention_head_size = 64

    def propagate(self, module, input_tensor, attention_mask, positive_sample=True):
        # propagate input tensor to the module
        self.positive_sample = positive_sample
        output_tensor = self.propagate_embeddings(module.embeddings, input_tensor)
        output_tensor = self.propagate_encoder(module.encoder, output_tensor, attention_mask)
        output_tensor = self.propagate_pooler(module.pooler, output_tensor)
        output_tensor = module.dropout(output_tensor)
        output_tensor = self.propagate_classifier(module.classifier, output_tensor)
        return output_tensor

    def propagate_embeddings(self, module, input_tensor):
        def embeddings_hook(module, input, output):
            pass

        handle = module.register_forward_hook(embeddings_hook)
        output_tensor = module(input_tensor)
        handle.remove()
        return output_tensor

    def propagate_encoder(self, module, input_tensor, attention_mask):
        attention_mask = get_extended_attention_mask(attention_mask)
        for i, encoder_block in enumerate(module.encoder_blocks):
            block_outputs = self.propagate_encoder_block(encoder_block, input_tensor, attention_mask, None)
            input_tensor = block_outputs[0]
        return input_tensor

    def propagate_encoder_block(self, module, input_tensor, attention_mask, head_mask=None):

        def ff2_hook(module, input, output):
            output_features, input_features = module.shape
            current_weight, current_bias = module.weight, module.bias
            original_weight, original_bias = module.get_parameters()

            original_output = module.layer(input[0])

            for s in range(original_output.shape[1]):
                temp = original_output[:, s, :]
                temp = temp.squeeze(0)
                temp = temp - torch.mean(temp)
                for i in range(output_features):
                    if self.positive_sample:
                        if temp[i] <= 0:
                            current_weight[i, :] = 0
                            current_bias[i] = 0
                        else:
                            if self.flag:
                                current_weight[i, :] = original_weight[i, :]
                                current_bias[i] = original_bias[i]
                            else:
                                current_row = current_weight[i]
                                original_row = original_weight[i]

                                mask = original_row > 0
                                tmp = torch.max(current_row, original_row)
                                tmp[tmp < 0] = 0
                                updated_row = torch.where(
                                    mask,
                                    torch.min(current_row, original_row),
                                    tmp
                                )
                                current_weight[i] = updated_row
                                current_bias[i] = original_bias[i]
                    #
                    # else:
                    #     if temp[i] > 0:
                    #         current_weight[i, :] = original_weight[i, :]
                    #         current_bias[i] = original_bias[i]
                    if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.1:
                        self.flag = True
                    else:
                        self.flag = False

            module.set_parameters(current_weight, current_bias)

        attn_outputs = module.attention(input_tensor, attention_mask, head_mask)
        self.attn_probs = module.attention.self_attention.attention_probs

        intermediate_output = module.feed_forward1(attn_outputs)
        handle = module.feed_forward2.dense.register_forward_hook(ff2_hook)
        layer_output = module.feed_forward2(intermediate_output, attn_outputs)
        handle.remove()
        return (layer_output,)

    def propagate_attention_module(self, module, input_tensor, attention_mask, head_mask):
        # def attention_hook(module, input, output):
        #     print(module.transpose_for_scores(module.key(input[0])).shape)
        #     print(output.shape)
        #     print(module.attention_probs.shape)
        #     output_t = module.transpose_for_scores(output)
        #     prob1 = module.attention_probs[:,:,0,:]
        #     print(torch.sum(module.attention_probs))
        #     x = torch.matmul(output_t.transpose(-1,-2), prob1.unsqueeze(-1))
        #     x = x.squeeze(-1)
        #     print(x.shape)
        #     # if len(x) > 1:raise "batch size is greater than one."
        #     # for i in range(module.num_attention_heads):
        #     #     print(f"p{torch.sum(x[0,i,:] > 0) /module.attention_head_size}")
        #     #
        #     # print(1111111111)
        #
        #
        # def output_hook(module, input, output):
        #     pass

        # handle = module.self_attention.register_forward_hook(attention_hook)
        self_outputs = module.self_attention(input_tensor, attention_mask, head_mask)
        # handle.remove()
        # handle = module.output.register_forward_hook(output_hook)
        attention_output = module.output(self_outputs[0], input_tensor)
        # handle.remove()
        return attention_output

    def propagate_pooler(self, module, input_tensor):
        def pooler_hook(module, input, output):
            # Get the shapes and original parameters (weights and biases) of the Pooler module
            output_features, input_features = module.shape
            current_weight, current_bias = module.weight, module.bias
            original_weight, original_bias = module.get_parameters()

            # Squeeze the output tensor to remove the batch dimension
            original_output = module.layer(input[0])
            temp = original_output.squeeze(0)

            for i in range(output_features):
                if self.positive_sample:
                    if temp[i] <= 0:
                        current_weight[i, :] = 0
                        current_bias[i] = 0
                    else:
                        if self.flag:
                            current_weight[i, :] = original_weight[i, :]
                            current_bias[i] = original_bias[i]
                        else:
                            current_row = current_weight[i]
                            original_row = original_weight[i]

                            mask = original_row > 0
                            tmp = torch.max(current_row, original_row)
                            tmp[tmp < 0] = 0
                            updated_row = torch.where(
                                mask,
                                torch.min(current_row, original_row),
                                tmp
                            )
                            current_weight[i] = updated_row
                            current_bias[i] = original_bias[i]

                # else:
                #     if temp[i] > 0:
                #         current_row = current_weight[i]
                #         original_row = original_weight[i]
                #
                #         mask = original_row < 0
                #         tmp = torch.min(current_row, original_row)
                #         updated_row = torch.where(
                #             mask,
                #             tmp,
                #             torch.max(current_row, original_row)
                #         )
                #         current_weight[i] = updated_row
                #         current_bias[i] = original_bias[i]

                if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.1:
                    self.flag = True
                else:
                    self.flag = False

            module.set_parameters(current_weight, current_bias)

        first_token_tensor = input_tensor[:, 0]
        # handle = module.dense.register_forward_hook(pooler_hook)
        output_tensor = module.dense(first_token_tensor)
        # handle.remove()
        output_tensor = module.activation(output_tensor)
        return output_tensor

    @staticmethod
    def propagate_classifier(module, input_tensor):

        # handle = module.register_forward_hook(classifier_hook)
        output_tensor = module(input_tensor)
        # handle.remove()
        return output_tensor

    def calc_central_activation_tendency(self, weight):
        abs_values = torch.abs(weight)
        return torch.mean(abs_values)



    # def positive_hook(self, module, input, output):
    #     target_layer = module.dense
    #     print(target_layer.shape)
    #     output_features, input_features = target_layer.shape
    #     current_weight, current_bias = target_layer.weight, target_layer.bias
    #     original_weight, original_bias = target_layer.get_parameters()
    #     temp = output.squeeze(0)
#         for i in range(output_features):
#             print(len(temp[:,i]))
#             if temp[:, i] <= 0:
#                 current_weight[i,:] = 0
#                 current_bias[i,:] = 0
#             else:
#                 if self.positive_sample:
#                     current_weight[i,:] = original_weight[i,:]
#                     current_bias[i] = original_bias[i]
#                 else:
#                     current_row = current_weight[i]
#                     original_row = original_weight[i]
#
#                     mask = original_row > 0
#
#                     updated_row = torch.where(
#                         mask,
#                         torch.min(current_row, original_row),
#                         torch.max(current_row, original_row)
#                     )
#                     current_weight[i] = updated_row

            # target_layer.set_parameters(current_weight, current_bias)
