import torch.nn as nn
from utils.model_utils.modular_layers import ModularLayer, ModularLayerList


class ConcernIdentificationBert:
    def __init__(self, model, config):


    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # processing through embeddings layer
        embeddings = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        # processing through encoder layer
        encoder_output = self.encoder(embeddings, attention_mask=attention_mask)

        # processing through pooler layer
        pooled_output = self.pooler(encoder_output[0])

        # Optionally include dropout and classifier
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)

        return logits

    def get_classfier_weights(self, prev_hidden_states):

        return self.classifier(prev_hidden_states)