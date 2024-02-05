def initModularLayers(model):
    layers_dict = {}

    # Save Encoding layers
    if hasattr(model.bert, 'encoder'):
        for i, layer in enumerate(model.bert.encoder.layer):
            layers_dict[f'encoder_layer{i}'] = layer

    # Save Decoding layers
    if hasattr(model, 'decoder'):
        for i, layer in enumerate(model.bert.decoder.layer):
            layers_dict[f'decoder_layer{i}'] = layer

    # Save Embedding layers
    if hasattr(model.bert, 'embeddings'):
        layers_dict['input_embedding'] = model.bert.embeddings
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'embeddings'):
        layers_dict['output_embedding'] = model.decoder.embeddings
    if hasattr(model.bert, 'pooler'):
        layers_dict['pooler'] = model.bert.pooler
    return layers_dict

