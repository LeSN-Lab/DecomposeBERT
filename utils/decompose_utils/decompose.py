if __name__ == "__main__":
    import os
    import torch
    from utils.model_utils.load_model import load_model
    from utils.model_utils.modular_layers import initEncoderLayers

    file = os.path.realpath("__file__")
    root = os.path.dirname(file)
    model_name = "sadickam/sdg-classification-bert"
    load_path = os.path.join(root, "SDGclassfierModelConfig")

    checkpoint_path = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, tokenizer, checkpoint = load_model(model_name, load_path, checkpoint_path)
    model = model.to(device)

    # initEmbeddingLayers(model.bert.embeddings)
    initEncoderLayers(model.bert.encoder)
    # print(model.bert.encoder)
    # initEncoderLayers(model.bert.encoder)
