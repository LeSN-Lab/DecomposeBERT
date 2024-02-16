# In[]: Import Libraries
import torch
from utils.model_utils.train_model import train_model
from utils.model_utils.evaluate import evaluate_model
from utils.model_utils.load_model import load_classification_model
from utils.data_utils.load_dataset import load_sdg
from utils.model_utils.model_config import ModelConfig
from transformers import AutoConfig


# In[]: Train model Examples
if __name__ == "__main__":
    model_name = "bert-base-uncased"
    model_dir = "bert"
    data = "SDG"
    num_labels = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_name = None
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)

    model_config = ModelConfig(
        _model_name=model_name,
        _model_dir=model_dir,
        _data=data,
        _is_pretrained=True,
        _transformer_config=config,
        _checkpoint_name=checkpoint_name,
        _device=device,
    )

    # If you have a checkpoint, uncomment this
    """
    model_config.model_config.checkpoint_name = 'epoch_1.pt'
    """

    # Train model

    model, tokenizer, checkpoint = load_classification_model(model_config)
    print(model)
    # train_model(
    #     model_config=model_config,
    #     epochs=20,
    #     batch_size=16,
    #     lr=5e-5,
    #     test=True,
    # )

    # If you want to evaluate an accuracy of the model, uncomment this
    # Evaluate model
    """model_config.checkpoint_name = "best_model.pt"
    model, tokenizer, checkpoint = load_classification_model(model_config)
    train_dataloader, valid_dataloader, test_dataloader = load_sdg(
        model_config, tokenizer, batch_size=32
    )
    evaluate_model(model, model_config, test_dataloader)"""
