# In[]: Import Libraries
import torch
from utils.model_utils.train_model import train_model
from utils.model_utils.evaluate import evaluate_model
from utils.model_utils.load_model import load_classification_model
from utils.data_utils.load_dataset import load_sdg
from utils.model_utils.model_config import ModelConfig


# In[]: Train model Examples
if __name__ == "__main__":
    model_dir = "SDGclassfier(bert_base_uncased)"
    model_name = "bert-base-uncased"
    data = "SDG"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_name = None

    # If you have a checkpoint, uncomment this
    """
    model_config.checkpoint_name = 'epoch_1.pt'
    """

    # Train model
    """model_config = ModelConfig(
        _model_name=model_name,
        _model_type="berts",
        _checkpoint_name=checkpoint_name,
        _num_labels=16,
        _device=device,
    )
    train_model(
        model_config=model_config,
        epochs=20,
        batch_size=16,
        lr=5e-5,
        test=True,
    )"""

    # If you want to evaluate an accuracy of the model, uncomment this
    # Evaluate model
    checkpoint_name = "best_model.pt"
    model_config = ModelConfig(
        _model_name=model_name,
        _data=data,
        _model_type="berts",
        _checkpoint_name=checkpoint_name,
        _num_labels=16,
        _device=device,
    )
    model, tokenizer, checkpoint = load_classification_model(model_config)
    train_dataloader, valid_dataloader, test_dataloader = load_sdg(
        model_config, tokenizer, batch_size=32
    )
    evaluate_model(model, model_config, test_dataloader)
