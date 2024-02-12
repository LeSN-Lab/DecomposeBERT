# In[]: Import Libraries
import torch
from utils.model_utils.train_model import train_model
from utils.model_utils.evaluate import evaluate_model
from utils.model_utils.modules import BertModel
from utils.data_utils.load_dataset import load_sdg
from utils.paths import p


# In[]: Train model Examples
if __name__ == "__main__":
    model_path = "SDGclassfier(pre_trained)"
    model_name = "sadickam/sdg-classification-bert"
    p.set(model_path=model_path, model_name=model_name)

    load_path = p.model_dir
    train_path = p.train_dir
    checkpoint_path = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # If you have a checkpoint, uncomment this
    """
    checkpoint_path = 'epoch_4.pt'
    """

    # Train model
    model, tokenizer, checkpoint = load_classification_model(checkpoint_path)
    train_model(
        device,
        epochs=20,
        batch_size=32,
        lr=5e-5,
        checkpoint_path=checkpoint_path,
        test=True,
    )

    # If you want to evaluate an accuracy of the model, uncomment this
    # Evaluate model
    """
    checkpoint_path = "best_model.pt"
    model, tokenizer, checkpoint = load_classification_model(checkpoint_path)
    train_dataloader, valid_dataloader, test_dataloader = load_sdg(tokenizer, batch_size=32)
    evaluate_model(model, test_dataloader, device)
    """
