# In[]: Import Libraries
import torch
from utils.model_utils.train_model import train_model
from utils.paths import p


# In[]: Train model Examples
if __name__ == "__main__":
    model_path = "SDGclassfier(pre_trained)"
    model_name = "sadickam/sdg-classification-bert"
    p.set(model_path=model_path, model_name=model_name)

    load_path = p.get_model_dir()
    train_path = p.get_train_dir()
    checkpoint_path = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # If you have a checkpoint, uncomment this
    """
    checkpoint_path = 'epoch_4.pt'
    model, tokenizer, checkpoint = load_model(checkpoint_path)
    model = model.to(device)
    """

    # Train model
    train_model(
        device,
        epochs=20,
        batch_size=32,
        checkpoint_path=checkpoint_path,
        test=True,
    )

    # If you want to evaluate an accuracy of the model, uncomment this
    """
    # Evaluate model
    model, tokenizer, checkpoint = load_model(model_name, load_path, i)
    model = model.to(device)
    train_dataloader, valid_dataloader, test_dataloader = load_sdg(tokenizer, batch_size=32)
    evaluate_model(model, test_dataloader, device)
    """
