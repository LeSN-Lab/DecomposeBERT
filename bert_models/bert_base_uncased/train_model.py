# In[]: Import Libraries
import os
import torch
from utils.model_utils.train_model import train_model


# In[]: Train model Examples
if __name__ == "__main__":
    file = os.path.realpath("__file__")
    root = os.path.dirname(file)
    model_name = "bert-base-uncased"
    load_path = os.path.join(root, "SDGclassfierModelConfig_bert_base")
    checkpoint_path = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """
    checkpoint_path = ['epoch_1.pt', 'epoch_2.pt', 'epoch_3.pt','epoch_4.pt']
    model, tokenizer, checkpoint = load_model(model_name, load_path, checkpoint_path)
    model = model.to(device)
    """

    # Train model
    epochs = 50

    train_model(
        model_name,
        load_path,
        device,
        epochs=epochs,
        batch_size=16,
        checkpoint_path=checkpoint_path,
        test=True,
    )

    """
    # Evaluate model
    model, tokenizer, checkpoint = load_model(model_name, load_path, i)
    model = model.to(device)
    train_dataloader, valid_dataloader, test_dataloader = load_sdg(tokenizer, batch_size=32)
    evaluate_model(model, test_dataloader, device)
    """
