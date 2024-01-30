# in[] Library
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os


# In[] Load/Save MultilingualModelConfig
def save_model(model_name, save_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return model, tokenizer


def load_model(model_name, load_path, checkpoint_path=None):
    if not os.path.isdir(load_path):
        model, tokenizer = save_model(model_name, load_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    checkpoint = None
    if not os.path.isdir("Models"):
        os.mkdir("Models")
    if checkpoint_path:
        model_path = os.path.join("Models", checkpoint_path)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["model_state_dict"])

    return model, tokenizer, checkpoint

