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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer


def load_model(model_name, load_path):
    if not os.path.isdir(load_path):
        model, tokenizer = save_model(model_name, load_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(model)

    return model, tokenizer
