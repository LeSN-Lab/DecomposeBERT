# in[] Library
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    BertConfig,
)
import torch
import os


# In[] Load/Save MultilingualModelConfig
def save_model(model_name, save_path, num_labels=None):
    if model_name == "bert-base-uncased":
        config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
        model = BertForSequenceClassification(config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    return model, tokenizer


def load_model(model_name, load_path, checkpoint_path=None, num_labels=None):
    if not os.path.isdir(load_path):
        print(f"Directory {load_path} does not exist. Saving a new model there.")
        model, tokenizer = save_model(model_name, load_path, num_labels)
    else:
        if model_name == "bert-base-uncased" and num_labels is not None:
            config = BertConfig.from_pretrained(load_path, num_labels=num_labels)
            model = BertForSequenceClassification.from_pretrained(
                load_path, config=config, ignore_mismatched_sizes=True
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(load_path)
        tokenizer = AutoTokenizer.from_pretrained(load_path)

    checkpoint = None
    if not os.path.isdir("Models"):
        os.mkdir("Models")
    if checkpoint_path:
        model_path = os.path.join("Models", checkpoint_path)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            if 'model_state_dict' in checkpoint:
                if num_labels is not None:
                    model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    return model, tokenizer, checkpoint
