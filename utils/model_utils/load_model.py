# in[] Library
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    BertConfig,
)
import torch
import os
from utils.paths import p


# In[] Load/Save MultilingualModelConfig
def save_model(num_labels=None):
    if p.model_name == "bert-base-uncased":
        config = BertConfig.from_pretrained(p.model_name, num_labels=num_labels)
        model = BertForSequenceClassification(config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(p.model_name)
    tokenizer = AutoTokenizer.from_pretrained(p.model_name)
    model.save_pretrained(p.model_dir)
    tokenizer.save_pretrained(p.model_dir)
    return model, tokenizer


def load_model(checkpoint_path=None, num_labels=None):
    load_path = p.get_model_path()
    if not p.check_dir(load_path):
        print(f"Directory {load_path} does not exist. Saving a new model there.")
        model, tokenizer = save_model(num_labels)
    else:
        if p.model_name == "bert-base-uncased" and num_labels is not None:
            config = BertConfig.from_pretrained(load_path, num_labels=num_labels)
            model = BertForSequenceClassification.from_pretrained(
                load_path, config=config, ignore_mismatched_sizes=True
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(load_path)
        tokenizer = AutoTokenizer.from_pretrained(load_path)

    checkpoint = None
    train_path = p.get_train_path()
    if checkpoint_path:
        model_path = os.path.join(train_path, checkpoint_path)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            if 'model_state_dict' in checkpoint:
                if num_labels is not None:
                    model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    return model, tokenizer, checkpoint
