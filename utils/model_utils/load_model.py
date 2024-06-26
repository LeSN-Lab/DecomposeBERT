# in[] Library
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.model_utils.load_tokenizer import load_tokenizer
import torch
import os


# In[] Load/Save ModelConfig
def save_classification_model(model_config):
    model_name = model_config.model_name
    model_dir = model_config.config_dir
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=model_config.transformer_config
    )

    # Save model and tokenizer
    model.save_pretrained(model_dir)
    return model


def load_classification_model(model_config, train_mode):
    if train_mode:
        load_path = model_config.train_dir
    else:
        load_path = model_config.module_dir
    config_path = model_config.config_dir
    # Check if the model exists
    if not model_config.is_downloaded:
        print(f"Directory {config_path} does not exist.")
        print("Saving a new model here.")
        model = save_classification_model(model_config)
        tokenizer = load_tokenizer(model_config)
    else:
        print(f"Directory {config_path} exists.")
        print("Loading the model.")
        model = AutoModelForSequenceClassification.from_pretrained(config_path)
        tokenizer = load_tokenizer(model_config)
    model_config.is_downloaded = True

    # load check point
    checkpoint = None
    if model_config.checkpoint_name is not None:
        checkpoint_path = os.path.join(load_path, model_config.checkpoint_name)
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=model_config.device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                model.to(model_config.device)
            else:
                print(
                    "Checkpoint structure is unrecognized. Check the keys or save format."
                )
        else:
            print(f"Checkpoint path {checkpoint_path} does not exist.")
            checkpoint = None
    else:
        model.to(model_config.device)
    return model, tokenizer, checkpoint
