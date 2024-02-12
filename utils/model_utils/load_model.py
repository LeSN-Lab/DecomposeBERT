# in[] Library
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2Model,
    GPT2Tokenizer,
)
import torch
import os
from utils.model_utils.constants import ArchitectureType


# In[] Load/Save ModelConfig
def save_classification_model(model_config):
    model = None
    tokenizer = None
    model_name = model_config.model_name
    model_dir = model_config.model_dir

    if model_config.model_type == ArchitectureType.Bert:
        config = AutoConfig.from_pretrained(model_name, num_labels=model_config.num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Save model and tokenizer
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    elif model_config.model_type == ArchitectureType.Transformer:
        pass
    elif model_config.model_type == ArchitectureType.GPT:
        pass
    return model, tokenizer


def load_classification_model(model_config):
    load_path = model_config.model_dir
    model = None
    tokenizer = None

    # Check if the model exists
    if not model_config.is_downloaded:
        print(f"Directory {load_path} does not exist. Saving a new model here.")
        model, tokenizer = save_classification_model(model_config)

    # load model
    if model_config.model_type == ArchitectureType.Bert:
        model = AutoModelForSequenceClassification.from_pretrained(load_path)
        tokenizer = AutoTokenizer.from_pretrained(load_path)
    elif model_config.model_type == ArchitectureType.Transformer:
        pass
    elif model_config.model_type == ArchitectureType.GPT:
        pass

    # load check point
    checkpoint = None
    if model_config.checkpoint_name is not None:
        checkpoint_path = os.path.join(model_config.train_dir, model_config.checkpoint_name)
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=model_config.device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                print("Checkpoint structure is unrecognized. Check the keys or save format.")
        else:
            print(f"Checkpoint path {checkpoint_path} does not exist.")
            checkpoint = None
    else:
        model.to(model_config.device)
    return model, tokenizer, checkpoint
