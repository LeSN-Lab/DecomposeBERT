# in[] Library
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os


# In[] Load/Save ModelConfig
def save_classification_model(model_config):
    model_name = model_config.model_name
    model_dir = model_config.model_dir
    model = None
    if model_config.is_pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=model_config.transformer_config
        )
    else:
        pass
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Save model and tokenizer
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def load_classification_model(model_config):
    load_path = model_config.model_dir
    # Check if the model exists
    if not model_config.is_downloaded:
        print(f"Directory {load_path} does not exist. Saving a new model here.")
        save_classification_model(model_config)

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(load_path)
    tokenizer = AutoTokenizer.from_pretrained(load_path)

    # load check point
    checkpoint = None
    if model_config.checkpoint_name is not None:
        checkpoint_path = os.path.join(
            model_config.train_dir, model_config.checkpoint_name
        )
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
