# In[]: Import Libraries
import os
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from utils.model_utils.load_model import load_classification_model
from utils.model_utils.evaluate import evaluate_model
from utils.dataset_utils.load_dataset import load_data
import torch
from transformers import Trainer


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(predictions, axis=1)
    accuracy = (predictions == labels).sum() / len(labels)
    loss = torch.nn.functional.cross_entropy(predictions, labels)
    return {"accuracy": accuracy, "loss": loss}

# In[]: Train model
def train_model(
    model,
    model_config,
    training_args,
    train_dataset,
    valid_dataset,
    tokenizer,
):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    return model
