# In[]: Import Libraries
import os
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from utils.model_utils.load_model import load_classification_model
from utils.model_utils.evaluate import compute_metrics
from utils.dataset_utils.load_dataset import load_data
import torch
from transformers import Trainer


# In[]: Train model
def train_model(
    model,
    training_args,
    train_dataset,
    valid_dataset,
):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    return trainer
