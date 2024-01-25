# In[]
import os
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from utils.modelConfig import save_model, load_model
from utils.load_dataset import load_sdg
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def decompose_multi_head_attention_layer():
    pass


def decompose_position_FCN():
    pass


# In[]
if __name__ == '__main__':
    file = os.path.realpath('__file__')
    root = os.path.dirname(file)
    model_name = "sadickam/sdg-classification-bert"
    load_path = os.path.join(root, "SDGclassfierModelConfig")

    model, tokenizer = load_model(model_name, load_path)
    model = model.to(device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(model)
    # positiveConcern = initModularLayers(model)
    # negativeConcern = initModularLayers(model)

    # positiveConcern = initModularLayers(model.layers)
    # labs = range(0, 16)
    # for j in labs:
    #     model
