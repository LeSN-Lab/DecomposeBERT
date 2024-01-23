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
    import os
    file = os.getcwd()
    root = os.path.dirname(file)
    model_name = "sadickam/sdg-classification-bert"
    save_path = os.path.join(root, "SDGclassfierModelConfig")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.isdir(save_path):
        model, tokenizer = load_model(model_name, save_path)
    else:
        model, tokenizer = save_model(model_name, save_path)
    model = model.to(device)

# In[]:
    print(model.parameters())