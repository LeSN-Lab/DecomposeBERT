import torch
import os


def save_module(module, save_path, module_name):
    save_path = os.path.join(save_path, module_name)
    torch.save(module.state_dict(), save_path)
