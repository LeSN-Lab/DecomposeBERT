import torch


def safe_std(tensor):
    numel = tensor.numel()
    if numel <= 1:
        return torch.tensor(0.0, device=tensor.device)
    else:
        return torch.std(tensor)