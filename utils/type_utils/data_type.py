import torch


def safe_std(tensor, dim=None, epsilon=1e-5, keepdim=True):
    std = torch.std(tensor, dim=dim, unbiased=False, keepdim=keepdim)
    return torch.max(std, torch.tensor(epsilon).to(std.device))