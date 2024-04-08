import torch


def safe_std(tensor, epsilon=None, unbiased=False, dim=None, keepdim=True):
    if tensor.numel():
        return nanstd(tensor, dim=dim, unbiased=unbiased, keepdim=keepdim)
    else:
        return torch.tensor(epsilon, dtype=tensor.dtype)


def nanstd(tensor, unbiased=False, dim=None, keepdim=True):
    mask = torch.isnan(tensor)
    n_obs = mask.logical_not().sum(dim=dim, keepdim=keepdim)
    mean = torch.nanmean(tensor, dim=dim, keepdim=keepdim)

    centered = tensor - mean
    centered = centered.masked_fill(mask, 0)
    sum_sq = torch.sum(centered ** 2, dim=dim, keepdim=keepdim)

    unbiased_factor = torch.where(n_obs > 1, n_obs - 1, n_obs)
    var = sum_sq / unbiased_factor

    std = torch.sqrt(var)
    if not keepdim:
        std = std.squeeze(dim)

    return std
