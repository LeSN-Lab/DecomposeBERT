import torch


class Tensor(torch.Tensor):
    def __init__(self, data):
        super().__init__(data)

    @staticmethod
    def to_tensor(data, device, dtype=torch.long):
        return torch.tensor(data, dtype=dtype).to(device)
