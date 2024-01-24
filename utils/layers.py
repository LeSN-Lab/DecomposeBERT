import torch
import torch.nn as nn
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_


class MLPlayers(nn.Module):
    def __init__(self, layers):
        self.layers = layers
