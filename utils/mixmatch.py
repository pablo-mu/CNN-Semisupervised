import torch
import torch.nn.functional as F
import numpy as np

def sharpen(prob, T):
    p_power = prob ** (1 / T)
    return p_power / p_power.sum(dim=1, keepdim=True)

