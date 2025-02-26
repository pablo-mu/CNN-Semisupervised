import torch
import torch.nn.functional as F
import numpy as np

class WeightEMA(object):
    def __init__(self, model, ema_model, lr, ema_decay):
        self.model = model
        self.ema_model = ema_model
        self.alpha = ema_decay
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr
        
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.data.copy_(param.data)
        
    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                #Custom weight decay
                param.mul_(1-self.wd)
                

def interleave_offsets(batch, nu):
    """Calculate the inidices to split a batch into several parts that are as balanced as possible.

    Args:
        batch (int): The batch size.
        nu (_type_): Number of parts to split the batch into.
    """
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    """Interleave labeled and unlabeled samples.

    Args:
        xy ( list of torch.Tensor): A list of tensors to interleave.
        batch (int): The batch size.

    Returns:
        list of torch.Tensor: The interleaved tensor.
    """
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]
  
    


def sharpen(prob, T):
    p_power = prob ** (1 / T)
    return p_power / p_power.sum(dim=1, keepdim=True)

def linear_rumpup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
        

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, epochs):
        probs_u = torch.softmax(outputs_u, dim = 1)
        
        # Supervised and unsupervised loss
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rumpup(epoch, epochs)
