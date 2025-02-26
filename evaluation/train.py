import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import cnn
from utils import mixmatch as mm
from data import datasets

epochs = 1024
batch_size = 64
lr = 0.002
n_labeled = 250
train_iteration = 1024
alpha = 0.75
lambda_u = 75
T = 0.5
ema_decay = 0.999

out_dir = 'results'
gpu = '0'  # Use the first GPU
os.environ['CUDA_VISIBLE_DEVICES'] = gpu  # Restrict visible GPUs
use_cuda = torch.cuda.is_available()

manualSeed = 0
if manualSeed is None:
    manualSeed = random.randint(1, 10000)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(manualSeed)

def create_model(ema = False):
    model = cnn.SimpleCNN(num_classes = 10).cuda()
    if ema:
        for param in model.parameters():
            param.detach_()    
    return model

def train(labeled_loader, unlabeled_loader, model, optimizer, ema_optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    
    for _ in range(train_iteration):
        try:
            inputs_x, targets_x = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            inputs_x, targets_x = next(labeled_iter)

        try:
            (inputs_u, inputs_u2), _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            (inputs_u, inputs_u2), _ = next(unlabeled_iter)

        batch_size = inputs_x.size(0)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda()
            inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()
        
        # Compute guessed labels for unlabeld samples
        with torch.no_grad():
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            targets_u = mm.sharpen(p, T)
            targets_u = targets_u.detach()
        
        # MixUp
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)
        l_mix = np.random.beta(alpha, alpha)
        l_mix = max(l_mix, 1 - l_mix)
        idx = torch.randperm(all_inputs.size(0))
        mixed_input = l_mix * all_inputs + (1 - l_mix) * all_inputs[idx]
        mixed_target = l_mix * all_targets + (1 - l_mix) * all_targets[idx]
        
       # Interleave for proper batch norm statistics. 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = mm.interleave(mixed_input, batch_size)
        logits = [model(mixed_input[0])]
        for inp in mixed_input[1:]:
            logits.append(model(inp))
        logits = mm.interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0) 
        
        # Compute losses and update
        Lx, Lu, weight = criterion(logits_x, mixed_target[:batch_size],
                                   logits_u, mixed_target[batch_size:], epoch)
        loss = Lx + weight * Lu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        total_loss += loss.item()

    return total_loss / train_iteration


