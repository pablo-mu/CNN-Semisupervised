import os
#import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import models.cnn as cnn
import utils.mixmatch as mm
import data.data as data

epochs = 10
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
    
    for batch_idx in range(train_iteration):
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
        
        # Transform labels to one-hot
        
        targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1,1).long(), 1)

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
                                   logits_u, mixed_target[batch_size:], epoch + batch_idx/train_iteration, epochs)
        loss = Lx + weight * Lu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        total_loss += loss.item()

    return total_loss / train_iteration


def validate(loader, model, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / total, 100. * correct / total

def main():
    global best_acc
    
    os.makedirs(out_dir, exist_ok=True)
    
    transform_train = transforms.Compose([
        data.RandomPandandCrop(32),
        data.RandomFlip(),
        data.ToTensor()
    ])
    
    transform_val = transforms.Compose([data.ToTensor()])
    print('==> Preparing data..')
    train_labeled_set, train_unlabeled_set, val_set, test_set = data.get_cifar10('./data', n_labeled, transform_train=transform_train, transform_val=transform_val, download=True)

    labeled_loader = DataLoader(train_labeled_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    unlabeled_loader = DataLoader(train_unlabeled_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, optimizer, EMA and loss
    model = create_model()
    ema_model = create_model(ema=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    semi_loss = mm.SemiLoss()
    ema_optimizer = mm.WeightEMA(model, ema_model, lr, ema_decay)

    # For evaluation we use the standard cross entropy loss
    ce_loss = nn.CrossEntropyLoss()

    writer = SummaryWriter(out_dir)
    best_val_acc = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch:3d}:")
        t_loss = train(labeled_loader, unlabeled_loader, model, optimizer, ema_optimizer, semi_loss, epoch)
        v_loss, v_acc = validate(val_loader, ema_model, ce_loss)
        t_loss_val, t_acc = validate(test_loader, ema_model, ce_loss)

        writer.add_scalar('Loss/Train', t_loss, epoch)
        writer.add_scalar('Loss/Val', v_loss, epoch)
        writer.add_scalar('Loss/Test', t_loss_val, epoch)
        writer.add_scalar('Accuracy/Val', v_acc, epoch)
        writer.add_scalar('Accuracy/Test', t_acc, epoch)

        print(f"Epoch {epoch:3d}: Train Loss {t_loss:.4f} | Val Loss {v_loss:.4f}, Val Acc {v_acc:.2f}% | Test Acc {t_acc:.2f}%")
        
        # Save the best model (based on validation accuracy)
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(out_dir, 'best_model.pth'))

    writer.close()

if __name__ == '__main__':
    main()
