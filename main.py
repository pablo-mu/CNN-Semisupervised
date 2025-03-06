import os
import os
#import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import utils.mixmatch as mm
import data.data as data

from models.create_model import create_model
from trainers.mixmatch_trainer import MixMatchTrainer
from utils.logger import Logger
from configs.config import config

CHECKPOINT_PATH = 'results/checkpoint.pth'

def save_checkpoint(epoch, model, ema_model, optimizer, ema_optimizer, best_acc):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'ema_optimizer_state_dict': ema_optimizer.state_dict(),
        'best_acc': best_acc
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Checkpoint saved at epoch {epoch}")
    
def load_checkpoint(model, ema_model, optimizer, ema_optimizer):
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #ema_optimizer.load_state_dict(checkpoint['ema_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch, best_acc
    return 0, 0  # Start from scratch if no checkpoint

def main():
    
    writer = Logger(config.out_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    transform_train = transforms.Compose([
        data.RandomPandandCrop(32),
        data.RandomFlip(),
        data.ToTensor()
    ])
    
    transform_val = transforms.Compose([data.ToTensor()])
    
    # Data
    print('==> Preparing data..')
    train_labeled_set, train_unlabeled_set, val_set, test_set = data.get_cifar10('./data', config.n_labeled, transform_train=transform_train, transform_val=transform_val, download=True)

    labeled_loader = DataLoader(train_labeled_set, batch_size=config.batch_size, shuffle=True, num_workers=2, drop_last=True)
    unlabeled_loader = DataLoader(train_unlabeled_set, batch_size=config.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=2)
    
    # Model, optimizer, EMA and Loss
    model = create_model(ema = False, model = 'wideresnet')
    ema_model = create_model(ema = True, model = 'wideresnet')
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    semi_loss = mm.SemiLoss()
    ema_optimizer = mm.WeightEMA(model, ema_model, config.lr, config.ema_decay)
    
    # For evaluation we use the standard cross entropy loss
    ce_loss = nn.CrossEntropyLoss()
    
    trainer = MixMatchTrainer(
        model = model,
        ema_model = ema_model,
        optimizer = optimizer,
        ema_optimizer = ema_optimizer,
        criterion = semi_loss,
        config = config,
        writer = writer,
        device = device
    )
    
    start_epoch, best_acc = load_checkpoint(model, ema_model, optimizer, ema_optimizer)
    
    for epoch in range(start_epoch, config.epochs):
        print(f'Epoch {epoch }')
        
        # Train
        train_loss = trainer.train_epoch(labeled_loader, unlabeled_loader, epoch)
        writer.add_scalar('Train/loss', train_loss, epoch +1)
        
        # Validate
        val_loss, val_acc = trainer.validate(val_loader, ema_model, ce_loss)
        writer.add_scalar('Loss/Validation', val_loss, epoch +1)
        writer.add_scalar('Accuracy/Vaidation', val_acc, epoch +1)
        
        # Test
        test_loss, test_acc = trainer.validate(test_loader, ema_model, ce_loss, 'Test')
        writer.add_scalar('Loss/Test', test_loss, epoch +1)
        writer.add_scalar('Accuracy/Test', test_acc, epoch +1)
        
        # Graphs and histograms
        
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(config.out_dir, 'best_model.pth'))
        
        # Save checkpoint
        save_checkpoint(epoch, model, ema_model, optimizer, ema_optimizer, best_acc)
    
    writer.close()

if __name__ == '__main__':
    main()