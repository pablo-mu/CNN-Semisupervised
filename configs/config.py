import os
import torch
import numpy as np
import random

class Config:
    def __init__(self):
        # Training Hyperparameters
        self.epochs = 10
        self.batch_size = 64
        self.lr = 0.002
        self.n_labeled = 250
        self.train_iteration = 1024
        self.alpha = 0.75
        self.lambda_u = 75
        self.T = 0.5
        self.ema_decay = 0.999

        # Output Directory
        self.out_dir = 'results'
        os.makedirs(self.out_dir, exist_ok=True)

        # GPU Configuration
        self.gpu = '0'  # Use the first GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        self.use_cuda = torch.cuda.is_available()

        # Set random seed
        self.manualSeed = 0
        if self.manualSeed is None:
            self.manualSeed = random.randint(1, 10000)

        np.random.seed(self.manualSeed)
        torch.manual_seed(self.manualSeed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.manualSeed)

# Create a global instance
config = Config()
