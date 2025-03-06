import torch
import torch.nn.functional as F 
import numpy as np
import torchvision
from tqdm import tqdm
from utils.misc import AverageMeter

import utils.mixmatch as mm

class MixMatchTrainer:
    def __init__(self, model, ema_model, optimizer, ema_optimizer, criterion, config, writer, device):
        self.model = model
        self.ema_model = ema_model
        self.optimizer = optimizer
        self.ema_optimizer = ema_optimizer
        self.criterion = criterion
        self.config = config
        self.writer = writer
        self.device = device
        self.step = 0
        
    def train_epoch(self, labeled_loader, unlabeled_loader, epoch):
        self.model.train()
        total_loss = 0
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        progress_bar = tqdm(range(self.config.train_iteration), 
                            desc = f"Epoch {epoch + 1}/{self.config.epochs}")
        
        for batch_idx in progress_bar:
            inputs_x, targets_x = self._get_labeled_batch(labeled_iter, labeled_loader)
            (inputs_u, inputs_u2), _ = self._get_unlabeled_batch(unlabeled_iter, unlabeled_loader)
            
            self.batch_size = inputs_x.size(0)
            
            targets_x = torch.zeros(self.batch_size, 10).scatter_(1, targets_x.view(-1,1).long(), 1)
            
            if self.config.use_cuda:
                inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda()
                inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()
            
            mixed_input, mixed_target = self._mixmatch_forward(inputs_x, targets_x, inputs_u, inputs_u2)
            
            logits_x, logits_u = self._interleave_forward(mixed_input)
            
            # Compute losses and update
            Lx, Lu, weight = self.criterion(logits_x, mixed_target[:self.batch_size],
                                    logits_u, mixed_target[self.batch_size:], epoch + batch_idx/self.config.train_iteration,
                                    self.config.epochs)
            loss = Lx + weight * Lu
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema_optimizer.step()

            total_loss += loss.item()
            self.step += 1
        
            if batch_idx % 100 == 0:
                self._log_parameters()
                self._log_gradients()
                self._log_learning_rates()
            
            progress_bar.set_postfix({'Loss': total_loss / (batch_idx + 1)})
        
        return total_loss / self.config.train_iteration
    
    def validate(self, loader, model, criterion, mode = 'Val'):
        """Validate the model on a given dataset. 

        Args:
            loader (DataLoader): DataLoader for validation/test set
            model (nn.Module): Model to evaluate
            epoch (int): Current epoch number (for logging)
            mode (str): Mode of validation. Either 'Val' or 'Test'
        """
        model.eval()
        losses = AverageMeter()
        acc = AverageMeter()
        
    
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                if self.config.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                
                # Compute output
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                losses.update(loss.item(), inputs.size(0))
                
                # Compute accuracy
                _, predicted = outputs.max(1) # get the index of the max log-probability
                total = targets.size(0) # get the size of the batch
                correct = predicted.eq(targets).sum().item() # get the number of correct predictions
                acc.update(correct, total)
            return losses.avg, acc.avg
                
                
                    
    def _log_images(self, images, epoch, tag):
        grid = torchvision.utils.make_grid(images[:8])
        self.writer.add_image(tag, grid, epoch)

    def _log_parameters(self):
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(f"Parameters/{name}", param, self.step)

    def _log_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"Gradients/{name}", param.grad, self.step)

    def _log_learning_rates(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f"LearningRate/group_{i}", param_group['lr'], self.step)
    
    
    def _interleave_forward(self, mixed_input):
        mixed_input = list(torch.split(mixed_input, self.batch_size))
        mixed_input = mm.interleave(mixed_input, self.batch_size)
        logits = [self.model(mixed_input[0])]
        for inp in mixed_input[1:]:
            logits.append(self.model(inp))
        logits = mm.interleave(logits, self.batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0) 
        return logits_x, logits_u
        
            
    
    
    def _mixmatch_forward(self, inputs_x, targets_x, inputs_u, inputs_u2):
        """
        Performs forward pass for MixMatch.
        
        Args:
            inputs_x: Labeled data inputs.
            targets_x: Labeled data targets.
            inputs_u: First version of unlabeled inputs.
            inputs_u2: Second version of unlabeled inputs (augmented).
        
        Returns:
            logits_x: Logits for labeled data.
            logits_u: Logits for unlabeled data.
        """
        
        with torch.no_grad():
            outputs_u = self.model(inputs_u)
            outputs_u2 = self.model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            targets_u = mm.sharpen(p, self.config.T)
            targets_u = targets_u.detach()
        
        # MixUp
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)
        l_mix = np.random.beta(self.config.alpha, self.config.alpha)
        l_mix = max(l_mix, 1 - l_mix)
        idx = torch.randperm(all_inputs.size(0))
        mixed_input = l_mix * all_inputs + (1 - l_mix) * all_inputs[idx]
        mixed_target = l_mix * all_targets + (1 - l_mix) * all_targets[idx]
        return mixed_input, mixed_target
    
    
    def _get_labeled_batch(self, labeled_iter, labeled_loader):
        """Fetches the next batch of unlabeled data.
        If the labeled data loader is exhausted, it restarts the iterator.
    

        Args:
            labeled_iter: Iterator fot the labeled data loader.
            labeled_loader: Labeled data loader. 
        
        Returns:
            inputs_x: Labeled data inputs.
            targets_x: Labeled data targets.
        """
        try:
            inputs_x, targets_x = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            inputs_x, targets_x = next(labeled_iter)
        
        return inputs_x, targets_x
    
    def _get_unlabeled_batch(self, unlabeled_iter, unlabeled_loader):
        """
        Fetches the next batch of unlabeled data.
        If the unlabeled data loader is exhausted, it restarts the iterator.
        
        Args:
            unlabeled_iter: Iterator for the unlabeled data loader.
            unlabeled_loader: Unlabeled data loader.
        
        Returns:
            inputs_u: First version of unlabeled inputs.
            inputs_u2: Second version of unlabeled inputs (augmented).
        """
        try:
            (inputs_u, inputs_u2), _ = next(unlabeled_iter)
        except StopIteration:
            # Restart the iterator if the data loader is exhausted
            unlabeled_iter = iter(unlabeled_loader)
            (inputs_u, inputs_u2), _ = next(unlabeled_iter)
        
        return (inputs_u, inputs_u2), _