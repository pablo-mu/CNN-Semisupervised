from torch.utils.tensorboard import SummaryWriter
import os

class Logger(SummaryWriter):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        os.makedirs(log_dir, exist_ok = True)
    
    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
    
    def add_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)
        
    def add_graph(self, model, input_tensor):
        self.writer.add_graph(model, input_tensor)
    
    def close(self):
        self.writer.close()