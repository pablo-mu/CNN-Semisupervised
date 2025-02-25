'''
Basic CNN model to test the end-to-end pipeline of the project. However, the purpose of the project
is to implement a more complex model like ResNet, VGG, etc.
'''

import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Primera capa convolucional: de 3 canales a 32 filtros
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm para estabilizar y acelerar el entrenamiento
        
        # Segunda capa convolucional: de 32 a 64 filtros
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Capa de pooling para reducir dimensiones
        self.pool = nn.MaxPool2d(2, 2)
        
        # Capa totalmente conectada. 
        # Suponiendo imágenes de entrada de 32x32, tras dos operaciones de pooling se reduce a 8x8.
        self.fc = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        # Primera capa: convolución, BatchNorm, activación ReLU y pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Segunda capa: convolución, BatchNorm, activación ReLU y pooling
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Aplanar la salida para la capa FC
        x = x.view(x.size(0), -1)
        
        # Capa totalmente conectada
        x = self.fc(x)
        return x


        