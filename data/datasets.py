'''
CIFAR-10 dataset
'''

from torchvision import datasets
import torch
import numpy as np

def pad(x, border = 4):
    return np.pad(x, ((border, border), (border, border), (0, 0)), mode = 'reflect')

class RandomPandandCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)) # assert if output_size is an instance of int or tuple
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, x):
        x = pad(x,4)
        h, w = x.shape[1:]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        x = x[:, top: top + new_h, left: left + new_h]
        
        return x

class RandomFlip(object):
    """Flip the image randomly and horizontally
    Args:
        object (_type_): Image
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]
        return x
        

def normalize(x, mean, std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean
    x *= 1.0 / std
    return x

def transpose(x, source = 'NHWC', target = 'NCHW'):
    return x.transpose([source.index(d) for d in target])

class ToTensor(object):
    """ Transform an Image to Tensor

    Args:
        object (_type_): Image

    Returns:
        torch.Tensor: Image as Tensor    
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x
    

class CIFAR10_labeled(datasets.CIFAR10):
    def __init__(self, root, indexs = None, train = True, transform = None,
                 target_transform = None, download = False):
        super(CIFAR10_labeled, self).__init__(root, train = train, transform = transform,
                                              target_transform = target_transform, download = download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
    
    def __getitem__(self, index):
        """_summary_

        Args:
            index (int): Index
        
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target


class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])

    