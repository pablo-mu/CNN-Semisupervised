'''
CIFAR-10 dataset
'''

from torchvision import datasets
import torch
import numpy as np

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_mean_and_std(dataset, train_labeled_idxs, train_unlabeled_idxs):
    
    all_idxs = np.concatenate([train_labeled_idxs, train_unlabeled_idxs])
    
    # Images
    images = dataset.data[all_idxs].astype(np.float32)
    
    # Compute the mean and std of the images
    mean = np.mean(images, axis = (0, 1, 2)) / 255
    std = np.std(images, axis = (0, 1, 2)) / 255
    
    return mean, std
    
    
    
def get_cifar10(root, n_labeled, transform_train = None, transform_val = None, download = False):
    base_dataset = datasets.CIFAR10(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, int(n_labeled/10))
    
    # Calculate the mean and std of the train 
    mean, std = get_mean_and_std(base_dataset, train_labeled_idxs, train_unlabeled_idxs)
    

    train_labeled_dataset = CIFAR10_labeled(root, mean, std, train_labeled_idxs, train=True,transform=transform_train)
    train_unlabeled_dataset = CIFAR10_unlabeled(root, mean,std, train_unlabeled_idxs, train=True, transform=TransformTwice(transform_train))
    val_dataset = CIFAR10_labeled(root,mean,std, val_idxs, train=True, transform=transform_val, download=True)
    test_dataset = CIFAR10_labeled(root,mean,std, train=False, transform=transform_val, download=True)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset

def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []
    
    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)
    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

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
        x = torch.from_numpy(x.copy())
        return x
    

class CIFAR10_labeled(datasets.CIFAR10):
    def __init__(self, root, mean, std, indexs = None, train = True, transform = None,
                 target_transform = None, download = False):
        super(CIFAR10_labeled, self).__init__(root, train = train, transform = transform,
                                              target_transform = target_transform, download = download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalize(self.data,mean,std))
    
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
    def __init__(self, root,mean,std, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, mean,std, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])

    