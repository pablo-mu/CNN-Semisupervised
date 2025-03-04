from cnn import SimpleCNN
from wideresnet import WideResNet


def create_model(ema = False, model = 'cnn'):
    if model == 'cnn':
        model = SimpleCNN(num_classes = 10).cuda()
    else:
        model = WideResNet(depth=28, num_classes=10).cuda()
    if ema:
        for param in model.parameters():
            param.detach_()    
    return model