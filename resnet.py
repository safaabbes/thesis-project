import torch
import torch.nn as nn
from torch.nn import init
import torchvision
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
import numpy as np

np.random.seed(1234)
torch.manual_seed(1234)

######################################################################
##### SENTRY IMPLEMENTATION
######################################################################


# We use PyTorch [32] for all experiments. On DomainNet, OfficeHome, and VisDA2017, we modify the standard
# ResNet50 [18] CNN architecture to a few-shot variant used
# in recent DA work [9, 37, 45]: we replace the last linear
# layer with a C− way (for C classes) fully-connected layer
# with Xavier-initialized weights and no bias. We then L2-
# normalize activations flowing into this layer and feed its
# output to a softmax layer with a temperature T = 0.05. We
# match optimization details to Tan et al. [45]. 

class ResNet50(nn.Module):

    def __init__(self, num_cls=10, weights_init=None, l2_normalize=False, temperature=1.0):
        super(ResNet50, self).__init__()
        self.num_cls = num_cls
        self.l2_normalize = l2_normalize
        self.temperature = temperature       
        self.criterion = nn.CrossEntropyLoss()
        
        weights = ResNet50_Weights.DEFAULT   
        model = resnet50(weights=weights)  
    
        model.fc = nn.Identity()
        self.conv_params = model
        self.fc_params = nn.Identity()        
        self.classifier = nn.Linear(2048, self.num_cls)
        
        init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.zero_()

    def forward(self, x):
        # Extract features
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = x.clone()
        emb = self.fc_params(x)                             
        if self.l2_normalize: emb = F.normalize(emb)
        score = self.classifier(emb) / self.temperature
      
        return score

        
    def load(self, init_path):
        net_init_dict = torch.load(init_path, map_location=torch.device('cpu'))
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)


######################################################################
##### ORIGINAL IMPLEMENTATION (using the sent file)
######################################################################


class resnet_block(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=None, stride=1, expansion=1):
        super(resnet_block, self).__init__()

        self.expansion = expansion  # this is used in resnet50 where the out_channels of each block are always 4 times more than the in_channels, e.g. 64 -> 256
        self.stride = stride

        if self.expansion > 1:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, stride=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.expansion > 1:  # this is for resnet >= 50
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class _resnet(nn.Module):
    def __init__(self, resnet_block, blocks, img_channels=3, expansion=1):
        super(_resnet, self).__init__()

        self.in_channels = 64
        self.expansion = expansion

        ## FEATURE EXTRACTION
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv2_x = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                     self._make_layer(resnet_block, n_blocks=blocks[0], out_channels=(2 ** 0) * 64))

        self.conv3_x = self._make_layer(resnet_block, n_blocks=blocks[1], out_channels=(2 ** 1) * 64, stride=2)

        self.conv4_x = self._make_layer(resnet_block, n_blocks=blocks[2], out_channels=(2 ** 2) * 64, stride=2)

        self.conv5_x = self._make_layer(resnet_block, n_blocks=blocks[3], out_channels=(2 ** 3) * 64, stride=2)

    def _make_layer(self, resnet_block, n_blocks, out_channels, stride=1):

        # _make_layer is used to build conv2_x, conv3_x, conv4_x and conv5_x
        # n_blocks defines the number of blocks for each layer, e.g. 2

        downsample = None
        if stride != 1 or self.expansion > 1:
            # this is used for the skip connection to match the same number of channels and resolution
            # the only block that needs downsampling is the first, sec caption Tab. 1
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels))

        layers = []
        layers.append(resnet_block(self.in_channels, out_channels, downsample, stride=stride, expansion=self.expansion))
        self.in_channels = self.expansion * out_channels
        for _ in range(1, n_blocks):
            layers.append(resnet_block(self.in_channels, out_channels, expansion=self.expansion))

        return nn.Sequential(*layers)

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2_x(x1)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)

        return x1, x2, x3, x4, x5

def resnet18(img_channels=3, pre_trained=False):
    blocks = [2, 2, 2, 2]  # see Tab. 1 resnet paper
    net_target = _resnet(resnet_block, blocks=blocks, img_channels=img_channels)

    if pre_trained:
        print('[i] using pre-trained resnet18')
        net_source = models.resnet18(pretrained=True)
        for param_source, param_target in zip(net_source.parameters(), net_target.parameters()):
            if param_source.requires_grad:
                if param_source.shape == param_target.shape:
                    param_target.data.copy_(param_source.data)

    return net_target


def resnet34(img_channels=3, pre_trained=False):
    blocks = [3, 4, 6, 3]  # see Tab. 1 resnet paper
    net_target = _resnet(resnet_block, blocks=blocks, img_channels=img_channels)

    if pre_trained:
        print('[i] using pre-trained resnet34')
        net_source = models.resnet34(pretrained=True)
        for param_source, param_target in zip(net_source.parameters(), net_target.parameters()):
            if param_source.requires_grad:
                if param_source.shape == param_target.shape:
                    param_target.data.copy_(param_source.data)

    return net_target


def resnet50(img_channels=3, pre_trained=False):
    blocks = [3, 4, 6, 3]  # see Tab. 1 resnet paper
    net_target = _resnet(resnet_block, blocks=blocks, img_channels=img_channels, expansion=4)

    if pre_trained:
        print('[i] using pre-trained resnet50')
        weights = ResNet50_Weights.DEFAULT   
        net_source = resnet50(weights=weights)  
        for param_source, param_target in zip(net_source.parameters(), net_target.parameters()):
            if param_source.requires_grad:
                if param_source.shape == param_target.shape:
                    param_target.data.copy_(param_source.data)

    return net_target


def resnet101(img_channels=3, pre_trained=False):
    blocks = [3, 4, 23, 3]  # see Tab. 1 resnet paper
    net_target = _resnet(resnet_block, blocks=blocks, img_channels=img_channels, expansion=4)

    if pre_trained:
        print('[i] using pre-trained resnet101')
        net_source = models.resnet101(pretrained=True)
        for param_source, param_target in zip(net_source.parameters(), net_target.parameters()):
            if param_source.requires_grad:
                if param_source.shape == param_target.shape:
                    param_target.data.copy_(param_source.data)

    return net_target


def resnet152(img_channels=3, pre_trained=False):
    blocks = [3, 8, 36, 3]  # see Tab. 1 resnet paper
    net_target = _resnet(resnet_block, blocks=blocks, img_channels=img_channels, expansion=4)

    if pre_trained:
        print('[i] using pre-trained resnet152')
        net_source = models.resnet152(pretrained=True)
        for param_source, param_target in zip(net_source.parameters(), net_target.parameters()):
            if param_source.requires_grad:
                if param_source.shape == param_target.shape:
                    param_target.data.copy_(param_source.data)

    return net_target


if __name__ == '__main__':
    x = torch.randn((6, 3, 224, 224))
    net = resnet50(pre_trained=True)
    print(net)
    x1, x2, x3, x4, x5 = net(x)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)
    print(x5.shape)
