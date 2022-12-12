import torch
import torch.nn as nn
from torch.nn import init
import torchvision
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
import numpy as np

######################################################################
##### SENTRY IMPLEMENTATION
######################################################################
class SENTRY_ResNet50(nn.Module):

    def __init__(self, num_cls=40, weights_init=None, l2_normalize=True, temperature=0.05):
        super(SENTRY_ResNet50, self).__init__()
        self.num_cls = num_cls
        self.l2_normalize = l2_normalize
        self.temperature = temperature       
        self.criterion = nn.CrossEntropyLoss()
        model = resnet50(weights=ResNet50_Weights.DEFAULT)  
        model.fc = nn.Identity()
        self.conv_params = model
        self.fc_params = nn.Identity()        
        self.classifier = nn.Linear(2048, self.num_cls)
        init.xavier_normal_(self.classifier.weight)

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
##### ORIGINAL PYTORCH IMPLEMENTATION 
######################################################################
class Bottleneck(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, downsample=None, stride=1
    ):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, Bottleneck, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            Bottleneck, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            Bottleneck, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            Bottleneck, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            Bottleneck, layers[3], intermediate_channels=512, stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
        self.classifier = nn.Linear(in_features=2048, out_features=1000, bias=True)

    def forward(self, x, path = 'main'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f = self.layer3(x)
        f = self.layer4(f)
        f = self.avgpool(f)
        f = f.reshape(f.shape[0], -1)
        f = self.classifier(f)
        
        return f

    def _make_layer(self, Bottleneck, num_residual_blocks, intermediate_channels, stride):
        downsample = None
        layers = []
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            Bottleneck(self.in_channels, intermediate_channels, downsample, stride)
        )
        self.in_channels = intermediate_channels * 4
        for i in range(num_residual_blocks - 1):
            layers.append(Bottleneck(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)
    
    def load(self, init_path):
        net_init_dict = torch.load(init_path, map_location=torch.device('cpu'))
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)
        
######################################################################
        
def ResNet50(img_channel=3, num_classes=1000, n_super_classes = 5, pre_trained=True, progress=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3], img_channel, num_classes)
    if pre_trained:
        # Load pre-trained resnet50 weights 
        source = resnet50(weights=ResNet50_Weights.DEFAULT)
        #Transfer weights to new model
        for param_source, param_target in zip(source.parameters(), model.parameters()):
            if param_source.requires_grad:
                if param_source.shape == param_target.shape:
                    param_target.data.copy_(param_source.data)
        # initialize classifier 
        model.classifier = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    return model


######################################################################
#####  MODEL V1 IMPLEMENTATION 
######################################################################

class Model_V1(nn.Module):
    def __init__(self, Bottleneck, layers, image_channels, num_classes):
        super(Model_V1, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            Bottleneck, layers[0], intermediate_channels=64, stride=1                   # Linear in_features = 256
        )
        self.layer2 = self._make_layer(
            Bottleneck, layers[1], intermediate_channels=128, stride=2                  # Linear in_features = 512
        )
        self.layer3 = self._make_layer(
            Bottleneck, layers[2], intermediate_channels=256, stride=2                  # Linear in_features = 1024
        )
        self.layer4 = self._make_layer(
            Bottleneck, layers[3], intermediate_channels=512, stride=2                  # Linear in_features = 2048
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)
        self.fcb = nn.Linear(in_features=512, out_features=1000, bias=True)

    def forward(self, x, path = 'main'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if path == 'main':
            f = self.layer3(x)
            f = self.layer4(f)
            f = self.avgpool(f)
            f = f.reshape(f.shape[0], -1)
            f = self.fc(f)
            return f
        elif path == 'branch':
            g = self.layer3(x)
            g = self.layer4(g)
            g = self.avgpool(x)
            g = g.reshape(g.shape[0], -1)
            g = self.fcb(g)
            return g 
        else:
            raise Exception("Error! Incorrect Path Specified") 

    def _make_layer(self, Bottleneck, num_residual_blocks, intermediate_channels, stride):
        downsample = None
        layers = []
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            Bottleneck(self.in_channels, intermediate_channels, downsample, stride)
        )
        self.in_channels = intermediate_channels * 4
        for i in range(num_residual_blocks - 1):
            layers.append(Bottleneck(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)
    
    def load(self, init_path):
        net_init_dict = torch.load(init_path, map_location=torch.device('cpu'))
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

######################################################################

def Res50_V1(img_channel=3, num_classes=1000, n_super_classes = 5, pre_trained=True, progress=True):
    model = Model_V1(Bottleneck, [3, 4, 6, 3], img_channel, num_classes)
    if pre_trained:
        # Load pre-trained resnet50 weights 
        source = resnet50(weights=ResNet50_Weights.DEFAULT)
        #Transfer weights to new model
        for param_source, param_target in zip(source.parameters(), model.parameters()):
            if param_source.requires_grad:
                if param_source.shape == param_target.shape:
                    param_target.data.copy_(param_source.data)
        # re-initialize classifier (fc)
        model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        model.fcb = nn.Linear(in_features=512, out_features=n_super_classes, bias=True)
    return model


        