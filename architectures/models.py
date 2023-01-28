import copy
import torch
import torchvision
import torch.nn.functional as F

class resnet50_1head(torch.nn.Module):

    def __init__(self):
        super(resnet50_1head, self).__init__()

        self.temperature = 0.05

        # Backbone
        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        num_filters = resnet50.fc.in_features
        resnet50.fc = torch.nn.Identity()
        self.backbone = resnet50

        # Head
        self.head = torch.nn.Linear(num_filters, 40)
        torch.nn.init.xavier_normal_(self.head.weight)

    def forward(self, x):
        f = self.backbone(x)
        x = F.normalize(f)
        score1 = self.head(x) / self.temperature
        return score1