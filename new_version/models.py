import copy
import torch
import torchvision
import torch.nn.functional as F

class resnet50_1h(torch.nn.Module):

    def __init__(self, args):
        super(resnet50_1h, self).__init__()

        self.temperature = args.temperature

        # Backbone
        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        num_filters = resnet50.fc.in_features
        resnet50.fc = torch.nn.Identity()
        self.backbone = resnet50

        # Head
        self.head = torch.nn.Linear(num_filters, args.num_categories1)
        torch.nn.init.xavier_normal_(self.head.weight)

        # TODO set bias to 0
        
    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x)
        score1 = self.head(x) / self.temperature
        return score1

class resnet50_2h(torch.nn.Module):

    def __init__(self, args):
        super(resnet50_2h, self).__init__()

        self.temperature = args.temperature

        # Backbone
        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        num_filters = resnet50.fc.in_features
        resnet50.fc = torch.nn.Identity()
        self.backbone = resnet50

        # Heads
        self.head1 = torch.nn.Linear(num_filters, args.num_categories1)
        torch.nn.init.xavier_normal_(self.head1.weight)
        self.head2 = torch.nn.Linear(num_filters, args.num_categories2)
        torch.nn.init.xavier_normal_(self.head2.weight)

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x)
        score1 = self.head1(x) / self.temperature
        score2 = self.head2(x) / self.temperature
        return score1, score2


