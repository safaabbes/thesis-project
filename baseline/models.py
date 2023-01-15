import copy
import torch
import torchvision
import torch.nn.functional as F


class resnet50a(torch.nn.Module):

    def __init__(self, args):
        super(resnet50a, self).__init__()

        # Backbone
        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        num_filters = resnet50.fc.in_features
        resnet50.fc = torch.nn.Identity()
        self.backbone = resnet50

        # Heads
        self.head1 = torch.nn.Linear(num_filters, args.num_categories1)
        self.head2 = torch.nn.Linear(num_filters, args.num_categories2)

    def forward(self, x):
        x = self.backbone(x)
        x1 = self.head1(x)
        x2 = self.head2(x)
        return x1, x2


class resnet50b(torch.nn.Module):
    # Same as a

    def __init__(self, args):
        super(resnet50b, self).__init__()

        # Backbone
        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        num_filters = resnet50.fc.in_features
        resnet50.fc = torch.nn.Identity()
        self.backbone = resnet50

        # Heads
        self.head1 = torch.nn.Linear(num_filters, args.num_categories1)
        self.head2 = torch.nn.Linear(num_filters, args.num_categories2)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)  # (B, 2048, 1, 1)
        x = x.reshape(x.shape[0], -1)  # (B, 2048)
        x1 = self.head1(x)  # (B, C1)
        x2 = self.head2(x)  # (B, C2)
        return x1, x2


class resnet50c(torch.nn.Module):
    # Same as a, b

    def __init__(self, args):
        super(resnet50c, self).__init__()

        # Backbone
        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        num_filters = resnet50.fc.in_features
        resnet50.fc = torch.nn.Identity()
        self.backbone = resnet50

        # Heads
        self.head1 = torch.nn.Linear(num_filters, args.num_categories1)
        self.head2 = torch.nn.Linear(num_filters, args.num_categories2)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)

        # Main branch
        x1 = self.backbone.layer3(x)
        x1 = self.backbone.layer4(x1)
        x1 = self.backbone.avgpool(x1)  # (B, 2048, 1, 1)
        x1 = x1.reshape(x1.shape[0], -1)  # (B, 2048)
        x1 = self.head1(x1)  # (B, C1)

        # Secondary branch
        x2 = self.backbone.layer3(x)
        x2 = self.backbone.layer4(x2)
        x2 = self.backbone.avgpool(x2)  # (B, 2048, 1, 1)
        x2 = x2.reshape(x2.shape[0], -1)  # (B, 2048)
        x2 = self.head2(x2)  # (B, C2)

        return x1, x2


class resnet50d(torch.nn.Module):

    def __init__(self, args):
        super(resnet50d, self).__init__()

        # Backbone
        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        num_filters = resnet50.fc.in_features
        resnet50.fc = torch.nn.Identity()
        self.backbone1 = copy.deepcopy(resnet50)
        self.backbone2 = copy.deepcopy(resnet50)

        # Heads
        self.head1 = torch.nn.Linear(num_filters, args.num_categories1)
        self.head2 = torch.nn.Linear(num_filters, args.num_categories2)

    def forward(self, x):
        x = self.backbone1.conv1(x)
        x = self.backbone1.bn1(x)
        x = self.backbone1.relu(x)
        x = self.backbone1.maxpool(x)
        x = self.backbone1.layer1(x)
        x = self.backbone1.layer2(x)

        # Main branch
        x1 = self.backbone1.layer3(x)
        x1 = self.backbone1.layer4(x1)
        x1 = self.backbone1.avgpool(x1)  # (B, 2048, 1, 1)
        x1 = x1.reshape(x1.shape[0], -1)  # (B, 2048)
        x1 = self.head1(x1)  # (B, C1)

        # Secondary branch
        x2 = self.backbone2.layer3(x)
        x2 = self.backbone2.layer4(x2)
        x2 = self.backbone2.avgpool(x2)  # (B, 2048, 1, 1)
        x2 = x2.reshape(x2.shape[0], -1)  # (B, 2048)
        x2 = self.head2(x2)  # (B, C2)

        return x1, x2


class resnet50e(torch.nn.Module):

    def __init__(self, args):
        super(resnet50e, self).__init__()

        self.temperature = args.temperature

        # Backbone
        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        num_filters = resnet50.fc.in_features
        resnet50.fc = torch.nn.Identity()
        self.backbone1 = copy.deepcopy(resnet50)
        self.backbone2 = copy.deepcopy(resnet50)

        # Heads
        self.head1 = torch.nn.Linear(num_filters, args.num_categories1)
        torch.nn.init.xavier_normal_(self.head1.weight)
        self.head2 = torch.nn.Linear(num_filters, args.num_categories2)
        torch.nn.init.xavier_normal_(self.head2.weight)

    def forward(self, x):
        x = self.backbone1.conv1(x)
        x = self.backbone1.bn1(x)
        x = self.backbone1.relu(x)
        x = self.backbone1.maxpool(x)
        x = self.backbone1.layer1(x)
        x = self.backbone1.layer2(x)

        # Main branch
        x1 = self.backbone1.layer3(x)
        x1 = self.backbone1.layer4(x1)
        x1 = self.backbone1.avgpool(x1)  # (B, 2048, 1, 1)
        x1 = x1.reshape(x1.shape[0], -1)  # (B, 2048)
        x1 = F.normalize(x1)
        x1 = self.head1(x1) / self.temperature  # (B, C1)

        # Secondary branch
        x2 = self.backbone2.layer3(x)
        x2 = self.backbone2.layer4(x2)
        x2 = self.backbone2.avgpool(x2)  # (B, 2048, 1, 1)
        x2 = x2.reshape(x2.shape[0], -1)  # (B, 2048)
        x2 = F.normalize(x2)
        x2 = self.head2(x2) / self.temperature  # (B, C2)

        return x1, x2


class resnet50s(torch.nn.Module):

    def __init__(self, args):
        super(resnet50s, self).__init__()

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


class resnet50s_1head(torch.nn.Module):

    def __init__(self, args):
        super(resnet50s_1head, self).__init__()

        self.temperature = args.temperature

        # Backbone
        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        num_filters = resnet50.fc.in_features
        resnet50.fc = torch.nn.Identity()
        self.backbone = resnet50

        # Head
        self.head = torch.nn.Linear(num_filters, args.num_categories1)
        torch.nn.init.xavier_normal_(self.head.weight)

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x)
        score1 = self.head(x) / self.temperature
        return score1