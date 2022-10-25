import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights

from dataset.dataloader import split_dl
from training.train import training_step
from training.test import test_step

so_history = []
lr = 1e-3
epochs = 10

# load data

source_dataset_name = 'clipart'
clipart_stats = ([0.7335894,0.71447897,0.6807669],[0.3542898,0.35537153,0.37871686])
target_dataset_name = 'sketch'
#sketch_stats = ()

path = 'storage/TEV/sabbes/domainnet/'

img_transform_source = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(*clipart_stats),
])

img_transform_target = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(*sketch_stats),
])

s_dataset = ImageFolder(path+source_dataset_name, img_transform_source)
t_dataset = ImageFolder(path+target_dataset_name, img_transform_target)

print('number of product samples:', len(s_dataset))
print('number of real life samples:', len(t_dataset))
print('Does the source and target have the same labels?:', s_dataset.classes == t_dataset.classes)

s_train, s_test = split_dl(s_dataset, 64) 
t_train, t_test = split_dl(t_dataset, 64) 

#load model

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

#setup optimizer

optimizer = optim.Adam(model.parameters(), lr=lr)

# Test and Train for each epoch
for epoch in range(epochs):
    train_loss , train_accuracy = training_step(model,s_train, optimizer)
    s_test_loss, s_test_accuracy = test_step(model,s_test)
    t_test_loss, t_test_accuracy = test_step(model,t_test)

    # Print Results
    print('Epoch: {:d}'.format(epoch+1))
    print('\t Source Train loss {:.5f}, Source Train accuracy {:.2f}'.format(train_loss, train_accuracy))
    print('\t Source Test loss {:.5f}, Source Test accuracy {:.2f}'.format(s_test_loss, s_test_accuracy))
    print('\t Target Test loss {:.5f}, Target Test accuracy {:.2f}'.format(t_test_loss, t_test_accuracy))
    print('-----------------------------------------------------------------------')

    # Save history 
    so_history.append({'Source_Train': {'s_train_loss': train_loss, 's_train_acc': train_accuracy },
                    'Source_Test': {'s_test_loss': s_test_loss, 's_test_acc': s_test_accuracy },
                    'Target_Test': {'t_test_loss': t_test_loss, 't_test_acc': t_test_accuracy }
                    })

