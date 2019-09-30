# TRAIN
# Terminal commands:
# cd /home/workspace/ImageClassifier
# python train.py data_dir

# Note: I used the following sources for ideas and assistance:
# Barnes, Rebecca. "Image Classifier Project." GitHub, 2018. https://github.com/rebeccaebarnes/DSND-Project-2
# Joshi, Kanchan. "Image Classifier Project." GitHub, 2018. https://github.com/koderjoker/Image-Classifier
# Kapotos, Fotis. "Image Classifier Project." GitHub, 2018. https://github.com/fotisk07/Image-Classifier
# Kussainov, Talgat. "Image Classifier Project." GitHub, 2018. https://github.com/Kusainov/udacity-image-classification
# Tabor, Sean. "Image Classifier Project." GitHub, 2018. https://github.com/S-Tabor/udacity-image-classifier-project 


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from torch.utils.data import DataLoader
import argparse


parser = argparse.ArgumentParser(description = 'Image classifier: training')
parser.add_argument('data_dir', action = 'store', help = 'Path to training data')
parser.add_argument('--arch', dest = 'arch', action = 'store', default = 'vgg16', help = 'Model type, default VGG16')
parser.add_argument('--dropout', dest = 'dropout', action = 'store', default = 0.2, help = 'Dropout rate, default 0.2')
parser.add_argument('--epochs', dest = 'epochs', action = 'store', default = 1, help = 'Epochs, default 7')
parser.add_argument('--gpu', dest = 'gpu', action = 'store', default = 'gpu', help = 'GPU mode')
parser.add_argument('--hidden_layer_1', dest = 'hidden_layer_1', action = 'store', default = 4096, help = 'Hidden units layer 1, default 4096')
parser.add_argument('--hidden_layer_2', dest = 'hidden_layer_2', action = 'store', default = 512, help = 'Hidden units layer 2, default 512')
parser.add_argument('--learning_rate', dest = 'learning_rate', action = 'store', default = 0.003, help = 'Learning rate, default 0.003')
parser.add_argument('--save_dir', dest = 'save_dir', action = 'store', default = 'checkpoint.pth', help = 'Checkpoint location')

args = parser.parse_args()

data_dir = args.data_dir
arch = args.arch
dropout = args.dropout
epochs = args.epochs
gpu = args.gpu
hidden_layer_1 = args.hidden_layer_1
hidden_layer_2 = args.hidden_layer_2
learning_rate = args.learning_rate
save_dir = args.save_dir

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.Resize(255),
                                       transforms.RandomCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

if arch == 'vgg16':
    model = models.vgg16(pretrained = True)
    input_features = 25088
elif arch == 'densenet121':
    model = models.densenet121(pretrained = True)
    input_features = 1024
else:
    print('Either vgg16 or densenet121')
        
for param in model.parameters():
    param.requires_grad = False    

model.classifier = nn.Sequential(nn.Linear(input_features, hidden_layer_1),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_layer_1, hidden_layer_2),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_layer_2, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            
            running_loss = 0
            model.train()
            
model.class_to_idx = train_data.class_to_idx
model.cpu()

checkpoint = {'model': model,
              'model.class_to_idx': model.class_to_idx,
              'epochs': epochs,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'classifier' : model.classifier,
              'learning_rate': learning_rate,
              'input_size': input_features,
              'hidden_layer_1': hidden_layer_1,
              'hidden_layer_2': hidden_layer_2,
              'output_size': 102 
             }

torch.save(checkpoint, save_dir)

print('Done!')
