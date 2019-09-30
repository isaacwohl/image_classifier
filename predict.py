# PREDICT
# Terminal commands:
# python predict.py img_path checkpoint
# Example: img_path = ('flowers/test/73/image_00260.jpg')

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
from PIL import Image
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser(description = 'Image classifier: prediction')
parser.add_argument('--cat_to_name', dest = 'cat_to_name', action = 'store', default = 'cat_to_name.json', help = 'Path from categories to names, default cat_to_name.json')
parser.add_argument('--checkpoint', dest = 'checkpoint', action = 'store', default = 'checkpoint.pth', help = 'Checkpoint location, default checkpoint.pth')
parser.add_argument('--gpu', dest = 'gpu', action = 'store', default = 'gpu', help = 'GPU mode')
parser.add_argument('--img_path', dest = 'img_path', action = 'store', default = 'flowers/test/73/image_00260.jpg', help = 'Path to image, default water lily image')
parser.add_argument('--top_k', dest = 'top_k', action = 'store', default = 5, help = 'Number of most likely classes, default 5')

args = parser.parse_args()

cat_to_name = args.cat_to_name
checkpoint = args.checkpoint
gpu = args.gpu
img_path = args.img_path
top_k = args.top_k

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = checkpoint['model']
    model.class_to_idx = checkpoint['model.class_to_idx']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.classifier = checkpoint['classifier']
    
    model.load_state_dict(checkpoint['state_dict'])
    
    # Freeze training
    for param in model.parameters():
        param.requires_grad = False
    
    return model

model = load_checkpoint(checkpoint)

def process_image(image):
    image = Image.open(image)

    # Resize
    if image.size[0] > image.size[1]:
        image.thumbnail((image.size[0], 256))
    else:
        image.thumbnail((256, image.size[1]))
    
    # Crop 
    left_margin = (image.width - 224) / 2
    bottom_margin = (image.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    pil_image = image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    np_image = np.array(pil_image) / 255
    
    # Normalize: subtract the means from each color channel, then divide by stdev
    image_mean = np.array([0.485, 0.456, 0.406])
    image_std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - image_mean) / image_std
    
    # Reorder color channel from 3rd dimension to 1st dimension
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image  

def predict(image_path, model, topk, device):
    img_torch = process_image(image_path)
    img_torch = torch.from_numpy(img_torch).type(torch.FloatTensor)
    img_torch = img_torch.unsqueeze(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch)
        
    probability = F.softmax(output.data, dim = 1)
    
    model.to(device)
    
    return probability.topk(topk)

if gpu == 'gpu':
    device = 'cuda:0'
else:
    device = 'cpu'

prob, classes = predict(img_path, model, top_k, device)

prob = prob[0]
classes = classes[0]

labels = []
for cl in classes:
    labels.append(cat_to_name[str(cl.item())])

for k in range(top_k):
    print('{} with a probability of {}'.format(labels[k], prob[k]))
