""" Adapted from the pytorch tutorial on github for extracting features """
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/neural_style_transfer/main.py
# seems a version of the tutorial that is a little old but simple enough for our needs
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from scipy import spatial
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['28'] 
        self.vgg = models.vgg16(pretrained=True).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

def load_image(fname):
    """Load an image and convert it to a torch tensor."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #])
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))])

    img = Image.open(fname).convert('RGB')
    #image = transform(image).unsqueeze(0)
    image = transform(img)
    return image

def tensor_to_pil(image):
    pil_image = image.numpy().transpose(1,2,0)
    pil_image = (pil_image * 255).astype(int)
    return pil_image

def imshow(image):
    plt.imshow(image)
    plt.show()

def write_to_files(opath, features):
    for k,v in features.items():
        f = v.data
        np.save(opath + k, f)

def get_features_chunk(image_path, image_list, model):
    features = {} 
    images = image_list 

    for i,image in enumerate(images):
        img = load_image(image_path + image)
        f = model(img.unsqueeze(0))
        v = f[0]
        _,c,h,w = v.size()
        v = v.view(c*h*w)
        features[image] = v.data

    return features

def get_features(image_path, image, model):
    img = load_image(image_path + image)
    f = model(img.unsqueeze(0))
    v = f[0]
    _,c,h,w = v.size()
    v = v.view(c*h*w)
    features = np.array(v.data)
    return (image, features)

