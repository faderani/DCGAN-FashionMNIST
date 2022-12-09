#!/usr/bin/env python
# coding: utf-8

# In[9]:


from transformers import CLIPProcessor, CLIPModel
import torch
import torchvision
from torchvision.models import resnet50
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import clip
from PIL import Image
import requests
import torch.hub
import time
import pickle
import math

from utils import matching, stats, proggan, nethook, zdataset


# In[10]:


device = torch.device('cuda:0')


# In[11]:


from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)

gan = BigGAN.from_pretrained('biggan-deep-256').to(device)


# In[12]:


gan_layers = []
for name, layer in gan.named_modules():
    if "conv" in name:
        gan_layers.append(name)


# In[13]:


clipmodel, preprocess = clip.load("RN50", device=device)
clip_layers = [ "visual.layer1", "visual.layer2", "visual.layer3", "visual.layer4"]
for p in clipmodel.parameters(): 
    p.data = p.data.float() 


# In[14]:


batch_size = 10
epochs = 10000
save_path = "results" 


# #### Create dataset

# In[15]:


z_dataset = torch.randn((100000,128)).to(device)
c_dataset = torch.zeros((100000,1000)).to(device)
num_c = 100
for iii in range(1000):
    c_dataset[num_c*iii:iii+num_c, iii] = 1


# In[16]:


# z_dataset = torch.randn((10000,128)).to(device)
# c_dataset = torch.zeros((10000,1000)).to(device)
# for iii in range(1000):
#     c_dataset[10*iii:iii+10, iii] = 1


# In[17]:


start = time.time()
matching.activ_match_gan(gan, gan_layers,
                         clipmodel, clip_layers, 
                         (z_dataset, c_dataset),
                         epochs,
                         batch_size,
                         save_path,
                         device)
end = time.time()
print(end-start)


# In[ ]:





# In[ ]:




