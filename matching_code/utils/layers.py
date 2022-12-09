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

from utils import matching, stats, proggan, nethook, zdataset, loading, plotting




def get_layers(gan1, gan1_layers, gan2,gan2_layers, device):
    gan1.eval()
    gan2.eval()
    
    #### hook layers for GAN
    gan1 = nethook.InstrumentedModel(gan1)
    gan1.retain_layers(gan1_layers)
    gan2 = nethook.InstrumentedModel(gan2)
    gan2.retain_layers(gan2_layers)
    
    
    
    
   
        
      
    dataset = zdataset.z_dataset_for_model(gan1, size=1, seed=0)

    z = dataset[:][0][None,...][0].to(device).detach()
        
    batch_size = z.shape[0]
        
    #### Forward through GAN
    with torch.no_grad():
        img = gan1(z).detach()
        
        del img
        

        #### append GAN layer activations for batch
    gan1_activs = []
    for layer in gan1_layers:
        gan1_activation = gan1.retained_layer(layer, clear = True).detach()
        gan1_activs.append(gan1_activation)
                
    with torch.no_grad():
        img = gan2(z).detach()
        del z
        del img



        

    gan2_activs = []
    for layer in gan2_layers:
        gan2_activation = gan2.retained_layer(layer, clear = True).detach()
        gan2_activs.append(gan2_activation)

        #create dict of layers
    all_gan1_layers = {}
    for iii, gan1_activ in enumerate(gan1_activs):
        all_gan1_layers[gan1_layers[iii]] = gan1_activ.shape[1]
        
    all_gan2_layers = {}
    for iii, gan2_activ in enumerate(gan2_activs):
        all_gan2_layers[gan2_layers[iii]] = gan2_activ.shape[1]
        
    
            
    return all_gan1_layers, all_gan2_layers







def find_act(act_num, net_dict):
    layers_list = list(net_dict)
    
    layer = 0
    counter =0
    
    while act_num >= counter:
        layer +=1
        counter += net_dict[layers_list[layer-1]]
        
        
    act = act_num-counter+net_dict[layers_list[layer-1]]
    return (layer-1), act
        