import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import math
import cv2
from skimage import img_as_ubyte
import logging
import warnings
warnings.filterwarnings("ignore")

from utils import matching, stats, proggan, nethook, zdataset, loading, plotting, layers





def viz_matches(table, gan1, gan2, num, dataset, gan1layers, gan2layers, gan1stats, gan2stats):
    table_flattened = table.flatten()
    scores, flat_indices = torch.sort(table_flattened, descending = True)
    flat_indices_matches = flat_indices[15000:15000+num].cpu()
    indices_matches = np.unravel_index(flat_indices_matches, (table.shape[0], table.shape[1]))
    
    with torch.no_grad():
        for iii in range(num):
             #### through GAN
            gan1idx = layers.find_act(indices_matches[0][iii], gan1layers )
            img = gan1(dataset)
            gan1_act = gan1.retained_layer(list(gan1layers)[gan1idx[0]], clear = True)
            gan1_act = matching.normalize(gan1_act, gan1stats[gan1idx[0]])
            
            
            gan2idx = layers.find_act(indices_matches[1][iii], gan2layers )
            img2 = gan2(dataset)
            gan2_act = gan2.retained_layer(list(gan2layers)[gan2idx[0]], clear = True)
            gan2_act = matching.normalize(gan2_act, gan2stats[gan2idx[0]])

            ##### resize
            map_size = map_size = max((gan1_act.shape[2], gan2_act.shape[2]))
            gan1_act = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(gan1_act)
            gan2_act = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(gan2_act)        

            scores = torch.mul(gan1_act[:, gan1idx[1], :, :], gan2_act[:, gan2idx[1], :, :])
            scores = scores.view(scores.shape[0], -1)
            scores = torch.mean(scores, dim=-1, dtype = torch.float32)
            scores_sorted, indices = torch.sort(scores, descending = True)



            #### Prepare for plotting
            
            gan2_act_viz1 = gan2_act[indices[0], gan2idx[1]]
            gan2_act_viz = gan2_act_viz1.unsqueeze(0).unsqueeze(0)
            gan2_act_viz_orig = gan2_act_viz1
            gan2_act_viz = torch.nn.Upsample(size=(img.shape[2], img.shape[3]), mode='bilinear')(gan2_act_viz).cpu()
            gan2_act_viz = (gan2_act_viz-torch.min(gan2_act_viz))/(torch.max(gan2_act_viz)-torch.min(gan2_act_viz))
            gan2_act_viz = img_as_ubyte(gan2_act_viz.cpu().numpy())
            gan2_act_viz = cv2.applyColorMap(gan2_act_viz[0][0], cv2.COLORMAP_JET)

            gan1_act_viz1 = gan1_act[indices[0], gan1idx[1]]
            gan1_act_viz = gan1_act_viz1.unsqueeze(0).unsqueeze(0)
            gan1_act_viz_orig = gan1_act_viz1
            
            gan1_act_viz = torch.nn.Upsample(size=(img.shape[2], img.shape[3]), mode='bilinear')(gan1_act_viz).cpu()
            
            gan1_act_viz = (gan1_act_viz-torch.min(gan1_act_viz))/(torch.max(gan1_act_viz)-torch.min(gan1_act_viz)).cpu()
            gan1_act_viz = img_as_ubyte(gan1_act_viz.cpu().numpy())
            gan1_act_viz = cv2.applyColorMap(gan1_act_viz[0][0], cv2.COLORMAP_JET)


            img = (img+1)/2
            img2 = (img2+1)/2
            img_viz = np.transpose(img[indices[0]].cpu(), (1,2,0)).numpy()
            img2_viz = np.transpose(img2[indices[0]].cpu(), (1,2,0)).numpy()
            #img_viz = ((img_viz-torch.min(img_viz))/(torch.max(img_viz)-torch.min(img_viz))).numpy()


            ###Plot
            fig=plt.figure(figsize=(16, 5))
            plt.axis("off")
            plt.title("Similarity Match, Score: "+str(round(scores_sorted[0].item(), 3)), y=0.85)

            logger = logging.getLogger()
            old_level = logger.level
            logger.setLevel(100)

            alpha = 0.003
            minifig= fig.add_subplot(1, 6, 1)
            minifig.axis('off')
            minifig.title.set_text("GAN1 Image")
            plt.imshow(img_viz)


            minifig2 = fig.add_subplot(1, 6, 2)
            minifig2.axis('off')
            minifig2.title.set_text("GAN1 Layer "+str(gan1idx[0])+", Unit "+str(gan1idx[1])+"\n Heatmap")
            plt.imshow(alpha*gan1_act_viz+img_viz)
            
            minifig4 = fig.add_subplot(1, 6, 3)
            minifig4.axis('off')
            minifig4.title.set_text("GAN Activation Map")
            plt.imshow(gan1_act_viz_orig.cpu(), cmap = "gray")
            
            
            minifig6= fig.add_subplot(1, 6, 4)
            minifig6.axis('off')
            minifig6.title.set_text("GAN2 Image")
            plt.imshow(img2_viz)
            
            
            minifig3 = fig.add_subplot(1, 6, 5)
            minifig3.axis('off')
            minifig3.title.set_text("GAN2 Layer "+str(gan2idx[0])+", Unit "+str(gan2idx[1])+"\n Heatmap")
            plt.imshow(alpha*gan2_act_viz+img2_viz)

            minifig5 = fig.add_subplot(1, 6, 6)
            minifig5.axis('off')
            minifig5.title.set_text("GAN2 Activation Map")
            plt.imshow(gan2_act_viz_orig.cpu(), cmap = "gray")


            #plt.savefig("match_image_results/"+str(iii)+"_2.png", bbox_inches = 'tight',pad_inches = 0)




            logger.setLevel(old_level)
            
            del img
            del gan2_act_viz
            del gan2_act_viz1
            del gan1_act_viz
            del gan1_act_viz1
            torch.cuda.empty_cache()
    
    
    
        
        
        
        
        
        
        
                
        
        