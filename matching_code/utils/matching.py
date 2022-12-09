from utils import nethook, zdataset, stats, helpers

import torch
import torchvision
import numpy as np
import pickle
import os





def normalize(activation, stats_table):
    eps = 0.000001
    norm_input = (activation- stats_table[0])/(stats_table[1])
    
    return norm_input

def store_activs(model, layers):
    activs = []
    for layer in layers:
        activation = model.retained_layer(layer, clear = True)
        activs.append(activation)
        
    return activs

def dict_layers(activs):
    all_layers = {}
    for iii, activ in enumerate(activs):
        all_layers[activs[iii]] = activ.shape[1]
    return all_layers
    


def activ_match_gan(gan1, gan1_layers, gan2, gan2_layers,epochs, batch_size,save_path, device):
    gan1.eval()
    gan2.eval()
    
    #### hook layers for GAN
    gan1 = nethook.InstrumentedModel(gan1)
    gan1.retain_layers(gan1_layers)
    
    gan2 = nethook.InstrumentedModel(gan2)
    gan2.retain_layers(gan2_layers)
    
    
    
    #get dataset stats
    gan1_stats_table, gan2_stats_table = stats.get_mean_std(gan1, gan1_layers, gan2, gan2_layers, epochs, batch_size, device)
    helpers.save_array(gan1_stats_table, os.path.join(save_path, "gan1_stats.pkl"))
    helpers.save_array(gan2_stats_table, os.path.join(save_path, "gan2_stats.pkl"))
    
    print("Done")
    print("Starting Activation Matching")
    
    

    for iteration in range(0, epochs):
        with torch.no_grad():
            print("Iteration: "+ str(iteration))
            #### dataset
            dataset = zdataset.z_dataset_for_model(gan1, size=batch_size, seed=5555+iteration)
            z = dataset[:][0][None,...][0].to(device)
            #### Forward through GAN
            img = gan1(z)
            
            del img
            #### append GAN layer activations for batch
            gan1_activs = store_activs(gan1, gan1_layers)
            
            img = gan2(z)
            del z
            del img
            #### append GAN layer activations for batch
            gan2_activs = store_activs(gan2, gan2_layers)

           


            #create dict of layers with number of activations
            all_gan1_layers = dict_layers(gan1_activs)
            all_gan2_layers = dict_layers(gan2_activs)
            
            
            if iteration == 0:
                num_gan1_activs = sum(all_gan1_layers.values())
                num_gan2_activs = sum(all_gan2_layers.values())
                final_match_table = torch.zeros((num_gan1_activs, num_gan2_activs)).to(device)



            ##### Matching
            all_match_table = []

            #
            for ii, gan1_activ in enumerate(gan1_activs):
                print("GAN1 Layer:" + str(ii))
                print("________________________")

                match_table = []
                gan1_activ = normalize(gan1_activ, gan1_stats_table[ii])
                gan1_activ_shape = gan1_activ.shape

                for jj, gan2_activ in enumerate(gan2_activs):
                    print("GAN2 Layer " + str(jj))
                    gan2_activ_new = normalize(gan2_activ, gan2_stats_table[jj]) 
                    #### Get maps to same size
                    map_idx = np.argmax((gan1_activ_shape[2], gan2_activ.shape[2]))
                    map_size = max((gan1_activ_shape[2], gan2_activ.shape[2]))
                    
                    
                    if gan1_activ_shape[2] != gan2_activ.shape[2]:
                        if map_idx == 0:
                            gan2_activ_new = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear', align_corners=True)(gan2_activ_new) 
                            gan1_activ_new = gan1_activ
                        else:
                            gan1_activ_new = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear', align_corners=True)(gan1_activ)
                            
                    else: 
                        gan1_activ_new = gan1_activ
                            
                        
                        
                    scores = torch.einsum('aixy,ajxy->ij', gan1_activ_new,gan2_activ_new)/(batch_size*map_size**2)  
                    scores = scores.cpu()
                    
                    match_table.append(scores)
                    del gan1_activ_new
                    del gan2_activ_new
                    del scores

                all_match_table.append(match_table)
                del match_table


            ##create table
            batch_match_table = create_final_table(all_match_table, all_gan1_layers, all_gan2_layers, batch_size, device)
            final_match_table += batch_match_table
            #helpers.save_array(final_match_table, os.path.join(save_path, "norm_table_"+str(iteration)+".pkl"))
        
            del all_match_table
            del batch_match_table
            del gan1_activs
            del gan2_activs
            torch.cuda.empty_cache()
            
    #average and save
    final_match_table /= epochs
    helpers.save_array(final_match_table, os.path.join(save_path, "table.pkl"))
    
    
    
def create_final_table(all_match_table, model1_dict, model2_dict, batch_size, device ):
    num_activs1 = sum(model1_dict.values())
    num_activs2 = sum(model2_dict.values())
    final_match_table = torch.zeros((num_activs1, num_activs2)).to(device)
    
    model1_activ_count = 0 
    for ii in range(len(all_match_table)):
        model2_activ_count = 0
        for jj in range(len(all_match_table[ii])):
            num_model1activs = all_match_table[ii][0].shape[0]
            num_model2activs = all_match_table[0][jj].shape[1]
            final_match_table[model1_activ_count: model1_activ_count+num_model1activs, \
                            model2_activ_count:model2_activ_count+num_model2activs] = all_match_table[ii][jj]
            model2_activ_count += num_model2activs
        model1_activ_count += num_model1activs
    return final_match_table
    
    
    
    
    
    
    
    
    
    
    