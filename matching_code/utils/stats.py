from utils import nethook, zdataset
import torch
import torchvision





def get_mean_std(gan1, gan1_layers, gan2, gan2_layers, epochs, batch_size, device):
    print("Collecting Dataset Statistics")
    
    gan1_stats_list = []
    gan2_stats_list = []
    with torch.no_grad():
        for iteration in range(0, epochs):
            print(iteration)
            dataset = zdataset.z_dataset_for_model(gan1, size=batch_size, seed=5555+iteration)
            z = dataset[:][0][None,...][0].to(device)

            img = gan1(z)

            #### append GAN layer activations for batch
            gan1_activs = {}
            for layer in gan1_layers:
                gan1_activs[layer] = []    
                gan1_activation = gan1.retained_layer(layer, clear = True)
                gan1_activs[layer].append(gan1_activation)
                
                
                
            img = gan2(z)

            #### append GAN layer activations for batch
            gan2_activs = {}
            for layer in gan2_layers:
                gan2_activs[layer] = []    
                gan2_activation = gan2.retained_layer(layer, clear = True)
                gan2_activs[layer].append(gan2_activation)






            batch_gan1_stats_list = []
            for layer in gan1_layers:
                gan1_activs[layer] = torch.cat(gan1_activs[layer], 0) #images x channels x m x m
                gan1_activs[layer] = torch.permute(gan1_activs[layer], (1,0,2,3)).contiguous() #channels x images x m x m
                gan1_activs[layer] = gan1_activs[layer].view(gan1_activs[layer].shape[0], -1) 
                batch_gan1_stats_list.append([torch.mean(gan1_activs[layer],dim=-1, dtype=torch.float64).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device),\
                                      torch.std(gan1_activs[layer], dim=-1).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)])

            del gan1_activs
            gan1_stats_list.append(batch_gan1_stats_list)
            
            
            
            
            
            
            batch_gan2_stats_list = []
            for layer in gan2_layers:
                gan2_activs[layer] = torch.cat(gan2_activs[layer], 0) #images x channels x m x m
                gan2_activs[layer] = torch.permute(gan2_activs[layer], (1,0,2,3)).contiguous() #channels x images x m x m
                gan2_activs[layer] = gan2_activs[layer].view(gan2_activs[layer].shape[0], -1) 
                batch_gan2_stats_list.append([torch.mean(gan2_activs[layer],dim=-1, dtype=torch.float64).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device),\
                                      torch.std(gan2_activs[layer], dim=-1).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)])

            del gan2_activs
            gan2_stats_list.append(batch_gan2_stats_list)


            



            

    
    ####################### After iterating
        print("Done Iteration for Stats")
    
        final_gan1_stats_list = []

        for iii in range(len(batch_gan1_stats_list)):
            means = torch.zeros_like(batch_gan1_stats_list[iii][0])
            stds = torch.zeros_like(batch_gan1_stats_list[iii][1])
            for jjj in range(epochs):
                means+=gan1_stats_list[jjj][iii][0]
                stds+=gan1_stats_list[jjj][iii][1]**2

            final_gan1_stats_list.append([means/epochs, torch.sqrt(stds/epochs)])
        
        
        final_gan2_stats_list = []

        for iii in range(len(batch_gan2_stats_list)):
            means = torch.zeros_like(batch_gan2_stats_list[iii][0])
            stds = torch.zeros_like(batch_gan2_stats_list[iii][1])
            for jjj in range(epochs):
                means+=gan2_stats_list[jjj][iii][0]
                stds+=gan2_stats_list[jjj][iii][1]**2

            final_gan2_stats_list.append([means/epochs, torch.sqrt(stds/epochs)])




    return final_gan1_stats_list, final_gan2_stats_list