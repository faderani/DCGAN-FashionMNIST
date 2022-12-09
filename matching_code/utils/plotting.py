import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_stats(gan1_stats, gan2_stats, table):
    
    gan1_means = []
    gan1_stds = []
    for iii, layer in enumerate(gan1_stats):
        gan1_means.append(gan1_stats[iii][0].flatten().unsqueeze(0))
        gan1_stds.append(gan1_stats[iii][1].flatten().unsqueeze(0))

    gan1_means = torch.cat(gan1_means,1)
    gan1_stds = torch.cat(gan1_stds,1)
    
    
    
    
    
    gan2_means = []
    gan2_stds = []
    for iii, layer in enumerate(gan2_stats):
        gan2_means.append(gan2_stats[iii][0].flatten().unsqueeze(0))
        gan2_stds.append(gan2_stats[iii][1].flatten().unsqueeze(0))

    gan2_means = torch.cat(gan2_means,1)
    gan2_stds = torch.cat(gan2_stds,1)
    
    
    
    
    ### scores
    
    table_flattened = table.flatten()
    scores, flat_indices = torch.sort(table_flattened, descending = True)
    flat_indices_matches = flat_indices[0:10000].cpu()
    indices_matches = np.unravel_index(flat_indices_matches, (table.shape[0], table.shape[1]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    #fig=plt.figure(figsize=(15, 15))
    #plt.title('Activation Statistics')
    #plt.axis('off')
    fig, ax = plt.subplots(figsize =(3, 3))
    ax.hist(gan2_means.cpu(), bins = [-20, -10, -5, -1, 0, 1, 5, 10, 20])
    plt.title("GAN2 Means")
    plt.show()
    
    
    fig, ax = plt.subplots(figsize =(3, 3))
    ax.hist(gan2_stds.cpu(), bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5,1, 5])
    plt.title("GAN2 STDs")
   # plt.xticks(np.arange(0, 1, 0.1), size =5)
    plt.show()
    
    
    fig, ax = plt.subplots(figsize =(3, 3))
    ax.hist(gan1_means.cpu(), bins = [-20, -10, -5, -1, 0, 1, 5, 10, 20])
    plt.title("GAN1 Means")
    plt.show()
    
    
    fig, ax = plt.subplots(figsize =(3, 3))
    ax.hist(gan1_stds.cpu(), bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5,1, 5])
    plt.title("GAN1 STDs")
    plt.show()
    
    
    gan1_score_idxs,_ = torch.max(table, dim = 1)
    fig, ax = plt.subplots(figsize =(10,6))
    ax.plot(gan1_score_idxs.cpu())
    plt.title("Scores vs. GAN1 Depth")
    plt.ylim(0,1.01)
    plt.show()
    
    
    
    gan2_score_idxs,_ = torch.max(table, dim = 0)
    fig, ax = plt.subplots(figsize =(10,6))
    ax.plot(gan2_score_idxs.cpu())
    plt.title("Scores vs. GAN2 Depth")
    plt.show()
    print(torch.sum(sum(gan1_means==gan2_means)))
    print(torch.sum(sum(gan1_stds==gan2_stds)))