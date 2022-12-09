import torch
import pickle
import os

def load_stats(root, device):
    print("Loading...")
    file_name = os.path.join(root, "table.pkl")
    with open(file_name, 'rb') as f:
        table = pickle.load(f)
        table = table.to(device)#cpu()
    
    with open(os.path.join(root,"gan2_stats.pkl"), 'rb') as f:
        gan2_stats = pickle.load(f)
        
        for iii, item1 in enumerate(gan2_stats):
            for jjj, item2 in enumerate(gan2_stats[iii]):
                gan2_stats[iii][jjj] = gan2_stats[iii][jjj].to(device)
                
        
    with open(os.path.join(root,"gan1_stats.pkl"), 'rb') as f:
        gan1_stats = pickle.load(f)
        for iii, item1 in enumerate(gan1_stats):
            for jjj, item2 in enumerate(gan1_stats[iii]):
                gan1_stats[iii][jjj] = gan1_stats[iii][jjj].to(device)
                
        
        
    print("Done")
    return table, gan1_stats, gan2_stats