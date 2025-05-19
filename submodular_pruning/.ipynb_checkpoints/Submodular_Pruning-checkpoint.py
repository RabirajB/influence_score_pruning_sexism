import pandas as pd
import numpy as np
import os
import torch
import pickle
import math
from submodlib import FacilityLocationFunction
import argparse


def get_embeds(embedding_path, dataet_path, dataset_name, run_num):
    with open(os.path.join(os.path.expanduser(embedding_path), f'cmsb_embeds_{run_num}.pkl'), 'rb') as embeds_file:
        embeds = pickle.load(embeds_file)
    df_train = pd.read_csv(os.path.join(os.path.expanduser(dataset_path), dataset_name))
    identities = df_train['id'].to_list()
    tensor = embeds[identities[0]].unsqueeze(0)
    for identity in identities[1:]:
        tensor = torch.cat((tensor, embeds[identity].unsqueeze(0)), dim  = 0)
    return tensor

def get_subsets(budget, data_train, objFL):
    #print('Hit')
    subset_list = objFL.maximize(budget = budget, optimizer = 'LazierThanLazyGreedy')
    #print('Hit')
    dict_id_gains = {'idx': [idx for idx, _ in subset_list], 'gain' : [gain for _, gain in subset_list]}
    train_df_subset = data_train[data_train.index.isin(dict_id_gains['idx'])].reset_index(drop = True)
    train_df_subset['gain'] = dict_id_gains['gain']
    return train_df_subset

def main(embedding_path, dataset_path, dataset_name, run_num):
    dataset_name = dataset_name + ".csv"
    df_train = pd.read_csv(os.path.join(os.path.expanduser(dataset_path), dataset_name))
    embeds_tensor = get_embeds(embedding_path, dataset_path, dataset_name, run_num)
    embeds_tensor = embeds_tensor.to(torch.float16)
    objFL_data = FacilityLocationFunction(n = embeds_tensor.shape[0], data = embeds_tensor, mode = 'dense', metric = 'cosine')
    storage_path = f'submod_data_{run_num}'
    for prune_rate in (5, 10, 15, 20, 25, 30, 35, 40, 50, 60):
        budget = math.ceil((1 - (prune_rate / 100)) * embeds_tensor.shape[0])
        df_subset = get_subsets(budget, df_train, objFL_data)
        df_subset.to_csv(os.path.join(storage_path, f'sexist_data_submodular_{prune_rate}_train.csv'), index  = False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', required = True, type = str)
    parser.add_argument('--dataset_name', required = True, type = str, help = "Write the name without the csv")
    parser.add_argument('--dataset_path', required = True, type = str)
    parser.add_argument('--run_num', required = True, type = str)
    args = parser.parse_args()
    embedding_path = args.embedding_path
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    run_num = args.run_num
    main(embedding_path, dataset_path, dataset_name, run_num)
            
            
            
            
            
            
            


    
    



    
    
    