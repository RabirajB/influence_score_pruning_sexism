import pandas as pd
import numpy as np
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import argparse
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import gc
from functools import partial

#################################################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#################################################################################
class CustomDataset(Dataset):
    def __init__(self, df, text_id, text):
        super(CustomDataset, self).__init__()
        self.df = df
        self.text_id = text_id
        self.text = text
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        text_id = self.df.iloc[idx][self.text_id]
        text = self.df.iloc[idx][self.text]
        return text_id, text

def get_model_and_tokenizer(model_name):
    model_classifier, tokenizer = None, None
    if model_name == 'bert':
        model_classifier = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels = 2)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    elif model_name == 'roberta':
        model_classifier = AutoModelForSequenceClassification.from_pretrained('FacebookAI/roberta-base', num_labels = 2)
        tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
        
    return model_classifier, tokenizer
    

def load_embeddings(embed_tensor, sentence_ids, tensor_dict):
    for i, sentence_id in enumerate(sentence_ids):
        if sentence_id not in tensor_dict:
            tensor_dict[sentence_id] = embed_tensor[i]
        else:
            continue

def data_collator(batch, tokenizer):
    list_ids, list_text = [text_id for text_id, _ in batch], [text for _, text in batch]
    inputs = tokenizer(list_text, padding = 'longest', return_tensors = 'pt', truncation = True)
    return list_ids, inputs


def get_model_embeddings(model_path, dataloader, model_name, dataset_name, file_name):
    model_classifier, _ = get_model_and_tokenizer(model_name)
    for run_num in range(1, 6):
        tensor_dict = {}
        model_path_run = model_path + f'_{run_num}'
        model_name_full = model_name + '_classifier_5.pt'
        model_classifier.load_state_dict(torch.load(os.path.join(os.path.expanduser(model_path_run), model_name_full), map_location = device))
        model_classifier.eval()
        for text_ids, inputs in dataloader:
            with torch.no_grad():
                sentence_embeddings = model_classifier(**inputs, output_hidden_states = True)['hidden_states'][-1][:, 0, :].detach().cpu()
            load_embeddings(sentence_embeddings, text_ids, tensor_dict)
        torch.cuda.empty_cache()
        gc.collect()
        #Saving the individual embeddings
        
        with open(os.path.join(os.path.expanduser(model_path_run), f'{file_name}_embeds_{run_num}.pkl'), 'wb') as embedding_file:
            pickle.dump(tensor_dict, embedding_file)

def main(dataset_path, dataset_name, model_path, model_name, file_name):
    _, tokenizer = get_model_and_tokenizer(model_name)
    dataset_full_name = dataset_name + ".csv"
    df = pd.read_csv(os.path.join(os.path.expanduser(dataset_path), dataset_full_name))
    dataset = CustomDataset(df, 'id', 'text')
    dataloader = DataLoader(dataset, batch_size=64, shuffle = True, collate_fn = partial(data_collator, tokenizer = tokenizer))
    tensor_dict = get_model_embeddings(model_path, dataloader, model_name, dataset_name, file_name)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required = True, type = str)
    parser.add_argument('--dataset_name', required = True, type = str, help = "Write the name without the csv")
    parser.add_argument('--dataset_path', required = True, type = str)
    parser.add_argument('--model_name', required = True, type = str)
    parser.add_argument('--file_name', required = True, type = str)
    args = parser.parse_args()
    model_path = args.model_path
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    model_name = args.model_name
    file_name = args.file_name
    main(dataset_path, dataset_name, model_path, model_name, file_name)
    
    
    
    
    
    
    
    
        
        
        