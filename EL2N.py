import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import math 
import os
import argparse
#from datasets import load_metric
import numpy as np
from collections import defaultdict
from functools import partial
from tqdm import tqdm
import gc
from functools import partial
import argparse
from sklearn.preprocessing import OneHotEncoder
###################################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
encoder = OneHotEncoder().fit([[0], [1]])
###################################################################
def get_model_tokenizer(model_name):
    if model_name == 'bert':
        model_classifier = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels = 2)
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif model_name == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
        model_classifier = AutoModelForSequenceClassification.from_pretrained('FacebookAI/roberta-base', num_labels = 2)
    return tokenizer, model_classifier

def calculate_el2n(model, inputs, labels):
    model.eval()
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(**inputs, labels = labels)
    logits = nn.Softmax(dim = 1)(outputs.logits)
    labels = labels.reshape(labels.shape[0], 1).detach().cpu()
    one_hot_encoded = torch.Tensor(encoder.transform(labels).toarray())
    logits = logits.detach().cpu()
    one_hot_encoded = one_hot_encoded.detach().cpu()
    el2n_score_batch = torch.linalg.norm((logits - one_hot_encoded), dim  = 1)
    model.zero_grad()
    torch.cuda.empty_cache()
    gc.collect()
    return el2n_score_batch

def get_el2n_scores(model_name, model_path, checkpoint, df, batch_size = 32):
    el2n_scores = []
    tokenizer, model_classifier = get_model_tokenizer(model_name)
    model_classifier.load_state_dict(torch.load(os.path.join(model_path, f'{model_name}_classifier_{checkpoint}.pt')))
    print('Model Loaded')
    model_classifier.to(device)
    for j in range(0, len(df), batch_size):
        data = df[ j : j + batch_size]
        inputs = tokenizer(data['text'].to_list(), padding = 'longest', return_tensors = 'pt')
        labels = torch.LongTensor(data['numeric_labels'].to_list())
        el2n_score_batch = calculate_el2n(model_classifier, inputs, labels)
        for i in range(len(el2n_score_batch)):
            el2n_scores.append(el2n_score_batch[i].item())
    return el2n_scores 
'''
def get_predictions(model_name, model_path, checkpoint, input_col, df, batch_size = 32):
    model_classifier.load_state_dict(torch.load(os.path.join(model_path, f'{model_name}_classifier_{checkpoint}.pt')))
    classifier = pipeline('text-classification', model = model_classifier, tokenizer = tokenizer,
                          top_k = None, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    predicted_labels = []
    for i in range(0, len(df_test), batch_size):
        data_batch = df_test[i : i + batch_size]
        predictions = classifier_pipe(data_batch[input_col].tolist())
        for j in range(len(data_batch)):
            predicted_label = predictions[j]['label'] 
            predicted_labels.append(predicted_label)
    return predicted_labels
'''

def main(model_name, data_path, model_path, df_name, checkpoint, input_col, run_num):
    df = pd.read_csv(os.path.join(data_path, (df_name + f"_{checkpoint}.csv")))
    #calculate_vog(model_path, checkpoints, df)
    df_name = df_name + f"_{checkpoint}.csv"
    df[f'el2n_scores_{run_num}'] = get_el2n_scores(model_name, model_path, checkpoint, df)
    #df[f'predicted_labels_{run_num}'] = get_predictions(model_name, model_path, checkpoint, input_col, df)
    # Now we will calculate the class mean and then subtract it from the VOG scores
    df.to_csv(os.path.join(data_path, df_name), index = False)
    print("Saved and Done !!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, required = True)
    parser.add_argument('--model_name', type = str, required = True)
    parser.add_argument('--model_path', type = str, required  = True)
    parser.add_argument('--dataset_name', type = str, required = True)
    parser.add_argument('--run_num', type = int, required = True)
    parser.add_argument('--input_col', type = str, required = True)
    parser.add_argument('--checkpoint', type = int, required = True)
    parser.add_argument('--split', type = str, required = True)
    args = parser.parse_args()
    model_name = args.model_name
    data_path = args.data_path
    model_path = args.model_path
    df_name = args.dataset_name
    run_num = args.run_num
    input_col = args.input_col
    checkpoints = args.checkpoint
    split = args.split
    df_name = df_name + f'_{split}'
    #for checkpoint in range(1, checkpoints + 1):
    main(model_name, data_path, model_path, df_name, checkpoints, input_col, run_num)