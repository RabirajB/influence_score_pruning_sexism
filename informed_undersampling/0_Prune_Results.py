import json
import torch
from collections import defaultdict
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import numpy as np
from sklearn.metrics import f1_score
import warnings
import argparse
import os

###################################################################################################
warnings.filterwarnings('ignore')
##################################################################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
f1_score_dict = defaultdict(list)
##################################################################################################
def get_model_tokenizer(model_name):
    tokenizer, model_classifier = None, None
    if model_name == 'bert':
        model_classifier = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels = 2)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    if model_name == 'roberta':
        model_classifier = AutoModelForSequenceClassification.from_pretrained('FacebookAI/roberta-base', num_labels = 2)
        tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
    return tokenizer, model_classifier
    
def get_initial_f1_score(run_num, model_name, df_test, input_col, batch_size = 100):
    model_path = os.path.join(os.path.expanduser('~/influence_score_pruning_sexism'), f'cmsb_run_{run_num}')
    tokenizer, model_classifier = get_model_tokenizer(model_name)
    full_model_name = f'{model_name}_classifier_5.pt'
    model_classifier.load_state_dict(torch.load(os.path.join(model_path, full_model_name)))
    classifier_pipe = pipeline('text-classification', model = model_classifier, tokenizer = tokenizer, 
                               device = device, padding = 'max_length', truncation = True)
    predicted_labels = []
    for j in range(0, len(df_test), batch_size):
        data_batch = df_test[j : j + batch_size]
        predictions = classifier_pipe(data_batch[input_col].tolist())
        for i in range(len(data_batch)):
            predicted_label = predictions[i]['label'] 
            predicted_labels.append(predicted_label)
    return predicted_labels

def get_label(s):
    return int(s[s.index('_') + 1 : ])

def get_f1_score_runs(df_test, data_folder, model_name):
    for run_num in  (1, 2, 3, 4, 5):
        df_test['predicted_labels'] = get_initial_f1_score(run_num, model_name, df_test, 'text')
        df_test['pred_numeric_labels'] = df_test['predicted_labels'].apply(lambda x: get_label(x))
        df_test.to_csv(os.path.join(data_folder, f'0_prune_results_run_{run_num}.csv'), index = False)
        score = f1_score(y_true = df_test['numeric_labels'],
                         y_pred = df_test['pred_numeric_labels'], average = 'macro')
        f1_score_dict[0].append(score)

def main(dataset_name, data_folder, score_folder, model_name):
    if os.path.exists(score_folder):
        pass
    else:
        os.makedirs(score_folder)
    data_folder = os.path.join(os.path.expanduser('~/influence_score_pruning_sexism'), data_folder)
    dataset_name = dataset_name + ".csv"
    df_test = pd.read_csv(os.path.join(os.path.expanduser(data_folder), dataset_name))
    get_f1_score_runs(df_test, data_folder, model_name)
    with open(os.path.join(score_folder, '0_prune.json'), 'w') as no_prune_json:
        json.dump(f1_score_dict, no_prune_json)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required = True, type = str)
    parser.add_argument('--data_folder', required = True, type = str)
    parser.add_argument('--score_folder', required = True, type = str)
    parser.add_argument('--model_name', required = True, type = str)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    score_folder = args.score_folder
    data_folder = args.data_folder
    model_name = args.model_name
    main(dataset_name, data_folder, score_folder, model_name)
    
    
    

