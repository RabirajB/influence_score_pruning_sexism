import pandas as pd
import os
import numpy as np
import json
from collections import defaultdict
from sklearn.metrics import f1_score
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import warnings
import json
import argparse
from itertools import product

#################################################################################################
warnings.filterwarnings('ignore')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

################################################################################################
# Accomodating all the models
def get_model_and_tokenizer(model_name):
    model_classifier, tokenizer = None, None
    if model_name == 'bert':
        model_classifier = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels = 2)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    elif model_name == 'roberta':
        model_classifier = AutoModelForSequenceClassification.from_pretrained('FacebookAI/roberta-base', num_labels = 2)
        tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
    return model_classifier, tokenizer
## This code uses the model and the pipeline abstraction to get the labels
def get_predicted_labels(inf_score, run_num, prune_rate, df_test, batch_size, input_col, 
                         model_name,type_prune_sexist = None, type_prune_non_sexist = None):
    model_path = f'{inf_score}_runs'
    if not type_prune_sexist and not type_prune_non_sexist:
        model_path = f'random_run_{run_num}'
    else:
        model_path = os.path.join(model_path, f'{type_prune_sexist}_{type_prune_non_sexist}_run_{run_num}')
                                      
    model_classifier, tokenizer = get_model_and_tokenizer(model_name)
    model_name = f'{model_name}_classifier_{inf_score}_{prune_rate}_{run_num}.pt'
    model_classifier.load_state_dict(torch.load(os.path.join(model_path, model_name)))
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

def get_predictions_runs(df_test, inf_score, type_prune_sexist, type_prune_non_sexist, dataset_folder, 
                         score_folder, dataset_name, model_name):
    file_name = None
    score_prune_dict = {inf_score : {'macro': defaultdict(list)}}
    #score_prune_dict[inf_score] = {'macro': defaultdict(list)}
    #for run_num in (1, 2, 3, 4, 5):
    for prune_rate, run_num in list(product((5, 10, 15, 20, 25, 30, 35, 40, 50, 60), (1, 2, 3, 4, 5))):
        if not type_prune_sexist and not type_prune_non_sexist:
            file_name = f'{dataset_name}_{inf_score}_{prune_rate}_{run_num}.csv'
        else:
            file_name = f'{dataset_name}_{inf_score}_{prune_rate}_{type_prune_sexist}_{type_prune_non_sexist}_{run_num}.csv'
        df_test['predicted_labels'] = get_predicted_labels(inf_score, run_num, prune_rate, df_test, 100, 'text', 
                                                            model_name, type_prune_sexist, type_prune_non_sexist)
        df_test['pred_numeric_labels'] = df_test['predicted_labels'].apply(lambda x: get_label(x))
        macro_score = f1_score(y_true = df_test['numeric_labels'], y_pred = df_test['pred_numeric_labels'], average = 'macro')
        score_prune_dict[inf_score]['macro'][prune_rate].append(macro_score)

    json_file_name = f'{inf_score}_{type_prune_sexist}_{type_prune_non_sexist}_all_scores.json'
    with open(os.path.join(score_folder, json_file_name), 'w') as scores_file:
        json.dump(score_prune_dict, scores_file)

    print("Done")
        
            
    #df_test.to_csv(os.path.join(folder_name, file_name), index = False)

def get_label(x):
    return int(x[x.index('_') + 1 : ])
    
def main(dataset_name, dataset_folder, score_folder, model_name):
    df_name = dataset_name + ".csv"
    df_test = pd.read_csv(os.path.join(os.path.expanduser(dataset_folder), df_name))
    type_prunes_sexist = ['hard', 'easy']
    type_prunes_non_sexist = ['hard', 'easy']
    #print(list(product(type_prunes_sexist, type_prunes_non_sexist)))
    for p in list(product(type_prunes_sexist, type_prunes_non_sexist)):
        
        get_predictions_runs(df_test, inf_score = 'pvi', type_prune_sexist = p[0], type_prune_non_sexist = p[1], dataset_folder = dataset_folder,
                             score_folder = score_folder, dataset_name = dataset_name, model_name = model_name)
        #get_predictions_runs(df_test, 'vog', type_prune_sexist = p[0], type_prune_non_sexist = p[1], folder_name = folder_name,
        #                     dataset_name = dataset_name, model_name = model_name)
        get_predictions_runs(df_test, inf_score = 'el2n', type_prune_sexist = p[0], type_prune_non_sexist = p[1], dataset_folder = dataset_folder,
                            score_folder = score_folder, dataset_name = dataset_name, model_name = model_name)
        #get_f1_scores('pvi', p[0], p[1], score_folder, dataset_name)
        #get_f1_scores('el2n', p[0], p[1], score_folder, dataset_name)
        #get_f1_scores('vog', p[0], p[1], folder_name, dataset_name)
        #get_f1_scores('random', None, folder_name, dataset_name)
        #store_jsons(inf_score = 'random', type_prune = None, folder_name = score_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', required = True, type = str)
    parser.add_argument('--dataset_name', required = True, type = str, help = "Write the name without the csv")
    parser.add_argument('--score_folder', required = True, type = str)
    parser.add_argument('--model_name', required = True, type = str)
    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    dataset_folder = args.dataset_folder
    score_folder = args.score_folder
    main(dataset_name, dataset_folder, score_folder, model_name)
    