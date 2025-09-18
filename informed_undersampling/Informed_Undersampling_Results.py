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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################################################################################
def get_model_and_tokenizer(model_name):
    model_classifier, tokenizer = None, None
    if model_name == 'bert':
        model_classifier = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels = 2)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    elif model_name == 'roberta':
        model_classifier = AutoModelForSequenceClassification.from_pretrained('FacebookAI/roberta-base', num_labels = 2)
        tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
    return model_classifier, tokenizer

def get_predicted_labels(inf_score, run_num, prune_rate, df_test, batch_size, input_col, type_prune, model_name):
    model_path = None
    if type_prune == 'hard':
        model_path = f'hard_run_{run_num}'
    if type_prune == 'easy':
        model_path = f'easy_run_{run_num}'
    if type_prune == None:
        model_path = f'random_run_{run_num}'
    
    model_classifier, tokenizer = get_model_and_tokenizer(model_name)
    model_name = f'{model_name}_classifier_{inf_score}_{prune_rate}_{run_num}.pt'
    model_classifier.load_state_dict(torch.load(os.path.join(model_path, model_name)))
    classifier_pipe = pipeline('text-classification', model = model_classifier, 
                               tokenizer = tokenizer, device = device, padding = 'max_length', truncation = True)
    predicted_labels = []
    for j in range(0, len(df_test), batch_size):
        data_batch = df_test[j : j + batch_size]
        predictions = classifier_pipe(data_batch[input_col].tolist())
        for i in range(len(data_batch)):
            predicted_label = predictions[i]['label'] 
            predicted_labels.append(predicted_label)
    return predicted_labels

def get_predictions_runs(df_test, inf_score, type_prune, score_folder, dataset_name, model_name):

    if os.path.exists(score_folder):
        pass
    else:
        os.makedirs(score_folder)
    file_name, score_file_name = None, None
    score_dict = {inf_score: {'macro': defaultdict(list)}}
    for prune_rate, run_num in list(product((5, 10, 15, 20, 25, 30, 35, 40, 50, 60), (1, 2, 3, 4, 5))):
        if type_prune == None:
            file_name = f'{dataset_name}_{inf_score}_{prune_rate}_{run_num}.csv'
        else:
            file_name = f'{dataset_name}_{inf_score}_{prune_rate}_{type_prune}_{run_num}.csv'
        df_test['predicted_labels'] = get_predicted_labels(inf_score, run_num, prune_rate, df_test, 100, 'text', type_prune, model_name)
        df_test['pred_numeric_labels'] = df_test['predicted_labels'].apply(lambda x: get_label(x))
        macro_score = f1_score(y_true = df_test['numeric_labels'], y_pred = df_test['pred_numeric_labels'], average = 'macro')
        score_dict[inf_score]['macro'][prune_rate].append(macro_score)

    if type_prune is not None:
        score_file_name = f'{inf_score}_{type_prune}_all_scores.json'
    else:
        score_file_name = f'{inf_score}_all_scores.json'
    with open(os.path.join(score_folder, score_file_name), 'w') as score_file:
        json.dump(score_dict, score_file)
    print("Done")

def get_label(x):
    return int(x[x.index('_') + 1 : ])

def main(dataset_name, folder_name, score_folder, model_name):
    folder_name = os.path.join(os.path.expanduser('~/influence_score_pruning_sexism'), folder_name)
    df_name = dataset_name + ".csv"
    df_test = pd.read_csv(os.path.join(os.path.expanduser(folder_name), df_name))
    for inf_score, type_prune in list(product(('pvi', 'el2n'), ('easy', 'hard'))):
        get_predictions_runs(df_test, inf_score = inf_score, type_prune = type_prune, score_folder = score_folder,
                             dataset_name = dataset_name, model_name = model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', required = True, type = str, help="Write the name of the folder where the test dataset is stored")
    #parser.add_argument('--results_folder', required = True, type = str, help = "For storing the intermediate datasets")
    parser.add_argument('--dataset_name', required = True, type = str, help = "Write the name without the csv")
    parser.add_argument('--score_folder', required = True, type = str)
    parser.add_argument('--model_name', required = True, type = str)
    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    data_folder = args.data_folder
    #results_folder = args.results_folder
    score_folder = args.score_folder
    main(dataset_name = dataset_name, folder_name = data_folder, score_folder = score_folder, model_name = model_name)
    