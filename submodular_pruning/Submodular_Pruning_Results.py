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

def get_predicted_labels(inf_score, run_num, prune_rate, df_test, batch_size, 
                         input_col, model_name):
    model_path = f'{inf_score}_run_{run_num}'                              
    model_classifier, tokenizer = get_model_and_tokenizer(model_name)
    model_name = f'{model_name}_classifier_{prune_rate}_{run_num}.pt'
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

def get_predictions_runs(df_test, inf_score, dataset_folder, score_folder, dataset_name, model_name):
    file_name = None
    score_prune_dict = {inf_score : {'macro': defaultdict(list)}}
    for prune_rate, run_num in list(product((5, 10, 15, 20, 25, 30, 35, 40, 50, 60), (1, 2, 3, 4, 5))):
        #file_name = f'{dataset_name}_{inf_score}_{prune_rate}_{run_num}.csv'
        df_test['predicted_labels'] = get_predicted_labels(inf_score, run_num, prune_rate, df_test, 100, 'text', 
                                                            model_name)
        df_test['pred_numeric_labels'] = df_test['predicted_labels'].apply(lambda x: get_label(x))
        macro_score = f1_score(y_true = df_test['numeric_labels'], y_pred = df_test['pred_numeric_labels'], average = 'macro')
        score_prune_dict[inf_score]['macro'][prune_rate].append(macro_score)

    json_file_name = f'{inf_score}_all_scores_{model_name}.json'
    with open(os.path.join(score_folder, json_file_name), 'w') as scores_file:
        json.dump(score_prune_dict, scores_file)
    print("Done")

def get_label(x):
    return int(x[x.index('_') + 1 : ])


def main(dataset_name, dataset_folder, score_folder, model_name):
    df_name = dataset_name + ".csv"
    df_test = pd.read_csv(os.path.join(os.path.expanduser(dataset_folder), df_name))
    get_predictions_runs(df_test, inf_score = 'submod', dataset_folder = dataset_folder, 
                        score_folder = score_folder, dataset_name = dataset_name, model_name = model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', required = True, type = str)
    parser.add_argument('--dataset_name', required = True, type = str, help = "Write the name without the csv")
    parser.add_argument('--score_folder', required = True, type = str)
    parser.add_argument('--model_name', required = True, type = str)
    args = parser.parse_args()
    dataset_folder = args.dataset_folder
    model_name = args.model_name
    dataset_name = args.dataset_name
    score_folder = args.score_folder
    main(dataset_name, dataset_folder, score_folder, model_name)
    