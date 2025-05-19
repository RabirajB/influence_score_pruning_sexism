import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import torch
import pandas as pd
import os
import argparse
import warnings

################################################################
# Initializing the models, tokenizer and paths
warnings.filterwarnings('ignore')
# Saving the data path
################################################################
parser = argparse.ArgumentParser()
################################################################
def initiate_model_tokenizers(model_name):
    tokenizer = None
    model_classifier = None
    null_model_classifier = None
    if model_name == 'bert':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        model_classifier = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels = 2)
        null_model_classifier = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels = 2)    
    elif model_name == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
        model_classifier = AutoModelForSequenceClassification.from_pretrained('FacebookAI/roberta-base', num_labels = 2)
        null_model_classifier = AutoModelForSequenceClassification.from_pretrained('FacebookAI/roberta-base', num_labels = 2)
    return tokenizer, model_classifier, null_model_classifier

def v_entropy(model, data, tokenizer, input_col, label_col, batch_size = 100):
    classifier = pipeline('text-classification', model = model, tokenizer = tokenizer,
                          top_k = None, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    entropies, correct, predicted_labels = [], [], []
    
    for j in range(0, len(data), batch_size):
        batch = data[j:j+batch_size]
        predictions = classifier(batch[input_col].tolist())

        for i in range(len(batch)):
            prob = next(d for d in predictions[i] if int(d['label'][d['label'].index('_')+1:]) == batch.iloc[i][label_col])['score']
            entropies.append(-1 * np.log2(prob))
            predicted_label = max(predictions[i], key=lambda x: x['score'])['label'] 
            predicted_labels.append(predicted_label)
            correct.append(int(predicted_label[predicted_label.index('_')+1:]) == batch.iloc[i][label_col])
    return entropies, correct, predicted_labels

def v_info(data_path, data_fn, model, null_data_fn, null_model, tokenizer, out_fn, input_col, label_col, null_input_col, run_num):
    data = pd.read_csv(os.path.join(data_path, data_fn))
    null_data = pd.read_csv(os.path.join(data_path, null_data_fn))
    data['H_yb'], _, _ = v_entropy(null_model, null_data, tokenizer, 
                                   input_col = null_input_col, label_col = label_col) 
    data['H_yx'], data['correct_yx'], data['predicted_label'] = v_entropy(model, data, tokenizer, 
                                                                          input_col = input_col, label_col = label_col)
    data[f'pvi_{run_num}'] = data['H_yb'] - data['H_yx']
    if out_fn:
        data.to_csv(out_fn)

    return data

def main(data_path, model_path, model_name, epoch, data_fn, null_data_fn, input_col, label_col, null_input_col, run_num, results_path):
    tokenizer, model_classifier, null_model_classifier = initiate_model_tokenizers(model_name)
    full_model_name = f'{model_name}_classifier_{epoch}.pt'
    null_model_name = f'{model_name}_classifier_null_{epoch}.pt'
    model_classifier.load_state_dict(torch.load(os.path.join(model_path, full_model_name)))
    null_model_classifier.load_state_dict(torch.load(os.path.join(model_path, null_model_name)))
    model_classifier.eval()
    null_model_classifier.eval()
    result_data_fn = data_fn[:data_fn.index('.csv')] + '_' + f'{epoch}' + '.csv'
    out_fn = os.path.join(results_path, result_data_fn)
    data = v_info(data_path, data_fn, model = model_classifier, null_data_fn = null_data_fn, null_model = null_model_classifier,
           tokenizer = tokenizer, out_fn = out_fn, input_col = input_col, label_col = label_col, null_input_col = null_input_col, run_num = run_num)
    return data


if __name__ == '__main__':
    parser.add_argument('--dataset_name', help = 'give the name of dataset', required = True, type = str)
    parser.add_argument('--split', help = 'give the split name of the dataset' , required = True, type = str)
    parser.add_argument('--model_name', help = "name of the model", required = True, type = str)
    parser.add_argument('--checkpoint', help = "provide the last known epoch", required = True, type = int)
    parser.add_argument('--model_path', required = True, type = str)
    parser.add_argument('--data_path', required = True, type = str)
    parser.add_argument('--results_path', required = True, type = str)
    parser.add_argument('--run_num', required = True, type = int)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    split = args.split
    model_path = args.model_path
    data_path = args.data_path
    data_fn = dataset_name + f'_{split}.csv'
    null_data_fn = dataset_name + f'_null_{split}.csv'
    run_num = args.run_num
    #checkpoint = args.checkpoint
    results_path = args.results_path
    print(data_fn + " " + null_data_fn)
    checkpoints = args.checkpoint
    for checkpoint in range(1, checkpoints + 1):
        main(data_path, model_path, model_name = model_name, epoch = checkpoint, data_fn = data_fn,
            null_data_fn = null_data_fn, input_col = 'text', label_col = 'numeric_labels', null_input_col = 'empty_strings' , 
            run_num = run_num, results_path = results_path)

    