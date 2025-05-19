import numpy as np
import pandas as pd
import argparse
import os
import re
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit

class SemEvalDatasetTransformation(object):
    def __init__(self, input_dir, output_dir, dataset_name):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        
    def transform(self, x_col, y_col, split_col):
        df = pd.read_csv(os.path.join(self.input_dir, self.dataset_name))
        lb = LabelBinarizer()
        df['numeric_labels'] = lb.fit_transform(df[y_col])
        train_df = df[df[split_col] == 'train']
        test_df = df[df[split_col] != 'train']
        train_df[[x_col, 'numeric_labels']].to_csv(os.path.join(self.output_dir, f"{self.dataset[:self.dataset.index('.csv')]}_train.csv"), index = False)
        test_df[[x_col, 'numeric_labels']].to_csv(os.path.join(self.output_dir, f"{self.dataset[:self.dataset.index('.csv')]}_test.csv"), index = False)  
        return 'Done'
class SemEvalNullDatasetTransformation(SemEvalDatasetTransformation):
    def __init__(self, output_dir, dataset):
        super(SemEvalNullDatasetTransformation, self).__init__(output_dir, output_dir, dataset)
    
    def transform(self, y_col, split_col):
        df = pd.read_csv(os.path.join(self.input_dir, self.dataset_name))
        lb = LabelBinarizer()
        df['numeric_labels'] = lb.fit_transform(df[y_col])
        df['empty_strings'] = " "
        train_df = df[df[split_col] == 'train']
        test_df = df[df[split_col] != 'train']
        train_df[['empty_strings', 'numeric_labels']].to_csv(os.path.join(self.output_dir, f"{self.dataset[:self.dataset.index('.csv')]}_null_train.csv"), index = False)
        test_df[['empty_strings', 'numeric_labels']].to_csv(os.path.join(self.output_dir, f"{self.dataset[:self.dataset.index('.csv')]}_null_test.csv"), index = False)
        return 'Done'

class CallMeSexistButTransformation(object):
    def __init__(self, output_dir, input_dir, dataset):
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.dataset = dataset
    
    def transform(self, x_col, y_col):
        df = pd.read_csv(os.path.join(self.input_dir, self.dataset))
        dict_map = {e : i for i, e in enumerate(df[y_col].unique().tolist())}
        df['numeric_labels'] = df[y_col].map(dict_map)
        X = df[[x_col]]
        y = df[['numeric_labels']]
        splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42) # For reproducing experiments
        for train_indices, test_indices in splitter.split(X, y):
            X_train = X.loc[train_indices]
            X_test = X.loc[test_indices]
            y_train = y.loc[train_indices]
            y_test = y.loc[test_indices]
        train_df = pd.concat([X_train, y_train], axis = 1).reset_index(drop = True)
        test_df = pd.concat([X_test, y_test], axis = 1).reset_index(drop = True)
        train_df.to_csv(os.path.join(self.output_dir, f"{self.dataset[:self.dataset.index('.csv')]}_train.csv"), index = False)
        test_df.to_csv(os.path.join(self.output_dir, f"{self.dataset[:self.dataset.index('.csv')]}_test.csv"), index = False)  
        return "Done"

class CallMeSexistButNullTransformation(CallMeSexistButTransformation):
    def __init__(self, output_dir, dataset):
        super(CallMeSexistButNullTransformation, self).__init__(output_dir, output_dir, dataset)

    def transform(self, y_col):
        df = pd.read_csv(os.path.join(self.input_dir, self.dataset))
        dict_map = {e : i for i, e in enumerate(df[y_col].unique().tolist())}
        df['empty_strings'] = " "
        df['numeric_labels'] = df[y_col].map(dict_map)
        X = df[['empty_strings']]
        y = df[['numeric_labels']]
        splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42) # For reproducing experiments
        for train_indices, test_indices in splitter.split(X, y):
            X_train = X.loc[train_indices]
            X_test = X.loc[test_indices]
            y_train = y.loc[train_indices]
            y_test = y.loc[test_indices]
        train_df = pd.concat([X_train, y_train], axis = 1).reset_index(drop = True)
        test_df = pd.concat([X_test, y_test], axis = 1).reset_index(drop = True)
        train_df[['empty_strings', 'numeric_labels']].to_csv(os.path.join(self.output_dir, f"{self.dataset[:self.dataset.index('.csv')]}_null_train.csv"), index = False)
        test_df[['empty_strings', 'numeric_labels']].to_csv(os.path.join(self.output_dir, f"{self.dataset[:self.dataset.index('.csv')]}_null_test.csv"), index = False)  
        return "Done"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', help='raw_data directory', required=True, type=str)
    parser.add_argument('--x_column', help = 'x column name', required = False, type = str)
    parser.add_argument('--y_column', help = 'y column name', required = False, type = str)
    parser.add_argument('--split_col', help = 'split of the dataset not required if the dataset name is sem_eval', required = False, type = str)
    parser.add_argument('--dataset_name', help = "dataset name", required = False, type = str)
    args = parser.parse_args()
    data_dir = args.raw_data_dir
    x_col = args.x_column 
    y_col = args.y_column
    split_col = args.split_col
    dataset_name = args.dataset_name
    if dataset_name == 'sem_eval':
        SemEvalDatasetTransformation(data_dir, data_dir, dataset_name).transform(x_col, y_col, split_col)
        SemEvalNullDatasetTransformation(data_dir, dataset_name).transform(y_col, split_col)
    else:
        CallMeSexistButTransformation(data_dir, data_dir, dataset_name).transform(x_col, y_col)
        CallMeSexistButNullTransformation(data_dir, data_dir, dataset_name).transform(y_col)
        
    
    
    