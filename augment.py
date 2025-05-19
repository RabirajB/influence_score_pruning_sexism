"""
This module is used for cleaning dataset files and transforming the
input to extract a particular attribute (e.g., the hypothesis-premise
overlap in SNLI).

Each dataset has a parent class in which the cleaning is done and several
subclasses, one for each transformation.
"""
import string
import spacy
import os
import re
import pandas as pd
import random
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelBinarizer

parser = argparse.ArgumentParser()

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')


class SNLITransformation(object):
    """
    Parent class for transforming the SNLI input to extract some particular input
    attribute (e.g., just the hypothesis by leaving out the premise). Also needed
    to reformat the input into a single string. The transformed data is saved as a CSV.
    """
    def __init__(self, name, output_dir, train_size=1.0):
        """
        Args:
            name: Transformation name
            output_dir: where to save the CSV with the transformed attribute
            train_size: fraction of the training data to use
        """
        self.train_data = load_dataset('snli', split='train').filter(lambda x: x['label'] != -1)
        self.test_data = load_dataset('snli', split='test').filter(lambda x: x['label'] != -1)
        self.name = name
        self.output_dir = output_dir
        self.train_size = train_size

    def transformation(self, example):
        raise NotImplementedError

    def transform(self):
        logging.info(f'Applying {self.name} to SNLI')

        if self.train_size < 1:
            train_data = self.train_data.train_test_split(train_size=self.train_size)['train']
        else:
            train_data = self.train_data

        train_data.map(self.transformation).to_csv(os.path.join(self.output_dir, f'snli_train_{self.name}' + (f'_{self.train_size}' if self.train_size < 1.0 else '') + '.csv'))
        self.test_data.map(self.transformation).to_csv(os.path.join(self.output_dir, f'snli_test_{self.name}.csv'))


class MultiNLITransformation(object):
    """
    Parent class for transforming the MNLI input to extract some particular input
    attribute (e.g., just the hypothesis by leaving out the premise). Also needed
    to reformat the input into a single string. The transformed data is saved as a CSV.
    """
    def __init__(self, name, output_dir):
        self.train_data = load_dataset('multi_nli', split='train').filter(lambda x: x['label'] != -1)
        self.validation_data = load_dataset('multi_nli', split='validation_matched').filter(lambda x: x['label'] != -1)
        self.name = name
        self.output_dir = output_dir

    def transformation(self, example):
        raise NotImplementedError

    def transform(self):
        logging.info(f'Applying {self.name} to MutliNLI')
        self.train_data.map(self.transformation).to_pandas()[['sentence1', 'label']].to_csv(
            os.path.join(self.output_dir, f'multinli_train_{self.name}.csv'))
        self.validation_data.map(self.transformation).to_pandas()[['sentence1', 'label']].to_csv(
            os.path.join(self.output_dir, f'multinli_validation_{self.name}.csv'))


class DWMWTransformation(object):
    def __init__(self, name, output_dir):
        self.data = pd.read_csv('data/dwmw/labeled_data.csv').rename({"tweet" : "sentence1", "class" : "label"}, axis=1)
        self.name = name
        self.output_dir = output_dir

    def transformation(self, example):
        raise NotImplementedError

    def transform(self):
        logging.info(f'Applying {self.name} to DWMW')
        self.data.apply(self.transformation, axis=1).to_csv(
            os.path.join(self.output_dir, f'dwmw_{self.name}.csv'))


class COLATransformation(object):
    def __init__(self, name, output_dir, train_size=1):
        self.train_data = pd.read_csv('data/cola_public/raw/in_domain_train.tsv', sep='\t', names=['annotator', 'label', 'stars', 'sentence1'])
        self.id_dev_data = pd.read_csv('data/cola_public/raw/in_domain_dev.tsv', sep='\t', names=['annotator', 'label', 'stars', 'sentence1'])
        self.ood_dev_data = pd.read_csv('data/cola_public/raw/out_of_domain_dev.tsv', sep='\t', names=['annotator', 'label', 'stars', 'sentence1'])
        self.name = name
        self.output_dir = output_dir
        self.train_size = train_size

    def transformation(self, example):
        raise NotImplementedError

    def transform(self):
        logging.info(f'Applying {self.name} to COLA')

        self.train_data.sample(frac=self.train_size, random_state=1).apply(self.transformation, axis=1).to_csv(
            os.path.join(self.output_dir, f'cola_train_{self.name}' + (f'_{self.train_size}' if self.train_size < 1.0 else '') + '.csv'))
        self.id_dev_data.apply(self.transformation, axis=1).to_csv(
            os.path.join(self.output_dir, f'cola_id_dev_{self.name}.csv'))
        self.ood_dev_data.apply(self.transformation, axis=1).to_csv(
            os.path.join(self.output_dir, f'cola_ood_dev_{self.name}.csv'))


class SNLIStandardTransformation(SNLITransformation):
    def __init__(self, output_dir, train_size=1, suffix=''):
        super().__init__(f'std{suffix}', output_dir, train_size=train_size)

    def transformation(self, example):
        example['sentence1'] = f"PREMISE: {example['premise']} HYPOTHESIS: {example['hypothesis']}"
        return example

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
        df_train = pd.read_csv(os.path.join(self.input_dir, f"{self.dataset[:self.dataset.index('.csv')]}_train.csv"), index = False)
        df_test = pd.read_csv(os.path.join(self.input_dir, f"{self.dataset[:self.dataset.index('.csv')]}_test.csv"), index = False)
        df_train['empty_strings'] = " "
        df_test["empty_strings"] = " "
        df_train[['empty_strings', 'numeric_labels']].to_csv(os.path.join(self.output_dir, f"{self.dataset[:self.dataset.index('.csv')]}_null_train.csv"), index = False)
        df_test[['empty_strings', 'numeric_labels']].to_csv(os.path.join(self.output_dir, f"{self.dataset[:self.dataset.index('.csv')]}_null_test.csv"), index = False)  
        return "Done"

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
        train_df[[x_col, 'numeric_labels']].to_csv(os.path.join(self.output_dir, f"{self.dataset_name[:self.dataset_name.index('.csv')]}_train.csv"), index = False)
        test_df[[x_col, 'numeric_labels']].to_csv(os.path.join(self.output_dir, f"{self.dataset_name[:self.dataset_name.index('.csv')]}_test.csv"), index = False)  
        return 'Done'

class SemEvalNullDatasetTransformation(SemEvalDatasetTransformation):
    def __init__(self, output_dir, dataset_name):
        super(SemEvalNullDatasetTransformation, self).__init__(output_dir, output_dir, dataset_name)
    
    def transform(self):
        train_df = pd.read_csv(os.path.join(self.input_dir, f"{self.dataset[:self.dataset.index('.csv')]}_train.csv"), index = False)
        test_df = pd.read_csv(os.path.join(self.input_dir, f"{self.dataset[:self.dataset.index('.csv')]}_test.csv"), index = False)
        train_df["empty_strings"] = " "
        test_df["empty_strings"] = " "
        train_df[['empty_strings', 'numeric_labels']].to_csv(os.path.join(self.output_dir, f"{self.dataset_name[:self.dataset_name.index('.csv')]}_null_train.csv"), index = False)
        test_df[['empty_strings', 'numeric_labels']].to_csv(os.path.join(self.output_dir, f"{self.dataset_name[:self.dataset_name.index('.csv')]}_null_test.csv"), index = False)
        return 'Done'

class SNLINullTransformation(SNLITransformation):
    def __init__(self, output_dir, train_size=1, suffix=''):
        super().__init__(f'null{suffix}', output_dir, train_size=train_size)

    def transformation(self, example):
        example['sentence1'] = " " # using only empty string can yield problems
        return example


class SNLIHypothesisOnlyTransformation(SNLITransformation):
    def __init__(self, output_dir):
        super().__init__('hypothesis', output_dir)

    def transformation(self, example):
        example['sentence1'] = f"HYPOTHESIS: {example['hypothesis']}"
        return example


class SNLIPremiseOnlyTransformation(SNLITransformation):
    def __init__(self, output_dir):
        super().__init__('premise', output_dir)

    def transformation(self, example):
        example['sentence1'] = f"PREMISE: {example['premise']}"
        return example


class SNLIOverlapTransformation(SNLITransformation):
    def __init__(self, output_dir):
        super().__init__('overlap', output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased') 

    def transformation(self, example):
        hypothesis_tokens = self.tokenizer.tokenize(example['hypothesis'])
        overlap_tokens = [ t for t in hypothesis_tokens if t in self.tokenizer.tokenize(example['premise']) ]
        overlap = len(overlap_tokens) / len(hypothesis_tokens)

        if overlap >= 0.75:
            msg = "HIGH OVERLAP"
        elif overlap >= 0.5:
            msg = "MEDIUM OVERLAP"
        elif overlap >= 0.25:
            msg = "LOW OVERLAP"
        else:
            msg = "NO OVERLAP"

        example['sentence1'] = f"{msg}."	
        return example


class SNLIRawOverlapTransformation(SNLITransformation):
    def __init__(self, output_dir):
        super().__init__('raw_overlap', output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased') 

    def transformation(self, example):
        hypothesis_tokens = self.tokenizer.tokenize(example['hypothesis'])
        premise_tokens = self.tokenizer.tokenize(example['premise']) 
        overlap = set(hypothesis_tokens) & set(premise_tokens)
        hypothesis = " ".join([ (t if t in overlap else self.tokenizer.mask_token) for t in hypothesis_tokens ])
        premise = " ".join([ (t if t in overlap else self.tokenizer.mask_token) for t in premise_tokens ])
        example['sentence1'] = f"PREMISE: {premise} HYPOTHESIS: {hypothesis}"	
        return example


class SNLIShuffleTransformation(SNLITransformation):
    def __init__(self, output_dir):
        super().__init__('shuffled', output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased') 

    def transformation(self, example):
        """
        Randomly reorder the words in the hypothesis and premise.
        """
        hyp = self.tokenizer.tokenize(example['hypothesis'])
        random.shuffle(hyp)
        hyp = self.tokenizer.convert_tokens_to_string(hyp)

        prem = self.tokenizer.tokenize(example['premise'])
        random.shuffle(prem)
        prem = self.tokenizer.convert_tokens_to_string(prem)

        example['sentence1'] = f"PREMISE: {prem} HYPOTHESIS: {hyp}"
        return example


class SNLILengthTransformation(SNLITransformation):
    def __init__(self, output_dir):
        super().__init__('length', output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased') 

    def transformation(self, example):
        hyp = ' '.join(['#'] * len(self.tokenizer.tokenize(example['hypothesis'])))
        prem = ' '.join(['#'] * len(self.tokenizer.tokenize(example['premise'])))
        example['sentence1'] = f"PREMISE: {prem} HYPOTHESIS: {hyp}"
        return example


class MultiNLIStandardTransformation(MultiNLITransformation):
    def __init__(self, output_dir):
        super().__init__('std', output_dir)

    def transformation(self, example):
        example['sentence1'] = f"PREMISE: {example['premise']} HYPOTHESIS: {example['hypothesis']}"
        return example


class MultiNLINullTransformation(MultiNLITransformation):
    def __init__(self, output_dir):
        super().__init__('null', output_dir)

    def transformation(self, example):
        example['sentence1'] = " " # using only empty string can yield problems
        return example


class DWMWStandardTransformation(DWMWTransformation):
    def __init__(self, output_dir):
        super().__init__('std', output_dir)

    def transformation(self, example):
        return example
         

class DWMWNullTransformation(DWMWTransformation):
    def __init__(self, output_dir):
        super().__init__('null', output_dir)

    def transformation(self, example):
        example['sentence1'] = " " # using only empty string can yield problems
        return example


class DWMWVocabTransformation(DWMWTransformation):
    def __init__(self, output_dir):
        super().__init__('bad_vocab', output_dir)
        # potentially offensive words were manually selected
        self.bad_words = [ 'nigga', 'niggas', 'niggah', 'niggahs', 'hoe', 'hoes', 'bitch', 'bitches', 'whitey', 'white trash', 'cracker', 'crackers', 'beaner', 'beaners',
            'pussy', 'pussies', 'fag', 'fags', 'faggot', 'faggots', 'ho', 'hos', 'redneck', 'rednecks', 'porn', 'fuck', 'fucks', 'fucker', 'fuckers', 'motherfucker', 
            'motherfuckers', 'nigger', 'niggers', 'coon', 'coons', 'niggaz', 'nig', 'nigs', 'slut', 'sluts', 'wigger', 'wiggers', 'fucked', 'fucking', 'wigga', 'wiggas',
            'retard', 'retards', 'retarded' ] 

    def transformation(self, example):
        pattern = re.compile(rf"(?!\b({'|'.join(self.bad_words)})\b)\b[^ ]+\b", re.IGNORECASE)
        example['sentence1'] = re.sub(pattern, "", example['sentence1'])
        example['sentence1'] = example['sentence1'].translate(str.maketrans('', '', string.punctuation))
        example['sentence1'] = example['sentence1'].strip()

        if example['sentence1'] == "":
            example['sentence1'] = ' ' #using only empty string can yield problems

        return example


class DWMWSentimentVocabTransformation(DWMWTransformation):
    def __init__(self, output_dir):
        super().__init__('sentiment_vocab', output_dir)
        self.bad_vocab = DWMWVocabTransformation(output_dir)

    def transformation(self, example):
        polarity = nlp(example['sentence1'])._.polarity 

        if -0.10 <= polarity <= 0.10:
            sentiment = 'neutral'
        elif polarity > 0.10:
            sentiment = 'positive'
        else:
            sentiment = 'negative'

        example['sentence1'] = ' '.join([sentiment, self.bad_vocab.transformation(example)['sentence1']])

        if example['sentence1'] == "":
            example['sentence1'] = ' ' #using only empty string can yield problems

        return example


class DWMWSentimentTransformation(DWMWTransformation):
    def __init__(self, output_dir):
        super().__init__('sentiment', output_dir)
        self.bad_vocab = DWMWVocabTransformation(output_dir)

    def transformation(self, example):
        polarity = nlp(example['sentence1'])._.polarity 

        if -0.10 <= polarity <= 0.10:
            sentiment = 'neutral'
        elif polarity > 0.10:
            sentiment = 'positive'
        else:
            sentiment = 'negative'

        example['sentence1'] = sentiment

        return example

class COLAStandardTransformation(COLATransformation):
    def __init__(self, output_dir, train_size=1):
        super().__init__('std', output_dir, train_size)

    def transformation(self, example):
        return example
         

class COLANullTransformation(COLATransformation):
    def __init__(self, output_dir, train_size=1):
        super().__init__('null', output_dir, train_size)

    def transformation(self, example):
        example['sentence1'] = " " # using only empty string can yield problems
        return example


class COLAShuffleTransformation(COLATransformation):
    def __init__(self, output_dir, train_size=1):
        super().__init__('shuffled', output_dir, train_size)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased') 

    def transformation(self, example):
        """
        Randomly reorder the words in the hypothesis and premise.
        """
        sentence = self.tokenizer.tokenize(example['sentence1'])
        random.shuffle(sentence)
        example['sentence1'] = self.tokenizer.convert_tokens_to_string(sentence)
        return example


if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)   

    parser.add_argument('--raw_data_dir', help='raw_data directory', required=True, type=str)
    parser.add_argument('--x_column', help = 'x column name', required = False, type = str)
    parser.add_argument('--y_column', help = 'y column name', required = False, type = str)
    parser.add_argument('--dataset_name', help = "dataset name", required = False, type = str)
    parser.add_argument('--split_col', help = 'Column to Split on', required = False, type  = str, default = None) 
    args = parser.parse_args()
    data_dir = args.raw_data_dir
    x_col = args.x_column 
    y_col = args.y_column
    dataset_name = args.dataset_name
    split_col = args.split_col
    if "sem_eval" in dataset_name:
        SemEvalDatasetTransformation(data_dir, data_dir, dataset_name).transform(x_col, y_col, split_col)
        SemEvalNullDatasetTransformation(data_dir, dataset_name).transform(y_col, split_col)
    # Handling the case of no string and empty string inputs
    elif x_col and y_col and len(x_col) > 0 and len(y_col) > 0 and dataset_name == 'sexism_data.csv': 
        CallMeSexistButTransformation(data_dir, data_dir, dataset= dataset_name).transform(x_col, y_col)
        CallMeSexistButNullTransformation(data_dir, dataset = dataset_name).transform(y_col)
    else:  
        SNLIStandardTransformation(data_dir).transform()
        SNLINullTransformation(data_dir).transform()
        SNLIHypothesisOnlyTransformation(data_dir).transform()
        SNLIPremiseOnlyTransformation(data_dir).transform()
        SNLIRawOverlapTransformation(data_dir).transform()
        SNLIShuffleTransformation(data_dir).transform()

        DWMWStandardTransformation(data_dir).transform()
        DWMWNullTransformation(data_dir).transform()
        DWMWVocabTransformation(data_dir).transform()
        DWMWSentimentVocabTransformation(data_dir).transform()
        DWMWSentimentTransformation(data_dir).transform()
    
        COLAStandardTransformation(data_dir).transform()
        COLANullTransformation(data_dir).transform()
        COLAShuffleTransformation(data_dir).transform()
    
        MultiNLIStandardTransformation(data_dir).transform()
        MultiNLINullTransformation(data_dir).transform()
        

        for suffix in ['_b', '_c', '_d', '_e']:
            SNLIStandardTransformation(f'{data_dir}/frac', train_size=0.99, suffix=suffix).transform()
            SNLIStandardTransformation(f'{data_dir}/frac', train_size=0.8, suffix=suffix).transform()
            SNLIStandardTransformation(f'{data_dir}/frac', train_size=0.6, suffix=suffix).transform()
            SNLIStandardTransformation(f'{data_dir}/frac', train_size=0.4, suffix=suffix).transform()
            SNLIStandardTransformation(f'{data_dir}/frac', train_size=0.2, suffix=suffix).transform()
            SNLIStandardTransformation(f'{data_dir}/frac', train_size=0.05, suffix=suffix).transform()

            SNLINullTransformation(f'{data_dir}/frac', train_size=0.99, suffix=suffix).transform()
            SNLINullTransformation(f'{data_dir}/frac', train_size=0.8, suffix=suffix).transform()
            SNLINullTransformation(f'{data_dir}/frac', train_size=0.6, suffix=suffix).transform()
            SNLINullTransformation(f'{data_dir}/frac', train_size=0.4, suffix=suffix).transform()
            SNLINullTransformation(f'{data_dir}/frac', train_size=0.2, suffix=suffix).transform()
            SNLINullTransformation(f'{data_dir}/frac', train_size=0.05, suffix=suffix).transform()
