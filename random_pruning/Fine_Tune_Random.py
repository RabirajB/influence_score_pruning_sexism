import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, SchedulerType
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import os
import argparse
import evaluate
from functools import partial
import json
import gc
import warnings

###################################################
warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
###################################################

class CustomDataset(Dataset):
    def __init__(self, df, col_x, col_y):
        super(CustomDataset, self).__init__()
        self.df = df
        self.col_x = col_x
        self.col_y = col_y
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        sentence = self.df.iloc[idx][self.col_x]
        label = int(self.df.iloc[idx][self.col_y])
        return sentence, label 
    
def get_model_and_tokenizer(model_name):
    model_path = None
    if model_name == 'bert':
        model_path = 'bert-base-cased'
    if model_name == 'roberta':
        model_path = 'FacebookAI/roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_name = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels = 2)
    return tokenizer, model_name

def data_collator(batch, tokenizer):
    list_sentence, list_labels = [sentence for sentence, _ in batch], [label for _, label in batch]
    inputs = tokenizer(list_sentence, padding = 'longest', return_tensors = 'pt')
    labels = torch.LongTensor(list_labels)
    return inputs, labels 

def calculate_accuracy(model, epoch, dataloader, phase):
    metric = evaluate.load('accuracy')
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs , labels = inputs.to(device), labels.to(device)
            outputs = model(**inputs, labels = labels)
            #print(outputs.logits.shape)
            normalized_logits = nn.Softmax(dim = 1) (outputs.logits)
            normalized_logits_filter = torch.argmax(normalized_logits, dim  = 1)
            metric.add_batch(predictions = normalized_logits_filter, references = labels)
            torch.cuda.empty_cache()
            gc.collect()
    accuracy = metric.compute()['accuracy']
    print(f"The {phase} accuracy for the model in {epoch} is {accuracy}")
    return accuracy

def train(model, model_name, model_path, train_dataloader, test_dataloader, lr_scheduler_type, num_train_epochs, 
          learning_rate, gradient_accumulation_steps, max_train_steps, num_warmup_steps, run_num, prune_rate): # Keeping the epoch to low ~ 5
    
        
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    else:
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    # As of PyTorch 2.1 AdamW is now included in torch.optim package
    optimizer = AdamW(model.parameters(), lr = learning_rate) # keeping the learning rate low so as to prevent overfitting
    accuracy_epoch = {}
    lr_scheduler = get_scheduler(
        name= lr_scheduler_type,
        optimizer= optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps= max_train_steps
        )
    model.to(device)
    total_loss_epoch, completed_steps = 0, 0
    for epoch in range(1, num_train_epochs + 1):
        total_loss_epoch = 0
        if not model.training:
            model.train()
        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss_batch = model(**inputs, labels = labels).loss
            loss_batch = loss_batch / gradient_accumulation_steps
            total_loss_epoch += loss_batch.item()
            loss_batch.backward()
            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            if completed_steps >= max_train_steps:
                break
            torch.cuda.empty_cache()
        # We will be checkpointing at each epoch
        print("Training loss for epoch %d is %0.4f"%(epoch, total_loss_epoch / len(train_dataloader)))
        accuracy_epoch[epoch] = {'train' : calculate_accuracy(model, epoch, train_dataloader, phase = "train"), 
                                 'test': test(model, test_dataloader, epoch)}
        if epoch  == num_train_epochs:
            torch.save(model.state_dict(), os.path.join(model_path, f'{model_name}_classifier_{prune_rate}_{run_num}.pt'.format(epoch)))
            with open(os.path.join(model_path, f'accuracy_trace_{prune_rate}_{run_num}.json'), 'w') as accuracy_trace:
                json.dump(accuracy_epoch, accuracy_trace)
def test(model, test_dataloader, epoch):
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss_batch = model(**inputs, labels = labels).loss
            total_test_loss += loss_batch.item()
        torch.cuda.empty_cache()
        gc.collect()
    print("Test loss for epoch %d is %0.4f"%(epoch, total_test_loss/len(test_dataloader)))
    return calculate_accuracy(model, epoch, test_dataloader, phase = "test")

    
def main(model_name, path_to_dataset, train_dataset_name, test_dataset_name, test_dataset_path, model_path, x_column, y_column, lr_scheduler_type, learning_rate, 
        gradient_accumulation_steps, num_train_epochs, num_warmup_steps, max_train_steps, run_num, prune_rate):
    tokenizer, model_classifier = get_model_and_tokenizer(model_name)
    df_train = pd.read_csv(os.path.join(train_dataset_path, (train_dataset_name + f"_{prune_rate}_train.csv")))
    df_test = pd.read_csv(os.path.join(os.path.expanduser(test_dataset_path), (test_dataset_name + "_test.csv")))
    train_dataset = CustomDataset(df = df_train, col_x = x_column, col_y = y_column)
    test_dataset = CustomDataset(df = df_test, col_x = x_column, col_y = y_column)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle = True,
                                  collate_fn = partial(data_collator, tokenizer = tokenizer))
    test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = True, 
                                 collate_fn = partial(data_collator, tokenizer = tokenizer))
    train(model_classifier, model_name, model_path, train_dataloader, test_dataloader,lr_scheduler_type, num_train_epochs, 
          learning_rate, gradient_accumulation_steps, max_train_steps, num_warmup_steps, run_num, prune_rate)
    
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_path', required = True, type = str)
    parser.add_argument('--test_dataset_path', required = True, type = str)
    parser.add_argument('--model_path',  required = True, type = str)
    parser.add_argument('--train_dataset_name',  required = True, type = str)
    parser.add_argument('--test_dataset_name', required = True, type = str)
    parser.add_argument('--run_num', required = True, type = str)
    parser.add_argument('--prune_rate', required = True, type = str)
    parser.add_argument('--model_name', required = True,type = str)
    parser.add_argument(
        '--type_data', 
        required = False, 
        type = str, 
        default = None, 
        help = "The type of data we are training on type null if training on null data else just leave as it is")
    parser.add_argument('--x_column' , required = True, type = str)
    parser.add_argument('--y_column', required = True, type = str)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float, 
        default=0.0, 
        help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    
    #parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()
    model_name = args.model_name
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path 
    #test_dataset_path = os.path.join(os.path.expanduser('~/work'), test_dataset_path)
    model_path = args.model_path
    train_dataset_name = args.train_dataset_name
    test_dataset_name = args.test_dataset_name
    type_data = args.type_data
    x_column = args.x_column
    y_column = args.y_column
    learning_rate = args.learning_rate
    gradient_accumulation_steps = args.gradient_accumulation_steps
    lr_scheduler_type = args.lr_scheduler_type
    num_train_epochs = args.num_train_epochs
    num_warmup_steps = args.num_warmup_steps
    max_train_steps = args.max_train_steps
    run_num = args.run_num
    model_path = model_path + f"_{run_num}"
    prune_rate = args.prune_rate
    main(model_name, train_dataset_path, train_dataset_name, test_dataset_name, test_dataset_path, model_path, x_column, y_column, lr_scheduler_type,
         learning_rate, gradient_accumulation_steps, num_train_epochs, num_warmup_steps, max_train_steps, run_num, prune_rate)
    
    
    