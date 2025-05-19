#!/bin/bash

#python run.py \
#  --train_on sexism \
#  --model_type pcfg \
#  --preterminals 64 \
#  --nonterminals 32 \
#  --max_length 128 \
#  --epochs 40 \
#  --validate_on sexism \
#  --train_batch_size 4 \
#  --eval_batch_size 4 \
#  --eval_on_train_datasets sexism \
#  --output_dir output/sexism_data \


#python run.py \
#p --train_on ood_1 \
#p --model_type pcfg \
#p --preterminals 64 \
#p --nonterminals 32 \
#p --max_length 128 \
#p --epochs 40 \
#p --validate_on ood_1 \
#p --train_batch_size 4 \
#p --eval_batch_size 4 \
#p --eval_on_train_datasets ood_1 \
#p --output_dir output/ood_1 \


python run.py \
  --train_on sexism_eval \
  --model_type pcfg \
  --preterminals 64 \
  --nonterminals 32 \
  --max_length 128 \
  --epochs 40 \
  --validate_on ood_1 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --eval_on_train_datasets sexism_eval \
  --output_dir output/sexism_eval

python run.py \
  --train_on ood_3 \
  --model_type pcfg \
  --preterminals 64 \
  --nonterminals 32 \
  --max_length 128 \
  --epochs 40 \
  --validate_on ood_3 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --eval_on_train_datasets ood_3 \
  --output_dir output/ood_3 \


  