#!/bin/bash

export model_path='~/influence_score_pruning_sexism/cmsb_run'
export dataset_name='sexist_data_final_train'
export dataset_path='~/influence_score_pruning_sexism/score_results'
export model_name="roberta"
export file_name="cmsb"

python Get_Model_Embeddings.py \
--model_path $model_path \
--dataset_name $dataset_name \
--dataset_path $dataset_path \
--file_name $file_name \
--model_name $model_name