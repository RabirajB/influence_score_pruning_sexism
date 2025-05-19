#!/bin/bash

export model_path='~/influence_scores_dataset_pruning/cmsb_run'
export dataset_name='sexist_data_final_train'
export dataset_path='~/influence_scores_dataset_pruning/score_results'
export model_name="bert"

python Get_Model_Embeddings.py \
--model_path $model_path \
--dataset_name $dataset_name \
--dataset_path $dataset_path \
--model_name $model_name