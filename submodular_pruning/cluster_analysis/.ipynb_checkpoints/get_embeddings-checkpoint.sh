#!/bin/bash
export dataset_path="~/Influence_Scores/sexism_data"
export dataset_name="sexist_data_train"
export model_name="roberta"
export model_path="~/Influence_Scores/cmsb_run"

python Get_Model_Embeddings.py \
--dataset_path $dataset_path \
--dataset_name $dataset_name \
--model_name $model_name \
--model_path $model_path