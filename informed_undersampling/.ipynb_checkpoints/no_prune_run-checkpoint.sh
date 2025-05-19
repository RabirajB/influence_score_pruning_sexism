#!/bin/bash
export dataset_name="sexist_data_test"
export data_folder="sexism_data"
export score_folder="sexism_data_f1_scores"
export model_name="bert"
python 0_Prune_Results.py
--dataset_name $dataset_name \
--data_folder $data_folder \
--score_folder $score_folder \
--model_name $model_name

export dataset_name="ood_1"
export data_folder="ood_datasets"
export score_folder="ood_1_f1_scores"
export model_name="bert"

python 0_Prune_Results.py \
--dataset_name $dataset_name \
--data_folder $data_folder \
--score_folder $score_folder \
--model_name $model_name

export dataset_name="ood_2"
export data_folder="ood_datasets"
export score_folder="ood_2_f1_scores"
export model_name="bert"

python 0_Prune_Results.py \
--dataset_name $dataset_name \
--data_folder $data_folder \
--score_folder $score_folder \
--model_name $model_name


export dataset_name="sexism_eval"
export data_folder="ood_datasets"
export score_folder="sexism_eval_f1_scores"
export model_name="bert"

python 0_Prune_Results.py \
--dataset_name $dataset_name \
--data_folder $data_folder \
--score_folder $score_folder \
--model_name $model_name

export dataset_name="ood_3"
export data_folder="~/Influence_Scores/ood_datasets"
export score_folder="ood_3_f1_scores"
export model_name="bert"

python 0_Prune_Results.py \
--dataset_name $dataset_name \
--data_folder $data_folder \
--score_folder $score_folder \
--model_name $model_name

 