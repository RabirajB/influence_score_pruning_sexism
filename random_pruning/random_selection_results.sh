#!/bin/bash
export dataset_folder="~/Influence_Scores/sexism_data"
export dataset_name="sexist_data_test"
export score_folder="sexism_data_f1_scores"
export model_name="bert"

python Random_Pruning_Results.py \
--dataset_folder $dataset_folder \
--dataset_name $dataset_name \
--score_folder $score_folder \
--model_name $model_name

export dataset_folder="~/Influence_Scores/ood_datasets"
export dataset_name="ood_1"
export score_folder="ood_1_f1_scores"
export model_name="bert"

python Random_Pruning_Results.py \
--dataset_folder $dataset_folder \
--dataset_name $dataset_name \
--score_folder $score_folder \
--model_name $model_name


export dataset_folder="~/Influence_Scores/ood_datasets"
export dataset_name="ood_2"
export score_folder="ood_2_f1_scores"
export model_name="bert"

python Random_Pruning_Results.py \
--dataset_folder $dataset_folder \
--dataset_name $dataset_name \
--score_folder $score_folder \
--model_name $model_name

export dataset_folder="~/Influence_Scores/ood_datasets"
export dataset_name="sexism_eval"
export score_folder="sexism_eval_f1_scores"
export model_name="bert"

python Random_Pruning_Results.py \
--dataset_folder $dataset_folder \
--dataset_name $dataset_name \
--score_folder $score_folder \
--model_name $model_name

export dataset_folder="~/Influence_Scores/ood_datasets"
export dataset_name="ood_3"
export score_folder="ood_3_f1_scores"
export model_name="bert"

python Random_Pruning_Results.py \
--dataset_folder $dataset_folder \
--dataset_name $dataset_name \
--score_folder $score_folder \
--model_name $model_name
