#!/bin/bash
export train_dataset_path="easy_easy"
export test_dataset_path="~/influence_score_pruning_sexism/sexism_data"
export model_path="pvi_runs/easy_hard_run"
export model_name="roberta"
export train_dataset_name="sexist_data_proportional"
export test_dataset_name="sexist_data"
export inf_score="pvi"
export learning_rate="1e-6"
export num_train_epochs="5"
export x_column="text"
export y_column="numeric_labels"
for prune_rate in 5 10 15 20 25 30 35 40 50 60
do
    for run_num in {1..5}
    do
        python Fine_Tune_Proportional_Sampling.py \
        --train_dataset_path $train_dataset_path \
        --test_dataset_path $test_dataset_path \
        --model_path $model_path \
        --model_name $model_name \
        --train_dataset_name $train_dataset_name \
        --test_dataset_name $test_dataset_name \
        --inf_score $inf_score \
        --run_num $run_num \
        --prune_rate $prune_rate \
        --learning_rate $learning_rate \
        --num_train_epochs $num_train_epochs \
        --x_column $x_column \
        --y_column $y_column
    done
done

export train_dataset_path="easy_easy"
export model_path="el2n_runs/easy_easy_run"
export inf_score="el2n"
for prune_rate in 5 10 15 20 25 30 35 40 50 60
do
    for run_num in {1..5}
    do
        python Fine_Tune_Proportional_Sampling.py \
        --train_dataset_path $train_dataset_path \
        --test_dataset_path $test_dataset_path \
        --model_path $model_path \
        --model_name $model_name \
        --train_dataset_name $train_dataset_name \
        --test_dataset_name $test_dataset_name \
        --inf_score $inf_score \
        --run_num $run_num \
        --prune_rate $prune_rate \
        --learning_rate $learning_rate \
        --num_train_epochs $num_train_epochs \
        --x_column $x_column \
        --y_column $y_column
    done
done