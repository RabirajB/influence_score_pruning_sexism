#!/bin/bash
export train_dataset_path="submod_data"
export test_dataset_path="sexism_data"
export model_path="submod_run"
export model_name="bert"
export train_dataset_name="sexist_data_submodular"
export test_dataset_name="sexist_data"
export learning_rate="1e-6"
export num_train_epochs="5"
export x_column="text"
export y_column="numeric_labels"

for run_num in {1..5}
do
    for prune_rate in 5 10 15 20 25 30 35 40 50 60
    do
        python Fine_Tune_Submodular.py \
        --train_dataset_path $train_dataset_path"_"$run_num \
        --test_dataset_path $test_dataset_path \
        --model_path $model_path \
        --model_name $model_name \
        --train_dataset_name $train_dataset_name \
        --test_dataset_name $test_dataset_name \
        --run_num $run_num \
        --prune_rate $prune_rate \
        --learning_rate $learning_rate \
        --num_train_epochs $num_train_epochs \
        --x_column $x_column \
        --y_column $y_column
    done
done