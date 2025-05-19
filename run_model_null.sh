#!/bin/bash
export path_to_dataset="sexism_data"
echo "$path_to_dataset"
export model_path="cmsb_run"
echo "$model_path"
export model_name="roberta"
echo "$model_name"
export train_dataset_name="sexist_data"
echo "$train_dataset_name"
export test_dataset_name="sexist_data"
echo "$test_dataset_name"
export learning_rate="1e-6"
echo "$learning_rate"
export num_train_epochs="5"
echo "$num_train_epochs"
export x_column="text"
echo "$x_column"
export x_column_null="empty_strings"
echo "$x_column_null"
export y_column="numeric_labels"
echo "$y_column"
export type_data="null"
for run_num in {1..5}
    do
        python Fine_Tune_Model.py \
        --path_to_dataset $path_to_dataset \
        --train_dataset_name $train_dataset_name \
        --test_dataset_name $test_dataset_name \
        --model_path $model_path \
        --model_name $model_name \
        --train_dataset_name $train_dataset_name \
        --test_dataset_name $test_dataset_name \
        --run_num $run_num \
        --learning_rate $learning_rate \
        --num_train_epochs $num_train_epochs \
        --type_data $type_data \
        --x_column $x_column \
        --x_column_null $x_column_null \
        --y_column $y_column
    done