#!/bin/bash
export test_dataset_path="~/Influence_Scores/sexism_data"
echo "$test_dataset_path"
export model_path="hard_run"
echo "$model_path"
export model_name="roberta"
echo "$model_name"
export train_dataset_name="sexist_data_undersample"
echo "$train_dataset_name"
export test_dataset_name="sexist_data"
echo "$test_dataset_name"
export learning_rate="1e-6"
echo "$learning_rate"
export num_train_epochs="5"
echo "$num_train_epochs"
export x_column="text"
echo "$x_column"
export y_column="numeric_labels"
echo "$y_column"
export inf_scores=("pvi" "el2n" "vog")
for inf_score in ${inf_scores[@]}
do
    export train_dataset_path="cmsb_"$inf_score"_hard"
    for prune_rate in 5 10 15 20 25 30 35 40 50 60
    do
        for run_num in {1..5}
        do
            python Fine_Tune_Undersampling.py \
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
done