#!/bin/bash
export dataset_name="sexist_data"
export split="train"
export model_name="roberta"
export checkpoint="5"
export data_path="sexism_data"
for run_num in {1..5}
do
    export model_path="cmsb_run_$run_num"
    echo "$model_path"
    export results_path="cmsb_run_$run_num"
    echo "$results_path"
    python P-Vinfo.py \
    --dataset_name $dataset_name \
    --split $split \
    --model_name $model_name \
    --checkpoint $checkpoint \
    --model_path $model_path \
    --data_path $data_path \
    --run_num $run_num \
    --results_path $results_path
done