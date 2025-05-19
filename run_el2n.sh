#!/bin/bash
export dataset_name="sexist_data"
export split="train"
export model_name="roberta"
export checkpoint="5"
export input_col="text"
for run_num in {1..5}
do
    export model_path="cmsb_run_$run_num"
    echo "$model_path"
    export data_path="cmsb_run_$run_num"
    python EL2N.py \
    --dataset_name $dataset_name \
    --split $split \
    --input_col $input_col \
    --model_name $model_name \
    --checkpoint $checkpoint \
    --model_path $model_path \
    --data_path $data_path \
    --run_num $run_num
done