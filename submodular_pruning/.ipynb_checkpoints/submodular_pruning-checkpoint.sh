#!/bin/bash
export embedding_path='~/<directory_name>'
export dataset_path='~/<dataset_path>'
export dataset_name='<dataset_name>'


for run_num in {1..5}
do
    python Submodular_Pruning.py \
    --embedding_path $embedding_path"_"$run_num \
    --dataset_path $dataset_path \
    --dataset_name $dataset_name \
    --run_num $run_num
done
