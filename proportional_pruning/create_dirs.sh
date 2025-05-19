#!/bin/bash
export sexist_prune_types=("hard" "easy")
export non_sexist_prune_types=("hard" "easy")

for sexist_prune in ${sexist_prune_types[@]}
do
    for non_sexist_prune in ${non_sexist_prune_types[@]}
    do
        mkdir $sexist_prune"_"$non_sexist_prune
    done
done
