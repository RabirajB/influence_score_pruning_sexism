#!/bin/bash

export non_sexist_prune_types=("hard" "easy")

for non_sexist_prune in ${non_sexist_prune_types[@]}
do
    mkdir $sexist_prune"_"$non_sexist_prune
done
