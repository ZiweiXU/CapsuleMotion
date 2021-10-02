#!/usr/bin/env bash

model_path=$1
op=$2

if [[ -z $model_path ]]; then exit 1; fi
if [[ -z $op ]]; then op="replay"; fi

src_filenames=('SepMixNewRe.py' 'train_rewrite_re.py')
subs_path=('model/SepMixNewRe.py' 'train_rewrite_re.py')

if [[ $op == "replay" ]]; then
    # backup HEAD file
    for p in $(seq 0 "$(calc ${#subs_path[@]}-1)"); do
        if test -f "${subs_path[p]}".bak; then
            echo "Another replay has not finished."
            exit 2
        fi
        
        echo "${subs_path[p]}" "->" "${subs_path[p]}".bak
        mv "${subs_path[p]}" "${subs_path[p]}".bak
        echo "${model_path}/${src_filenames[p]}" "->" "${subs_path[p]}"
        cp "${model_path}/${src_filenames[p]}" "${subs_path[p]}"
    done
    echo 'File replay finished. Raw command was:'
    cat "${model_path}/*.argv"
fi

if [[ $op == "restore" ]]; then
    # restore HEAD file
    for p in $(seq 0 "$(calc ${#subs_path[@]}-1)"); do
        echo "${subs_path[p]}".bak "->" "${subs_path[p]}"
        mv "${subs_path[p]}".bak "${subs_path[p]}"
    done
    echo 'File restored to HEAD.'
fi