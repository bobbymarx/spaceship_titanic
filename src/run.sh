#!/bin/bash

# Define an array of models
models=('rf' 'decision_tree_gini' 'decision_tree_entropy' 'KNN' 'svc' 'xgb' 'Ada' 'LDA' 'Bernoulli' 'NN' )

# Loop over each model
for model in "${models[@]}"; do
    echo "Processing model: $model"
    # Loop through folds 0 to 7
    for fold in {0..7}; do
        python train.py --fold "$fold" --model "$model"
    done
done