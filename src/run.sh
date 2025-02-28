#!/bin/bash

# Define an array of models
#models=('rf' 'xgb' 'Ada' 'CatBoost')
models=('rf1' 'rf2' 'rf3' 'rf4' 'rf5' 'rf6' 'rf7' 'rf8')

# Loop over each model
for model in "${models[@]}"; do
    echo "Processing model: $model"
    # Loop through folds 0 to 7
    for fold in {0..7}; do
        python train.py --fold "$fold" --model "$model" --search_type "grid"
    done
done