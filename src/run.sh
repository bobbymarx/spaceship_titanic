#!/bin/bash

# Define an array of models
#models=('rf' 'xgb' 'Ada' 'CatBoost')
models=('CatBoost9' 'ensemble_model_soft' 'ensemble_model_hard' 'ensemble_catboost_soft' 'ensemble_catboost_hard' 'Stack_ensemble_model' 'Stack_ensemble_catboost')

# Loop over each model

for fold in {0..7}; do
    for model in "${models[@]}"; do
        echo "Processing model: $model"
        python train.py --fold "$fold" --model "$model"
    done
    echo " "
done