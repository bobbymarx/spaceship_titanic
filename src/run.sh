#!/bin/bash

# Define an array of models
#models=('rf' 'xgb' 'Ada' 'CatBoost')
models=('rf1' 'rf2' 'rf3' 'rf4' 'rf5' 'rf6' 'rf7' 'rf8' 'Ada1' 'Ada2' 'Ada3' 'Ada4' 'Ada5' 'xgb1' 'xgb2' 'xgb3' 'xgb4' 
'CatBoost1' 'CatBoost2' 'CatBoost3' 'CatBoost4' 'CatBoost5' 'CatBoost6' 'CatBoost7' 'CatBoost8' 'CatBoost9')

# Loop over each model

for fold in {0..7}; do
    for model in "${models[@]}"; do
        echo "Processing model: $model"
        python train.py --fold "$fold" --model "$model"
    done
    echo " "
done