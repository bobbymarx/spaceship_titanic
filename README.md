# Spaceship Titanic Prediction

## Project Overview
This project tackles the Kaggle competition "Spaceship Titanic", achieving a public accuracy score of 80.6% using an ensemble stacking approach with CatBoost, XGBoost, and Random Forest models.

## Problem Description
The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars. While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary vessel collided with a spacetime anomaly hidden within a dust cloud. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!

## Solution Approach

### Data Preprocessing
- **Missing Value Imputation**:
  - HomePlanet: Imputed based on Last Name and Cabin patterns
  - Cabin: Imputed using Last Name mapping and spending patterns
  - CryoSleep: Inferred from spending behavior
  - VIP Status: Predicted using KNN based on cabin and spending features
  - Destination: Filled using family patterns and HomePlanet probabilities

### Feature Engineering
- Created family-related features based on Last Name and Cabin
- Generated gender predictions from passenger names
- Engineered spending-related features
- Split Cabin information into deck, number, and side

### Model Pipeline
1. **Data Preprocessing**: Comprehensive pipeline handling missing values and feature engineering
2. **Feature Encoding**: Mix of One-Hot Encoding for categorical variables and Standard Scaling for numeric features
3. **Model Selection**: Implemented Grid and Random Search for hyperparameter tuning

### Best Performing Model
Stacking Ensemble combining:
- CatBoost Classifier
- XGBoost Classifier
- Random Forest Classifier
with Logistic Regression as the meta-learner

## Project Structure
spaceship-titanic/
├── input/ # Folder containing dataset files
│ ├── train.csv # Training data
│ ├── test.csv # Test data
├── notebooks/ # Jupyter notebooks for EDA and modeling
│ ├── eda.ipynb # Exploratory Data Analysis
├── src/ # Source code for preprocessing and modeling
│ ├── config.py # Config Script
│ ├── train.py # Model training script
  ├── model_dispatcher.py # Model overviews
  ├── run.sh # bash script for running model experiments
│ └── create_folds.py # Script for generating CV
├── requirements.txt # Python dependencies
├── README.md # This file
└── .gitignore # Files to ignore in Git

## Approach
### Data Preprocessing:

Handled missing values by imputing with mean/mode or using advanced techniques like KNN imputation.

Encoded categorical variables using one-hot encoding or label encoding.

Scaled numerical features for better model performance.

### Feature Engineering:

Extracted useful information from the Cabin column (e.g., deck, side).

Created new features like FamilySize and IsAlone based on passenger groups.

### Modeling:

Experimented with various machine learning models, including Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.

Used cross-validation and hyperparameter tuning to optimize model performance.

### Evaluation:

Evaluated models using accuracy, precision, recall, and F1-score.

Generated a submission file for the Kaggle competition.

## Results