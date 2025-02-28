# Spaceship Titanic Prediction

## Project Overview
This project tackles the Kaggle competition "Spaceship Titanic", achieving a public accuracy score of 80.6% using an ensemble stacking approach with CatBoost, XGBoost, and Random Forest models.

## Problem Description
The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars. While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary vessel collided with a spacetime anomaly hidden within a dust cloud. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!

## Project Structure

```bash
spaceship_titanic/
├── input/ # Data directory
│ ├── train.csv # Training dataset
│ ├── test.csv # Test dataset
│ └── train_folds.csv # Training data with CV fold assignments
├── src/ # Source code
│ ├── config.py # Configuration settings (paths, etc.)
│ ├── create_folds.py # Script for creating cross-validation folds
│ ├── model_dispatcher.py # Model configurations and ensemble setups
│ ├── preprocessing.py # Data preprocessing pipeline classes
│ ├── train.py # Main training script
│ └── run.sh # Bash script for running experiments
├── notebooks/ # Jupyter notebooks
│ └── eda.ipynb # Exploratory Data Analysis
├── predictions/ # Model predictions
│ └── predictions.csv # Final predictions on test set
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```


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
with Logistic Regression as the meta-learner achieving an Accuracy of 80.6% on the public test set.


### Key Components Explained

#### Source Code (`src/`)
- **config.py**: Contains configuration settings like file paths and model parameters
- **create_folds.py**: Implements stratified k-fold cross-validation
- **model_dispatcher.py**: Defines model architectures and ensemble configurations
- **preprocessing.py**: Contains custom transformer classes for data preprocessing:
  - HomePlanetImputer
  - CabinImputer
  - CryoSleepImputer
  - SpendingImputer
  - FeatureEngineer
  - And more...
- **train.py**: Main script for model training and evaluation
- **run.sh**: Automation script for running multiple experiments

#### Data (`input/`)
- **train.csv**: Original training data
- **test.csv**: Test data for final predictions
- **train_folds.csv**: Training data split into cross-validation folds

#### Notebooks (`notebooks/`)
- **eda.ipynb**: Exploratory Data Analysis notebook with visualizations and insights

#### Outputs (`predictions/`)
- Contains model predictions and submission filess