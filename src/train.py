import config
import os 
import argparse
import model_dispatcher
import pandas as pd 
from preprocessing import processing
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import metrics
import numpy as np
pd.set_option('future.no_silent_downcasting', True)

def run(fold, model, test, search_type="random"):
    
    if test==True:
        train_df=pd.read_csv(config.Training_File)
        #train_df=df.drop("kfold", axis=1)
        valid_df=pd.read_csv(config.Test_File)
        x_valid=valid_df

    else:
        df=pd.read_csv(config.Training_File_Folds)
        train_df=df[df.kfold!= fold].reset_index(drop=True)
        valid_df=df[df.kfold==fold].reset_index(drop=True)
        y_valid=valid_df.Transported.values
        x_valid=valid_df.drop(["Transported"],axis=1)
        

    x_train=train_df.drop(["Transported"],axis=1)
    y_train=train_df.Transported.values
    

    x_train, x_valid= processing(x_train, x_valid)

    # Define parameter grids for each model
    param_grids = {
        "rf": {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        "Ada": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
            'algorithm': ['SAMME', 'SAMME.R']
        },
        "xgb": {
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.3, 0.001],
            'n_estimators': [100, 200, 300],
            'ccp_alpha': [0, 0.1, 0.2]
        },
        "CatBoost": {
            'iterations': [100, 500],
            'learning_rate': [0.05],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 10],
            'bagging_temperature': [0],
            'border_count': [32, 64, 128]
        }
    }

    

    # Get base model
    base_model = model_dispatcher.models[model]
    
    if not test and model in param_grids and search_type in ["random", "grid"]:
        # Configure the search
        if search_type == "random":
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grids[model],
                n_iter=40,  # Number of parameter settings sampled
                cv=3,       # Number of folds
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
        else:  # grid search
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grids[model],
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
        
        # Fit the search
        search.fit(x_train, y_train)
        
        # Print the best parameters and score
        print(f"Best parameters: {search.best_params_}")
        print(f"Best cross-validation score: {search.best_score_:.4f}")
        
        # Use the best estimator for predictions
        clf = search.best_estimator_
    else:
        # Use the default model if no parameter search is requested
        clf = base_model
        clf.fit(x_train, y_train)

    #use different proba for rf to improve precision

    preds=clf.predict(x_valid)

    if test==False:

        accuracy=metrics.accuracy_score(y_valid,preds)

        print(f"Fold={fold},Accuracy={accuracy}")
        #print(confusion_matrix(y_valid,preds))

        #counter=0
        #for runner in range(len(y_valid)):
        #    if y_valid[runner]==1:
        #        if preds[runner]==0:
        #            counter+=1
        
        #print(f"False negatives: {counter}")

        #joblib.dump(clf,
        #            os.path.join(config.Model_Output, f"{model}_{fold}.bin"))
    
    else:
        output=pd.DataFrame({
            'PassengerId': valid_df['PassengerId'],
            'Survived': preds
         })
        
        output.to_csv(os.path.join(config.Output_File, f"predictions.csv"), index=False)
        print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    
    parser= argparse.ArgumentParser()

    parser.add_argument("--fold",
                        type=int,
                        default=0,
                        help="The number of the fold to be used for testing")
    
    parser.add_argument("--model", 
                        type=str,
                        default="svc",
                        help="The model that should be used for the task (rf, ada, or xgb)")
    
    parser.add_argument("--test",
                        type=bool,
                        default=False,
                        help="True: if predictions should be made for the test set, else false")
    
    parser.add_argument(
        "--search_type",
        type=str,
        choices=["random", "grid", "none"],
        default="none",
        help="Type of hyperparameter search to perform"
    )
    
    #read arguments from the parser
    args= parser.parse_args()
    run(fold=args.fold, model=args.model, test=args.test, search_type=args.search_type)