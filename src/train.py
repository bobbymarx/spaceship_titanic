import config
import os 
import argparse
import model_dispatcher
import pandas as pd 
from preprocessing import processing
from sklearn.metrics import confusion_matrix
from sklearn import metrics
pd.set_option('future.no_silent_downcasting', True)

def run(fold, model, test):
    
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

    clf=model_dispatcher.models[model]

    clf.fit(x_train,y_train)

    #use different proba for rf to improve precision

    preds=clf.predict(x_valid)

    if test==False:

        accuracy=metrics.accuracy_score(y_valid,preds)

        print(f"Fold={fold},Accuracy={accuracy}")
        #print(confusion_matrix(y_valid,preds))

        counter=0
        for runner in range(len(y_valid)):
            if y_valid[runner]==1:
                if preds[runner]==0:
                    counter+=1
        
        #print(counter)

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
                        help="The model that should be used for the task")
    
    parser.add_argument("--test",
                        type=bool,
                        default=False,
                        help="True: if predictions should be made for the test set, else false")
    
    #read arguments from the parser
    args= parser.parse_args()
    run(fold=args.fold, model=args.model, test=args.test)