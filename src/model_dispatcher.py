#model dispatcher.py

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
#from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier



models= {
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    "KNN":KNeighborsClassifier(n_neighbors=20),
    "rf": RandomForestClassifier(class_weight='balanced', random_state=42),
    "svc": SVC(C=1),
    "xgb": GradientBoostingClassifier(random_state=42), 
    "Ada": AdaBoostClassifier(n_estimators=100, random_state=0),
    #"LightGBM": LGBMClassifier(random_state=5),
    "CatBoost": CatBoostClassifier(iterations=5, learning_rate=0.1, logging_level='Silent'),
    "LDA": LinearDiscriminantAnalysis(),
    "Bernoulli": BernoulliRBM(),
    "NN":MLPClassifier(random_state=1, max_iter=300),

    "rf1":RandomForestClassifier(class_weight='balanced', random_state=42, 
                                 max_depth=10,
                                 max_features='sqrt',
                                 min_samples_leaf=4,
                                 min_samples_split=10,
                                 n_estimators=200
                                 ),

    "rf2":RandomForestClassifier(class_weight='balanced', random_state=42, 
                                 max_depth=10,
                                 max_features='sqrt',
                                 min_samples_leaf=1,
                                 min_samples_split=2,
                                 n_estimators=300
                                 ),
    
    "rf3":RandomForestClassifier(class_weight='balanced', random_state=42, 
                                 max_depth=10,
                                 max_features='log2',
                                 min_samples_leaf=1,
                                 min_samples_split=2,
                                 n_estimators=300
                                 ),

    "rf4":RandomForestClassifier(class_weight='balanced', random_state=42, 
                                 max_depth=20,
                                 max_features='sqrt',
                                 min_samples_leaf=4,
                                 min_samples_split=2,
                                 n_estimators=200
                                 ),
    
    "rf5":RandomForestClassifier(class_weight='balanced', random_state=42, 
                                 max_depth=20,
                                 max_features='log2',
                                 min_samples_leaf=4,
                                 min_samples_split=2,
                                 n_estimators=100
                                 ),

    "rf6":RandomForestClassifier(class_weight='balanced', random_state=42, 
                                 max_depth=30,
                                 max_features='sqrt',
                                 min_samples_leaf=2,
                                 min_samples_split=10,
                                 n_estimators=100
                                 ),
    
    "rf7":RandomForestClassifier(class_weight='balanced', random_state=42, 
                                 max_depth=20,
                                 max_features='sqrt',
                                 min_samples_leaf=2,
                                 min_samples_split=2,
                                 n_estimators=200
                                 ),

    "rf8":RandomForestClassifier(class_weight='balanced', random_state=42, 
                                 max_depth=20,
                                 max_features='sqrt',
                                 min_samples_leaf=4,
                                 min_samples_split=10,
                                 n_estimators=100
                                 ),



}