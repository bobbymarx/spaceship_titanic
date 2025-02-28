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
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression



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

    "Ada1": AdaBoostClassifier(n_estimators=200, random_state=0, algorithm='SAMME.R',
                               learning_rate=0.1),
    
    "Ada2": AdaBoostClassifier(n_estimators=100, random_state=0, algorithm='SAMME.R',
                               learning_rate=1.0),

    "Ada3": AdaBoostClassifier(n_estimators=50, random_state=0, algorithm='SAMME.R',
                               learning_rate=1.0),
    
    "Ada4": AdaBoostClassifier(n_estimators=200, random_state=0, algorithm='SAMME',
                               learning_rate=1.0),
    
    "Ada5": AdaBoostClassifier(n_estimators=200, random_state=0, algorithm='SAMME.R',
                               learning_rate=1.0),

    "xgb1": GradientBoostingClassifier(random_state=42, ccp_alpha=0, learning_rate=0.1,
                                       max_depth=5, 
                                       n_estimators=100), 
    
    "xgb2": GradientBoostingClassifier(random_state=42, ccp_alpha=0, learning_rate=0.1,
                                       max_depth=3, 
                                       n_estimators=300), 
    
    "xgb3": GradientBoostingClassifier(random_state=42, ccp_alpha=0, learning_rate=0.01,
                                       max_depth=5, 
                                       n_estimators=300), 

    "xgb4": GradientBoostingClassifier(random_state=42, ccp_alpha=0, learning_rate=0.3,
                                       max_depth=3, 
                                       n_estimators=200), 

    "CatBoost1": CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=128,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=3),
    
    "CatBoost2": CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=32,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=3),
    
    "CatBoost3": CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=64,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=3),
    
    "CatBoost4": CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=128,
                                    depth=6,
                                    iterations=500,
                                    l2_leaf_reg=5),
    
    "CatBoost5": CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=32,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=10),
    
    "CatBoost6": CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=64,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=5),
    
    "CatBoost7": CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=64,
                                    depth=6,
                                    iterations=500,
                                    l2_leaf_reg=3),
    
    "CatBoost8": CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=64,
                                    depth=6,
                                    iterations=500,
                                    l2_leaf_reg=1),
                                     
    "CatBoost9": CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=128,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=10),

    #combining the best performing models

    "ensemble_model_soft": VotingClassifier(
    estimators=[
        ('catboost', CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=128,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=10)),
        ('xgb', GradientBoostingClassifier(random_state=42, ccp_alpha=0, learning_rate=0.1,
                                       max_depth=3, 
                                       n_estimators=300)),
        ('rf', RandomForestClassifier(class_weight='balanced', random_state=42, 
                                 max_depth=20,
                                 max_features='sqrt',
                                 min_samples_leaf=4,
                                 min_samples_split=2,
                                 n_estimators=200
                                 ))
    ],
    voting='soft'  # using soft voting to average predicted probabilities
        ),
"ensemble_model_hard": VotingClassifier(
    estimators=[
        ('catboost', CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=128,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=10)),
        ('xgb', GradientBoostingClassifier(random_state=42, ccp_alpha=0, learning_rate=0.1,
                                       max_depth=3, 
                                       n_estimators=300)),
        ('rf', RandomForestClassifier(class_weight='balanced', random_state=42, 
                                 max_depth=20,
                                 max_features='sqrt',
                                 min_samples_leaf=4,
                                 min_samples_split=2,
                                 n_estimators=200
                                 ))
    ],
    voting='hard'  # using soft voting to average predicted probabilities
),

"ensemble_catboost_soft": VotingClassifier(
    estimators=[
        ('cb1', CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=128,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=10)),
        ('cb2', CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=64,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=5)),
        ('cb3', CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=128,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=3))
    ],
    voting='soft'  # soft voting to combine probabilities
),
"ensemble_catboost_hard": VotingClassifier(
    estimators=[
        ('cb1', CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=128,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=10)),
        ('cb2', CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=64,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=5)),
        ('cb3', CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=128,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=3))
    ],
    voting='hard'  # soft voting to combine probabilities
),

"Stack_ensemble_model": StackingClassifier(
    estimators=[
        ('catboost', CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=128,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=10)),
        ('xgb', GradientBoostingClassifier(random_state=42, ccp_alpha=0, learning_rate=0.1,
                                       max_depth=3, 
                                       n_estimators=300)),
        ('rf', RandomForestClassifier(class_weight='balanced', random_state=42, 
                                 max_depth=20,
                                 max_features='sqrt',
                                 min_samples_leaf=4,
                                 min_samples_split=2,
                                 n_estimators=200
                                 ))
    ],
    final_estimator=LogisticRegression(),
    cv=5 
),

"Stack_ensemble_catboost": StackingClassifier(
    estimators=[
        ('cb1', CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=128,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=10)),
        ('cb2', CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=64,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=5)),
        ('cb3', CatBoostClassifier(learning_rate=0.05, logging_level='Silent', random_seed=42, bagging_temperature=0,
                                    border_count=128,
                                    depth=4,
                                    iterations=500,
                                    l2_leaf_reg=3))
    ],
    final_estimator=LogisticRegression(),
    cv=5 
)

}