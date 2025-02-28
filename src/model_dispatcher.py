#model dispatcher.py

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
#from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier
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
    #"CatBoost": CatBoostClassifier(iterations=5, learning_rate=0.1),
    "LDA": LinearDiscriminantAnalysis(),
    "Bernoulli": BernoulliRBM(),
    "NN":MLPClassifier(random_state=1, max_iter=300)

}