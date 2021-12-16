import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance


# load CSV into script
data = pd.read_csv("License_Data.csv", sep=",")


def compare_algo():
    algs = [('tree', DecisionTreeClassifier()),
            ('svc', LinearSVC(max_iter=10000)),
            ('knn', KNeighborsClassifier())]

    results = []
    names = []

    for name, model in algs:
        cv_results = cross_val_score(model, X_train, y_train)
        results.append(cv_results)
        names.append(name)

    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.xlabel("Alogrithm")
    plt.ylabel("cross-validation-score")
    plt.show()


def feature_importance(model, x, y, s=None):
    perm_importance = permutation_importance(model, x, y, scoring=s)

    feature_names = x.columns
    features = np.array(feature_names)

    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.show()


def runtime(model, x):
    start = time.time()
    model.predict(x)
    end = time.time()
    t = end - start
    print(f"{model} runtime: {t}")


data = data.set_index("ID")
data = data.drop(["LICENSE ID", "ACCOUNT NUMBER", "LEGAL NAME", "DOING BUSINESS AS NAME",
                  "ADDRESS", "LOCATION", "ZIP CODE", "LICENSE DESCRIPTION"], axis=1)
data = data.drop(columns=data.columns[((data.isna().sum()/len(data)) > 0.60)])
y = data["LICENSE STATUS"]

le = LabelEncoder()
data = data.apply(le.fit_transform)

fs = StandardScaler()
data = pd.DataFrame(fs.fit_transform(data), columns=data.columns)

X = data.drop(["LICENSE STATUS"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

svc = LinearSVC()
svc.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


# tree_score = cross_val_score(tree, X_train, y_train).mean().round(4)
# svc_score = cross_val_score(svc, X_train, y_train).mean().round(4)
# knn_score = cross_val_score(knn, X_train, y_train).mean().round(4)
#
# print("Cross Validation Scores:")
# print(f"\nDecision Tree Classifier: {tree_score}")
# print(f"Linear SVC: {svc_score}")
# print(f"KNeighborsClassifier {knn_score}")


# tree_pred = tree.predict(X_test)
# confusion_matrix(y_test, tree_pred)
#
# svc_pred = svc.predict(X_test)
# confusion_matrix(y_test, svc_pred)
#
# knn_pred = knn.predict(X_test)
# confusion_matrix(y_test, knn_pred)


# compare_algo()


# feature_importance(tree, X_test, y_test)
# feature_importance(svc, X_test, y_test)
# feature_importance(knn, X_test, y_test, 'neg_mean_squared_error')


# runtime(tree, X_test)
# runtime(svc, X_test)
# runtime(knn, X_test)