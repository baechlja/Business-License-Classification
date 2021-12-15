import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.inspection import permutation_importance


# load CSV into script
data = pd.read_csv("License_Data.csv", sep=",")


data = data.set_index("ID")
data = data.drop(["LICENSE ID", "ACCOUNT NUMBER", "LEGAL NAME", "DOING BUSINESS AS NAME",
                  "ADDRESS", "ZIP CODE", "LOCATION"], axis=1)
data = data.drop(columns=data.columns[((data.isna().sum()/len(data)) > 0.60)])

y = data["LICENSE STATUS"]
X = data.drop(["LICENSE STATUS"], axis=1)

le = LabelEncoder()
X = X.apply(le.fit_transform)

ss = StandardScaler()
X = pd.DataFrame(ss.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# tree = DecisionTreeClassifier()
# tree.fit(X_train, y_train)
# cv_score = cross_val_score(tree, X_train, y_train).mean().round(4)
# print(cv_score)
# y_train_pred = cross_val_predict(tree, X_train, y_train)
# c_matrix = confusion_matrix(y_train, y_train_pred)
# plt.matshow(c_matrix, cmap=plt.cm.gray)
# plt.show()
# feature_importance(tree.feature_importances_ )
# feature_importance(tree, X_test, y_test)


# svc = LinearSVC(max_iter=10000)
# svc.fit(X_train, y_train)
# cv_score = cross_val_score(svc, X_train, y_train).mean().round(4)
# print(cv_score)
# y_train_pred = cross_val_predict(tree, X_train, y_train)
# c_matrix = confusion_matrix(y_train, y_train_pred)
# plt.matshow(c_matrix, cmap=plt.cm.gray)
# plt.show()
# feature_importance(svc, X_test, y_test)


# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# cv_score = cross_val_score(knn, X_train, y_train).mean().round(4)
# print(cv_score)
# feature_importance(knn, X_test, y_test)


def feature_importance(model, x, y):
    perm_importance = permutation_importance(model, x, y)

    feature_names = x.columns
    features = np.array(feature_names)

    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.show()


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