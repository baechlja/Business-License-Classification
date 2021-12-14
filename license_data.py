import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.inspection import permutation_importance


# load CSV into script
data = pd.read_csv("License_Data.csv", sep=",")


def data_overview(df):
    # count null values per column
    print(df.isnull().sum(), sep='\n')
    # print column type
    print(df.dtypes)

#data_overview(data)


# remove wrong zip codes
# data["ZIP CODE"] = data["ZIP CODE"].replace(r"[a-zA-Z]", np.nan, regex=True)
# specify dytpe of zip code column
# data["ZIP CODE"] = pd.to_numeric(data["ZIP CODE"])


def feature_importance(model_imp):
    imp = model_imp
    features = X_train.columns
    indices = np.argsort(imp)
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), imp[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.show()


def f_importances(model, x, y):
    perm_importance = permutation_importance(model, x, y)

    feature_names = x.columns
    features = np.array(feature_names)

    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.show()


df_prep = data.copy()
df_prep = df_prep.drop(["LICENSE ID", "ACCOUNT NUMBER", "LEGAL NAME", "DOING BUSINESS AS NAME",
                        "ADDRESS", "CITY", "STATE", "ZIP CODE"], axis=1)
df_prep = df_prep.set_index("ID")
y = df_prep["LICENSE STATUS"]

le = LabelEncoder()
df_prep = df_prep.apply(le.fit_transform)

ss = StandardScaler()
df_prep = pd.DataFrame(ss.fit_transform(df_prep), columns=df_prep.columns)

X = df_prep.drop(["LICENSE STATUS"], axis=1)
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


# svc = LinearSVC(max_iter=10000)
# svc.fit(X_train, y_train)
# cv_score = cross_val_score(svc, X_train, y_train).mean().round(4)
# print(cv_score)
# y_train_pred = cross_val_predict(tree, X_train, y_train)
# c_matrix = confusion_matrix(y_train, y_train_pred)
# plt.matshow(c_matrix, cmap=plt.cm.gray)
# plt.show()
# f_importances(svc, X_test, y_test)


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
# cv_score = cross_val_score(knn, X_train, y_train).mean().round(4)
# print(cv_score)
f_importances(knn, X_test, y_test)