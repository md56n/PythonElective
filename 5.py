import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# machine learning
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

#reading data set
train_df = pd.read_csv('./train_preprocessed.csv')
test_df = pd.read_csv('./test_preprocessed.csv')

#count number of data in each category
print(train_df['Survived'].value_counts(dropna="False"))

#fix null values
print(train_df.isnull().sum())
#fill null value with most common
train_df["Embarked"]=train_df["Embarked"].fillna("S")
#drop categories with a lot of null values
train_df.drop("Cabin", asix=1, inplace=True)

#analyze p values (p>.05, correlated)
train_df[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)

#visualize with histogram
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
#drop cabin and ticket features
train_df=train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df=test_df.drop(['Ticket', 'Cabin'], axis=1)
combine=[train_df, test_df]

#convert categorical feature to numeric
for dataset in combine:
    dataset['Sex']=dataset['Set'].map({'female':1, 'male':0}).astype(int)

#read data
train_df=pd.read_csv('./train_preprocessed.csv')
test_df=pd.read_csv('./test_preprocessed.csv')
X_train=train_df.drop("Survived", axis=1)
Y_train=train_df["Survived"]
X_test=test_df.drop("PassengerId", axis=1).copy()
print(train_df[train_df.isnull().any(axis=1)])

#train model and evaluate SVM
svc=SVC()
svc.fit(X_train, Y_train)
Y_pred=svc.predict(X_test)
acc_svc=round(svc.score(X_test, Y_test)*100,2)
print("svm accuracy is:", acc_svc)

#train and evaluate KNN
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred=knn.predict(X_test)
acc_knn=round(knn.score(X_train, Y_train)*100,2)
print("KNN accuracy is:", acc_knn)


