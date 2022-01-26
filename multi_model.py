#!/usr/bin/env python
# coding: utf-8

# # Multi

# In[1]:


from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import graphviz
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# In[4]:


df309 = pd.read_csv('real_309_4_en.csv')
df309.head(3)


# In[5]:


X = df309.drop(columns = ['label', 'years','law','support', 'mind'])
y = df309['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X = PCA(n_components=2).fit_transform(X)
X = StandardScaler().fit_transform(X)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)


# In[6]:


from xgboost import XGBClassifier
xgbc_model=XGBClassifier()

from sklearn.ensemble import RandomForestClassifier
rfc_model=RandomForestClassifier()

from sklearn.ensemble import ExtraTreesClassifier
et_model=ExtraTreesClassifier()

from sklearn.naive_bayes import GaussianNB
gnb_model=GaussianNB()

from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier()

from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression()

from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier()

from sklearn.svm import SVC
svc_model=SVC()

xgbc_model.fit(X,y)

rfc_model.fit(X,y)

et_model.fit(X,y)

gnb_model.fit(X,y)

knn_model.fit(X,y)

lr_model.fit(X,y)

dt_model.fit(X,y)

svc_model.fit(X,y)

print("XGBoost：",cross_val_score(xgbc_model,X,y,cv=5).mean())
print("\tRandomForest：",cross_val_score(rfc_model,X,y,cv=5).mean())
print("\tET：",cross_val_score(et_model,X,y,cv=5).mean())
print("\tNaiveBayes：",cross_val_score(gnb_model,X,y,cv=5).mean())
print("\tKNN：",cross_val_score(knn_model,X,y,cv=5).mean())
print("\tLogisticR：",cross_val_score(lr_model,X,y,cv=5).mean())
print("\tDecisionTree：",cross_val_score(dt_model,X,y,cv=5).mean())
print("\tSVM：",cross_val_score(svc_model,X,y,cv=5).mean())


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = XGBClassifier(
 eta =0.01,
 n_estimators=100,
 max_depth=7,
 min_child_weight=10,
 gamma=0.8,
 subsample=1,
 colsample_bytree=0.7,
 reg_lambda = 1,
 reg_alpha = 2,
 objective= 'reg:logistic',
 nthread=4,
 scale_pos_weight=1,
 )

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
num_correct_samples = accuracy_score(y_test, y_pred, normalize=False)
con_matrix = confusion_matrix(y_test, y_pred)

print('number of correct sample: ', num_correct_samples)
print('accuracy: ', accuracy)
print('confusion matrix: ', con_matrix)
print('Report: ', classification_report(y_test, y_pred))


# In[ ]:




