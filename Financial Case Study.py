import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline

# load data
df = pd.read_csv('loan_train.csv')
df.head()

# convert to datetime 
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()

df['loan_status'].value_counts()

# visual
import seaborn as sns
bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

# visual 
bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# visual 
df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

# data transform 
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()

df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)

df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()

Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1) # one hot encode 
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

# Data set split
X = Feature
X = X.drop(columns = ['weekend']) # removing unneeded features 

y = df['loan_status'].values
y_binary = df.loan_status.apply(lambda x: 1 if x == 'PAIDOFF' else 0) # converting target field to binary classifier 


# Normalization
X_nor = preprocessing.StandardScaler().fit(X).transform(X)

# KNN Model 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_nor, y_binary, test_size=0.20, random_state=4)

Ks = 30
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
#Train Model and Evaluate  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train_knn,y_train_knn)
    yhat_knn =neigh.predict(X_test_knn)
    mean_acc[n-1] = metrics.accuracy_score(y_test_knn, yhat_knn)
    std_acc[n-1]=np.std(yhat_knn==y_test_knn)/np.sqrt(yhat_knn.shape[0])

print(mean_acc)
 
# Fit Best Model
neigh = KNeighborsClassifier(n_neighbors = 5).fit(X_train_knn,y_train_knn)

# Decision Tree

import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
import matplotlib.image as mpimg
from sklearn import tree
%matplotlib inline 
from sklearn import metrics
import matplotlib.pyplot as plt

X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_nor, y_binary, test_size=0.2, random_state=4)

dep = 10
mean_acc = np.zeros((dep-1))
for n in range(1,dep):
    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = n)
    drugTree.fit(X_train_dt,y_train_dt)
    predTree = drugTree.predict(X_test_dt)
    mean_acc[n-1] = metrics.accuracy_score(y_test_dt, predTree)

print(mean_acc)

# Determine new tree
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 2)
drugTree.fit(X_train_dt,y_train_dt)


# SVM Model
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
%matplotlib inline 
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools


# train test split 
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_nor, y_binary, test_size=0.2, random_state=4)

degree_ = 5
mean_acc = np.zeros((degree_-1))

for n in range(1,degree_):
    clf = svm.SVC(kernel = 'linear', degree = n)
    clf.fit(X_train_svm, y_train_svm) 
    yhat_svm = clf.predict(X_test_svm)
    mean_acc[n-1] = metrics.accuracy_score(y_test_svm, yhat_svm)

print(mean_acc)

# Best SVM
clf = svm.SVC(kernel = 'linear', degree = 1)
clf.fit(X_train_svm, y_train_svm) 

# Log Regression
from sklearn.externals.six import StringIO
import matplotlib.image as mpimg
from sklearn import tree
%matplotlib inline 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# training 
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_nor, y_binary, test_size=0.2, random_state=4)

LR = LogisticRegression(solver='newton-cg').fit(X_train_reg,y_train_reg)

# predict
yhat_reg = LR.predict(X_test_reg)
yhat_prob_reg = LR.predict_proba(X_test_reg)

###################################### Model evaluation ###################################### 
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

test_df = pd.read_csv('loan_test.csv')
df = test_df

# Convert to datetime # 
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])

# Day of Week #  
df['dayofweek'] = df['effective_date'].dt.dayofweek
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

# Convert gender field to binary field # 
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

# One Hot Encode categorical features #
Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)

# Further pre processing # 
X = Feature
X = X.drop(columns = ['weekend'])

# Target Field # 
y = df['loan_status'].values
y_binary = df.loan_status.apply(lambda x: 1 if x == 'PAIDOFF' else 0) 

# Normalization # 
X_nor = preprocessing.StandardScaler().fit(X).transform(X)

# Model Tuning
yhat_knn2 = neigh.predict(X_nor) # KNN
predTree2 = drugTree.predict(X_nor) # Decision Tree
yhat_svm2 = clf.predict(X_nor) # SVM 
yhat_reg2 = LR.predict(X_nor) # Logistic Regression 
yhat_prob_reg2 = LR.predict_proba(X_nor) # Logistic Regression

### Model Scores 

# KNN
jaccard_knn = jaccard_similarity_score(y_binary, yhat_knn2)
f1_score_knn = f1_score(y_binary, yhat_knn2) 

# Decision Tree
jaccard_dt = jaccard_similarity_score(y_binary, predTree2)
f1_score_dt = f1_score(y_binary, predTree2) 

# SVM
jaccard_svm = jaccard_similarity_score(y_binary, yhat_svm2)
f1_score_svm = f1_score(y_binary, yhat_svm2) 

# Log Reg
jaccard_reg = jaccard_similarity_score(y_binary, yhat_reg2)
f1_score_reg = f1_score(y_binary, yhat_reg2) 
log_loss_reg = log_loss(y_binary, yhat_prob_reg2)

table = [
            {'Algorithm':'KNN', 'Jaccard':jaccard_knn, 'F1-Score': f1_score_knn, 'LogLoss': 'NA'},
            {'Algorithm':'Decision Tree', 'Jaccard':jaccard_dt, 'F1-Score': f1_score_dt, 'LogLoss': 'NA'},
            {'Algorithm':'SVM', 'Jaccard':jaccard_svm, 'F1-Score': f1_score_svm, 'LogLoss': 'NA'},
            {'Algorithm':'LogisticRegression', 'Jaccard':jaccard_reg, 'F1-Score': f1_score_reg, 'LogLoss': log_loss_reg}
         ] 

table_df = pd.DataFrame.from_dict(table)
table_df

         