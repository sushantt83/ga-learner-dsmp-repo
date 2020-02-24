# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# store the dataframe
df = pd.read_csv(path)

# store independent variable
X = df.drop('insuranceclaim',axis=1)

# store dependent variable
y = df['insuranceclaim']

# spliting the dataset
X_train,X_test,y_train,y_test=train_test_split(X,y ,test_size=0.2,random_state=6)
# code ends here


# --------------
import matplotlib.pyplot as plt


# Code starts here

fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(X_train['bmi'])
q_value=X_train['bmi'].quantile(.95) 
y_train.value_counts(normalize=True)
print(q_value)
# Code ends here


# --------------
# Code starts here:

# check the correlation of X_train
relation = X_train.corr()
print(relation)

# create heatmap using seaborn
sns.pairplot(X_train)


# --------------
import seaborn as sns
import matplotlib.pyplot as plt

# code starts here

# store categorical variable
cols = ['children','sex','region','smoker']
        


# create subplot 

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,20))

# create loop for plotting countplot

for i in range(0,2):
    print(i)
    for j in range(0,2):
        print(j)
        col=cols[i*2 + j]
        print(X_train[col])
        sns.countplot(x=X_train[col], hue=y_train, ax=axes[i,j])

# Code ends here




# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here
lr=LogisticRegression(random_state=9)
grid=GridSearchCV(estimator=lr, param_grid=parameters )
grid.fit(X_train, y_train)
y_pred=grid.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(accuracy)


# Code ends here


# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Code starts here
score=  roc_auc_score(y_pred, y_test)
y_pred_proba=grid.predict_log_proba(X_test)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_pred, y_test)
roc_auc=roc_auc_score(y_test, y_pred_proba)
print("fpr", fpr)
print("tpr", tpr)
print("roc",roc_auc)
plt.plot(fpr,tpr,label="Logistic model, auc="+str(roc_auc))
# Code ends here


