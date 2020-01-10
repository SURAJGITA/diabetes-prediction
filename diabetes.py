#!/usr/bin/env python
# coding: utf-8

# # diabetes prediction

# In[1]:


import pandas as pd
df=pd.read_csv("data.csv")


# In[2]:


df.shape


# In[3]:


df.columns


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.head()


# In[7]:



# check if any null value is present
df.isnull().sum()


# In[8]:


df.corr()


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[10]:


diabetes_map = {True: 1, False: 0}


# In[11]:


df['diabetes'] = df['diabetes'].map(diabetes_map)


# In[12]:


df.head()


# In[13]:


print("number of rows missing skin: {0}".format(len(df.loc[df['skin'] == 0])))


# In[14]:


print("number of rows missing thickness: {0}".format(len(df.loc[df['thickness'] == 0])))


# In[15]:


print("number of rows missing glucose_conc: {0}".format(len(df.loc[df['glucose_conc'] == 0])))


# In[16]:


print("number of rows missing age: {0}".format(len(df.loc[df['age'] == 0])))


# In[17]:


df.drop("skin",axis=1,inplace=True)


# In[18]:


df.head()


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[20]:


print("total number of rows : {0}".format(len(df)))
print("number of rows missing glucose_conc: {0}".format(len(df.loc[df['glucose_conc'] == 0])))
print("number of rows missing diastolic_bp: {0}".format(len(df.loc[df['diastolic_bp'] == 0])))
#print("number of rows missing insulin: {0}".format(len(df.loc[df['insulin'] == 0])))
print("number of rows missing bmi: {0}".format(len(df.loc[df['bmi'] == 0])))
print("number of rows missing diab_pred: {0}".format(len(df.loc[df['diab_pred'] == 0])))
print("number of rows missing age: {0}".format(len(df.loc[df['age'] == 0])))
print("number of rows missing thickness: {0}".format(len(df.loc[df['thickness'] == 0])))


# In[21]:


.30*768


# In[22]:


df.drop("insulin",axis=1,inplace=True)


# In[23]:


df.head()


# In[24]:


df.num_preg.isnull().sum()


# In[25]:


df=df.loc[df.glucose_conc!=0]


# In[26]:


print("number of rows missing glucose_conc: {0}".format(len(df.loc[df['glucose_conc'] == 0])))


# In[27]:


df=df.loc[df.diastolic_bp!=0]
print("number of rows missing diastolic_bp: {0}".format(len(df.loc[df['diastolic_bp'] == 0])))


# In[28]:


df=df.loc[df.bmi!=0]
print("number of rows missing bmi: {0}".format(len(df.loc[df['bmi'] == 0])))


# In[29]:


import numpy as np
df.thickness.replace(0,df.thickness.median(),inplace=True)


# In[30]:


print("total number of rows : {0}".format(len(df)))
print("number of rows missing glucose_conc: {0}".format(len(df.loc[df['glucose_conc'] == 0])))
print("number of rows missing diastolic_bp: {0}".format(len(df.loc[df['diastolic_bp'] == 0])))
print("number of rows missing bmi: {0}".format(len(df.loc[df['bmi'] == 0])))
print("number of rows missing diab_pred: {0}".format(len(df.loc[df['diab_pred'] == 0])))
print("number of rows missing age: {0}".format(len(df.loc[df['age'] == 0])))
print("number of rows missing thickness: {0}".format(len(df.loc[df['thickness'] == 0])))


# In[31]:


df.head()


# In[32]:


df.shape


# In[33]:


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB


# In[34]:


#Create a Gaussian Classifier
model = GaussianNB()


# In[35]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(df.drop("diabetes",axis=1), df.diabetes, test_size=0.3,random_state=109)


# In[36]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[37]:


# Train the model using the training sets
model.fit(X_train,y_train)


# In[38]:


#Predict the response for test dataset
y_pred = model.predict(X_test)


# In[39]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[42]:



## Apply Algorithm

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)

random_forest_model.fit(X_train, y_train)


# In[43]:


predict_train_data = random_forest_model.predict(X_test)

from sklearn import metrics

print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))


# In[68]:



## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


# In[69]:


## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xgboost


# In[70]:


classifier=xgboost.XGBClassifier()


# In[71]:


random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)


# In[72]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[73]:


from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(df.drop("diabetes",axis=1), df.diabetes)
timer(start_time) # timing ends here for "start_time" variable


# In[74]:


random_search.best_estimator_


# In[75]:


# cross validation for random forest
classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.1,
              learning_rate=0.2, max_delta_step=0, max_depth=3,
              min_child_weight=3, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)


# In[76]:



from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,df.drop("diabetes",axis=1), df.diabetes,cv=10)


# In[77]:


score


# In[78]:


score.mean()


# In[79]:


#cross validation for naive bayes classifier

classifier=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=10, verbose=0,
                       warm_start=False)
score=cross_val_score(classifier,df.drop("diabetes",axis=1), df.diabetes,cv=10)


# In[80]:


score


# In[81]:


score.mean()


# In[82]:


classifier=GaussianNB(priors=None, var_smoothing=1e-09)
score=cross_val_score(classifier,df.drop("diabetes",axis=1), df.diabetes,cv=10)


# In[83]:


score


# In[84]:


score.mean()


# In[ ]:


#naive bayes is giving us best accuracy

