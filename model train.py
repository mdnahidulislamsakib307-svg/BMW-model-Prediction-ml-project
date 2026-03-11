#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


df = pd.read_csv("bmw_global_sales_2018_2025.csv")


# In[3]:


df.shape


# In[6]:


df.head()


# In[7]:


df.dtypes


# In[4]:


df.isnull().sum()


# In[5]:


df.duplicated().sum()


# In[8]:


x = df.drop(['Model'],axis=1)
y=df['Model']


# In[12]:


numerical_cols =x.select_dtypes(include=['int64','float64']).columns.tolist()


# In[13]:


categorical_cols = x.select_dtypes(include=['object']).columns.tolist()


# In[14]:


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])



# In[15]:


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[16]:


preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[18]:


model = Pipeline(steps=[
    ('pre',preprocessor),('reg',LogisticRegression(max_iter=1000)),
])


# In[19]:


model.fit(X_train,y_train)


# In[22]:


y_pred= model.predict(X_test)
print(f'Accuracy:{accuracy_score(y_test,y_pred)*100:.2f}')
print(f'{classification_report(y_test,y_pred,zero_division=0)}')


# In[23]:


jb.dump(model,'LogisticRegression.pkl')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




