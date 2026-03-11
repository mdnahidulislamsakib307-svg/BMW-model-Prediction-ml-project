#!/usr/bin/env python
# coding: utf-8

# In[8]:


import joblib as jb
import streamlit as st
import pandas as pd 


# In[9]:


df=pd.read_csv("bmw_global_sales_2018_2025.csv")


# In[10]:


df.head()


# In[11]:


df.dtypes


# In[13]:


load_model = jb.load("LogisticRegression.pkl")

st.title("BMW model Prediction App")

st.write("Enter BMW model information to make a prediction")

Year = st.number_input("Year")
Month = st.number_input("Month")

Region = st.selectbox(
    "Region",
    ["Europe", "Asia", "North America", "Other"]
)



Units_Sold= st.number_input("Units Sold")

Avg_Price_EUR= st.number_input("Average Price (EUR)")
Revenue_EUR = st.number_input('Revenue_EUR')

BEV_Share = st.number_input("BEV Share")

Premium_Share  = st.number_input("Premium Share")

GDP_Growth = st.number_input("GDP Growth")

Fuel_Price_Index = st.number_input("Fuel Price Index")


# Prediction
if st.button("Predict"):

    data = pd.DataFrame({
        "Year":[Year],
        "Month":[Month],
        "Region":[Region],
        "Revenue_EUR":[Revenue_EUR],
        "Units_Sold":[Units_Sold],
        "Avg_Price_EUR":[Avg_Price_EUR],
        "BEV_Share":[BEV_Share],
        "Premium_Share":[Premium_Share],
        "GDP_Growth":[GDP_Growth],
        "Fuel_Price_Index":[Fuel_Price_Index]
    })

    prediction = load_model.predict(data)

    st.success(f"Predicted Value: {prediction[0]}")


# In[ ]:





# In[ ]:




