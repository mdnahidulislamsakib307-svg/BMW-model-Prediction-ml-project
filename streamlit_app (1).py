#!/usr/bin/env python
# coding: utf-8

# In[2]:


import joblib as jb
import streamlit as st
import pandas as pd 


# In[3]:


df=pd.read_csv("bmw_global_sales_2018_2025.csv")


# In[4]:


df.head()


# In[5]:


df.dtypes


# In[16]:


import streamlit as st
import requests

st.title("BMW Sales Prediction App")

Year = st.number_input("Year")
Month = st.number_input("Month")

Region = st.selectbox(
    "Region",
    ["Europe", "Asia", "North America", "Other"]
)

Revenue_EUR = st.number_input("Revenue EUR")
Units_Sold = st.number_input("Units Sold")
Avg_Price_EUR = st.number_input("Average Price EUR")
BEV_Share = st.number_input("BEV Share")
Premium_Share = st.number_input("Premium Share")
GDP_Growth = st.number_input("GDP Growth")
Fuel_Price_Index = st.number_input("Fuel Price Index")

if st.button("Predict"):

    data = {
        "Year":Year,
        "Month":Month,
        "Region":Region,
        "Revenue_EUR":Revenue_EUR,
        "Units_Sold":Units_Sold,
        "Avg_Price_EUR":Avg_Price_EUR,
        "BEV_Share":BEV_Share,
        "Premium_Share":Premium_Share,
        "GDP_Growth":GDP_Growth,
        "Fuel_Price_Index":Fuel_Price_Index
    }

    response = requests.post("YOUR_API_URL/predict", json=data)

    prediction = response.json()["prediction"]

    st.success(f"Prediction: {prediction}")

