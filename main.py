#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib as jb



# In[2]:


df = pd.read_csv("bmw_global_sales_2018_2025.csv")


# In[5]:


df.dtypes


# In[4]:


model = jb.load("LogisticRegression.pkl")

app = FastAPI(title="BMW model Prediction API")


class SalesData(BaseModel):
    Year: int
    Month: int
    Region: str
    Revenue_EUR: str
    Units_Sold: float
    Avg_Price_EUR: float
    BEV_Share: float
    Premium_Share: float
    GDP_Growth: float
    Fuel_Price_Index: float


@app.get("/")
def home():
    return {"message": "BMW Sales Prediction API is running"}


@app.post("/predict")
def predict(data: SalesData):

    df = pd.DataFrame({
        "Year":[data.Year],
        "Month":[data.Month],
        "Region":[data.Region],
        "Revenue_EUR":[data.Revenue_EUR],
        "Units_Sold":[data.Units_Sold],
        "Avg_Price_EUR":[data.Avg_Price_EUR],
        "BEV_Share":[data.BEV_Share],
        "Premium_Share":[data.Premium_Share],
        "GDP_Growth":[data.GDP_Growth],
        "Fuel_Price_Index":[data.Fuel_Price_Index]
    })

    prediction = model.predict(df)

    return {
        "prediction": float(prediction[0])
    }

