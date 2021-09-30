#!/usr/bin/env python
# coding: utf-8

# ## IMPORT LIBRARIES

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# # ANALYSE

# In[2]:


dataset = pd.read_csv("Dataset.csv")
dataset.head()              


# In[3]:


dataset["Weekly_Sales"].max()


# In[4]:


dataset_groupby_store = dataset.groupby(["Store"]).mean().reset_index()

px.bar(dataset_groupby_store, x="Store", y="Weekly_Sales", color="Weekly_Sales")


# In[5]:


dataset_groupby_fuel = dataset.groupby("Fuel_Price")["Weekly_Sales"].mean().reset_index()

fig = px.line(dataset_groupby_fuel, x="Fuel_Price", y="Weekly_Sales")
fig.show("iframe")


# In[7]:


dataset_groupby_Day_of_week = dataset.groupby("Day_of_week")["Weekly_Sales"].mean().reset_index()

fig = px.bar(dataset_groupby_Day_of_week, x="Day_of_week", y="Weekly_Sales", color= "Weekly_Sales")
fig.show("iframe")


# In[8]:


dataset_groupby_Month = dataset.groupby("Month")["Weekly_Sales"].mean().reset_index()

fig = px.bar(dataset_groupby_Month, x="Month", y="Weekly_Sales", color= "Weekly_Sales")
fig.show("iframe")


# In[9]:


dataset_groupby_Temperature = dataset.groupby("Temperature")["Weekly_Sales"].mean().reset_index()

fig = px.scatter(dataset_groupby_Temperature, x="Temperature", y="Weekly_Sales", color= "Weekly_Sales")
fig.show("iframe")


# In[10]:


dataset_groupby_Holiday_Flag = dataset.groupby("Holiday_Flag")["Weekly_Sales"].mean().reset_index()

fig = px.bar(dataset_groupby_Holiday_Flag, x="Holiday_Flag", y="Weekly_Sales", color="Holiday_Flag")
fig.show("iframe")


# In[12]:


fig = px.pie(dataset,"Holiday_Flag")
fig.show("iframe")


# In[15]:


#on ne peut pas faire la moyenne car il y a bcp plus de jour non ferié que de jours feriés

dataset_groupby_holiday_flag = dataset.groupby("Holiday_Flag")["Weekly_Sales"].max().reset_index()
display(dataset_groupby_holiday_flag)
px.bar(dataset_groupby_holiday_flag, x="Holiday_Flag", y="Weekly_Sales")


# In[13]:


dataset_groupby_CPI = dataset.groupby("CPI")["Weekly_Sales"].mean().reset_index()

fig = px.line(dataset_groupby_CPI, x="CPI", y="Weekly_Sales")
fig.show("iframe")


# In[16]:


dataset_groupby_temp = dataset.groupby("Temperature")["Weekly_Sales"].mean().reset_index()

fig = px.line(dataset_groupby_temp, x="Temperature", y="Weekly_Sales")
fig.show("iframe")

