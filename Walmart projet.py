#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install plotly')


# In[2]:


get_ipython().system('pip install holidays')


# In[3]:


import pandas as pd
import seaborn as sns
import numpy as np
import random
import holidays
from datetime import datetime
from datetime import date
from random import randint
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
# setting Jedha color palette as default
pio.templates["jedha"] = go.layout.Template(
    layout_colorway=["#4B9AC7", "#4BE8E0", "#9DD4F3", "#97FBF6", "#2A7FAF", "#23B1AB", "#0E3449", "#015955"]
)
pio.templates.default = "jedha"
#pio.renderers.default = "svg" # to be replaced by "iframe" if working on JULIE
pio.renderers.default = "iframe" # to be replaced by "iframe" if working on JULIE


# In[4]:


import pandas as pd
import random
from datetime import datetime
from datetime import date
import holidays
from holiday import Holiday


# # La base de donnée

# In[5]:


dataset = pd.read_csv("Walmart_Store_sales.csv")
print("DATASET")
display(dataset.head())
print(" ")
print("Dataset Shape")
display(dataset.shape)
print(" ")
display(dataset.info())
print(" ")
display(dataset.describe())
print(" ")
print("Missing Values percentage")
missing_values = dataset.isnull().sum()/ dataset.shape[0]*100
missing_values.sort_values(ascending=False)


# # Data cleaning

# ###     -> Valeur cible

# - Suppression des lignes avec des valeurs manquantes pour la valeur cible

# In[6]:


dataset = dataset.dropna(subset=["Weekly_Sales"])


# ###     -> Date

# - Gestion de la colonne date

# In[7]:


# Conversion de la colonne date en datetime

dataset["Date"] = pd.to_datetime(dataset["Date"])
dataset.info()


# In[8]:


# Remplacer les dates manquantes par des dates aléatoire


# In[9]:


missing_values = dataset.isnull().sum()
missing_values


# In[10]:


#Identifier la plage

start_date = date(2010, 1, 1)
end_date = date(2012, 12, 31)


dataset["Date"] = dataset.Date.fillna(pd.Series(pd.date_range(start_date, end_date)))
dataset


# In[11]:


missing_values = dataset.isnull().sum()
missing_values


# - Gestion de la colonne Holiday_Flag 

# In[12]:


#Répartition des jours fériés

dataset_holiday = dataset.groupby("Holiday_Flag")["Store"].count().reset_index()
display(dataset_holiday)


# In[13]:


#import des jours fériés du calendrier US

us_holidays = holidays.US()


# In[14]:


# Remplacer les données manquantes par les jours fériés correspondants

for i in dataset["Date"]:
    if i in us_holidays:
        dataset["Holiday_Flag"].fillna(value=1, inplace=True)
    else:
        dataset["Holiday_Flag"].fillna(value=0, inplace=True)


# In[15]:


dataset_holiday = dataset.groupby("Holiday_Flag")["Store"].count().reset_index()
display(dataset_holiday)

missing_values = dataset.isnull().sum()
missing_values


# - Separation de la colonne Date

# In[16]:


dataset["Year"]= pd.DatetimeIndex(dataset["Date"]).year
dataset["Month"]= pd.DatetimeIndex(dataset["Date"]).month
dataset["Day"]= pd.DatetimeIndex(dataset["Date"]).day
dataset["Day_of_week"]= pd.DatetimeIndex(dataset["Date"]).dayofweek


# In[17]:


dataset = dataset.drop("Date", axis=1)
dataset.head()


#   

# ###     -> Outlier

# In[18]:


mask_columns_outliers = dataset.loc[:, ["Temperature", "Fuel_Price", "CPI", "Unemployment"]]
desc = mask_columns_outliers.describe()
desc.head()


# In[19]:


#Supression des outliers

col_out = mask_columns_outliers
mask = True
for col in col_out:
    q1 = desc.loc["25%", col]
    q3 = desc.loc["75%", col]
    ecart = q3 - q1
    cond1 = q1 - 1.5*ecart < mask_columns_outliers[col]
    cond2 = q3 + 1.5*ecart > mask_columns_outliers[col]
    mask = mask & cond1 & cond2
    
    
mask.value_counts()

dataset = dataset.loc[mask,:]


# ## Resumé de la base de donnée

# In[20]:


print("DATASET")
display(dataset.head())
print(" ")
print("Dataset Shape")
display(dataset.shape)
print(" ")
display(dataset.info())
print(" ")
display(dataset.describe())
print(" ")
print("Missing Values percentage")
missing_values = dataset.isnull().sum()/ dataset.shape[0]*100
missing_values.sort_values(ascending=False)


# ## ANALYSE ##

# In[21]:


dataset.to_csv(r'Dataset.csv', index = False)


# ## GET DUMMIES

# In[22]:


dataset = pd.get_dummies(data=dataset, columns=["Store", "Holiday_Flag", "Year", "Month", "Day", "Day_of_week"])


# In[23]:


dataset


# ## PREPROCESSING : Separate X et Y 

# In[24]:


X = dataset.loc[:, dataset.columns !="Weekly_Sales"]
Y = dataset.loc[:, "Weekly_Sales"]


# In[25]:


display(X.head())
display(Y.head())


# ## Preprocessing - scikit-learn ##

# In[26]:


from sklearn.model_selection import train_test_split
print("Dividing into train and test sets...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print("...Done.")
print()


# ## Column Tranformer ##

# In[27]:


# Create pipeline for numeric features
numeric_features = [2, 3, 4, 5] 
numeric_transformer = StandardScaler()


# In[28]:


# Create pipeline for categorical features
categorical_features = [0, 1, 6, 7, 8, 9]

categorical_transformer = OneHotEncoder(handle_unknown='ignore')


# In[29]:


# Use ColumnTranformer to make a preprocessor object that describes all the treatments to be done
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])


# In[30]:


# Preprocessings on train set
print("Performing preprocessings on train set...")
print(X_train[0:5])
X_train = preprocessor.fit_transform(X_train)
print('...Done.')
print(X_train[0:5])
print()

# Preprocessings on test set
print("Performing preprocessings on test set...")
print(X_test[0:5])
X_test = preprocessor.transform(X_test) 
print('...Done.')
print(X_test[0:5])
print()


#  

# # Model training

# In[31]:


Linear_Regression = LinearRegression()

Lasso = linear_model.Lasso()

Ridge = linear_model.Ridge()


# In[32]:


print("Training model...")
Linear_Regression.fit(X_train, Y_train)
Lasso.fit(X_train, Y_train)
Ridge.fit(X_train, Y_train)
print("...Done.")


#  

# ## Predictions

# In[33]:


print("Predictions on training set...")
Y_train_pred_reg = Linear_Regression.predict(X_train)
Y_train_pred_lasso = Lasso.predict(X_train)
Y_train_pred_ridge = Ridge.predict(X_train)
print("...Done.")
print(Y_train_pred_reg[0:5])
print(Y_train_pred_lasso[0:5])
print(Y_train_pred_ridge[0:5])
print()


# In[34]:


# Predictions on test set
print("Predictions on test set...")
Y_test_pred_reg = Linear_Regression.predict(X_test)
Y_test_pred_lasso = Lasso.predict(X_test)
Y_test_pred_ridge = Ridge.predict(X_test)
print("...Done.")
print(Y_test_pred_reg[0:5])
print(Y_test_pred_lasso[0:5])
print(Y_test_pred_ridge[0:5])
print()


# ## Score 

# In[35]:


# Print scores

r2score_training_reg =r2_score(Y_train, Y_train_pred_reg)
r2score_test_reg = r2_score(Y_test, Y_test_pred_reg)

r2score_training_lasso =r2_score(Y_train, Y_train_pred_lasso)
r2score_test_lasso = r2_score(Y_test, Y_test_pred_lasso)

r2score_training_ridge =r2_score(Y_train, Y_train_pred_ridge)
r2score_test_ridge = r2_score(Y_test, Y_test_pred_ridge)

score = {"Model":["Linear Regression", "Lasso", "Ridge"], 
        "R2-score on training set" : [r2score_training_reg, r2score_training_lasso, r2score_training_ridge],
         "R2-score on test set" : [r2score_test_reg, r2score_test_lasso, r2score_test_ridge]}

score = pd.DataFrame(score)
score


# #### The models overfit a lot.
# #### I will search for hyperparameter to fit it. 

# In[36]:


lasso_params = {'alpha': np.arange (0, 1, 0.01)}
ridge_params = {'alpha': np.arange(100, 1000, 10)}

Linear_Regression = LinearRegression()

Lasso = GridSearchCV(linear_model.Lasso(), 
                     param_grid=lasso_params,
                    cv=3)

Ridge = GridSearchCV(linear_model.Ridge(), 
                     param_grid=ridge_params,
                    cv=3)


# In[37]:


print("Training model...")
Linear_Regression.fit(X_train, Y_train)
Lasso.fit(X_train, Y_train).best_estimator_
Ridge.fit(X_train, Y_train).best_estimator_
print("...Done.")


# In[38]:


print("Predictions on training set...")
Y_train_pred_reg = Linear_Regression.predict(X_train)
Y_train_pred_lasso = Lasso.predict(X_train)
Y_train_pred_ridge = Ridge.predict(X_train)
print("...Done.")
print(Y_train_pred_reg[0:5])
print(Y_train_pred_lasso[0:5])
print(Y_train_pred_ridge[0:5])
print()


# In[39]:


# Predictions on test set
print("Predictions on test set...")
Y_test_pred_reg = Linear_Regression.predict(X_test)
Y_test_pred_lasso = Lasso.predict(X_test)
Y_test_pred_ridge = Ridge.predict(X_test)
print("...Done.")
print(Y_test_pred_reg[0:5])
print(Y_test_pred_lasso[0:5])
print(Y_test_pred_ridge[0:5])
print()


# In[40]:


# Print scores

r2score_training_reg =r2_score(Y_train, Y_train_pred_reg)
r2score_test_reg = r2_score(Y_test, Y_test_pred_reg)

r2score_training_lasso =r2_score(Y_train, Y_train_pred_lasso)
r2score_test_lasso = r2_score(Y_test, Y_test_pred_lasso)

r2score_training_ridge =r2_score(Y_train, Y_train_pred_ridge)
r2score_test_ridge = r2_score(Y_test, Y_test_pred_ridge)


score = {"Model":["Linear Regression", "Lasso", "Ridge"], 
        "R2-score on training set" : [r2score_training_reg, r2score_training_lasso, r2score_training_ridge],
         "R2-score on test set" : [r2score_test_reg, r2score_test_lasso, r2score_test_ridge],
         }

score = pd.DataFrame(score)
score

