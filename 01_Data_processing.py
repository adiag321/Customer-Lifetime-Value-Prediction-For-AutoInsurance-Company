# Problem Statement
# Customer Lifetime Value represents a customer’s value to a company over a period of 
# time. It’s a competitive market for insurance companies in 2019, and insurance 
# premium isn’t the only determining factor in a customer’s decisions. CLV is a 
# customer-centric metric, and a powerful base to build upon to retain valuable 
# customers, increase revenue from less valuable customers, and improve the customer 
# experience overall.
# Auto Insurance company is facing issues in retaining its customers and wants to advertise promotional 
# offers for its loyal customers. They are considering CLV as a parameter to classify loyal customers.

import pandas as pd
import numpy as np
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

# Get project root directory (relative to this script)
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / 'data'

data=pd.read_csv(DATA_DIR / 'AutoInsurance.csv')
data.head()

## Processing the data
def data_processing(df):
    print("The number of rows:", df.shape[0])
    print("The number of columns:",df.shape[1])
    
    df.rename(columns={'Customer Lifetime Value':'CLV'}, inplace=True)
    # basic information
    print("Data information:", df.info())
    # Selecting numerical columns
    numerical_cols = df.select_dtypes(include=["int64","float64"])
    numerical_cols_list = numerical_cols.columns.tolist()
    print("Numerical columns:", numerical_cols_list)
    numerical_cols = numerical_cols.drop(["Number of Policies","Number of Open Complaints"],axis=1)
    
    print("Checking null values in numerical columns:")
    print(numerical_cols.isnull().sum())
    
    # Extracting months from effective date
    df['Effective To Date'] = pd.to_datetime(df['Effective To Date'], infer_datetime_format=True)
    df["Months"] = df["Effective To Date"].dt.month
    df['Months'] = df['Months'].astype('object')
    
    # Selecting categorical columns
    cat_cols = df.select_dtypes(include="object")
    print("Categorical columns:", cat_cols.columns.tolist())
    cat_cols = cat_cols.drop(["Customer"], axis = 1)
    no_col = df[["Number of Open Complaints","Number of Policies"]]
    cat_cols = pd.concat([cat_cols, no_col],axis=1)
    cat_cols.head()

    # Encoding Categorical Columns
    cat_encoded = pd.get_dummies(cat_cols, drop_first=True)
    df_encoded = pd.concat([numerical_cols, cat_encoded],axis=1)

    # Replacing False with 0 and True with 1
    df_encoded = df_encoded.replace({False:0, True:1})
    
    return df_encoded
    

processed_data = data_processing(df = data)
processed_data.to_csv("./data/Processed_AutoInsurance.csv", index=False)