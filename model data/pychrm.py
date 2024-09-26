  # hello world this is mezbah khan from backend developer 
  # lets prepaired the dataset of data science model .....
  # lets do this with proper code 
  
from doctest import OutputChecker
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pathlib as path 
import os , time , select, functools

  # lets load the dataset --> 
data = pd . read_csv('Public_Schools_20240831.csv')
print(data)

  # lets check the datas information and datas structure 
data_info = data . info ()
print(data_info)

  # The dataset have  float64(2), int64(1), object(7) 
  # lets check the datas outliers and datas nullls values 

data_nulls_check = data . isnull().sum()
print(data_nulls_check)

data_describe = data . describe()
print(data_describe) 
  
  # The data have 0 null value  and no outliers in dataset
  # lets check the columns outliers with graph --> 
  
plt . figure(figsize=(5,10))
sns . boxplot(y='ZIP CODE'  , data = data)
plt . title(label='This is ZIP CODE outliers')
plt . grid(True)
plt . show()

plt . figure(figsize=(5,10))
sns . boxplot(y='LONGITUDE'  , data = data)
plt . title(label='This is LONGITUDE outliers')
plt . grid(True)
plt . show()


plt . figure(figsize=(5,10))
sns . boxplot(y='LATITUDE'  , data = data)
plt . title(label='This is LATITUDE outliers')
plt . grid(True)
plt . show()

   #  lets encode the data for model dataset --> 
   #  lets do this with proper code ...

data_columns_01 = data . columns
print(data_columns_01)

data_unique_01 = data['CATEGORY']. unique()
data_unique_02 = data['SCHOOL NAME']. unique()
data_unique_03 = data['ADDRESS'] . unique()
data_unique_04 = data['CITY'] . unique()

print(data_unique_01, data_unique_02)
print(data_unique_03, data_unique_04)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data_columns_02 = data.select_dtypes(['object']).columns
Output_01= data.copy()
for col in data_columns_02 : 
  Output_01[col]  = encoder.fit_transform(data[col])
  print(Output_01) 
  
from sklearn.preprocessing import MinMaxScaler
scalor = MinMaxScaler()
columns_name_03 = data . select_dtypes(['float64']).columns
data_fit_01 = scalor.fit_transform(data[columns_name_03])
Output_02 = pd . DataFrame(data_fit_01, columns=columns_name_03)
print(Output_02)

data_concat_01 =  pd. concat([Output_01, Output_02], axis = 1)

from sklearn.preprocessing import StandardScaler
snd_scaclor = StandardScaler()
columns_name_04 = data.select_dtypes(['int64']).columns
data_fit_02 = snd_scaclor.fit_transform(data[columns_name_04])
Output_03 = pd. DataFrame(data_fit_02, columns=columns_name_04)
print(Output_03)


data_concat_02 = pd . concat([data_concat_01, Output_03], axis = 1)

data_concat_02.to_excel('school_data.xlsx', index=False)
pd . read_excel('school_data.xlsx')