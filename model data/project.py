  # Hello wrold this is mezbah khan from backend developer 
  # lets create a praticeset on data-model 
  # lets do this with proper code ...

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import os , time , functools
import pathlib as path 

   # lets load the dataset -->
data = pd . read_csv('student_health_data_01.csv')

   # lets check the datas info and structure ...
print(data.info()) 

   # the dataset have 1000 rows and 15 coluums
   # the data have no NaN value and  float64(8), int64(2), object(5) 
  
data = data.drop(columns=['Unnamed: 14'])
print(data.isnull() .sum())

   # lets check the outliers and sloved it ---> 
print(data.describe())

  # the columns [heart_rate ,Blood_Pressure_Systolic ,Blood_Pressure_Diastolic,Stress_Level_Biosensor ] might have an outliers 
  # lets check the with graph with boxenplot or boxplot 

plt . figure(figsize=(5,10))
sns . boxplot(x='Heart_Rate', data = data )
plt . grid(True)
plt . title('This is Heart_Rate outliers graph ')
plt . show ( )


plt . figure(figsize=(5,10))
sns . boxplot(x='Blood_Pressure_Systolic', data = data )
plt . grid(True)
plt . title('This is Blood_Pressure_Systolic outliers graph ')
plt . show ( )


plt . figure(figsize=(5,10))
sns . boxplot(x='Blood_Pressure_Diastolic', data = data )
plt . grid(True)
plt . title('This is Blood_Pressure_Diastolic outliers graph ')
plt . show ( )


plt . figure(figsize=(5,10))
sns . boxplot(x='Stress_Level_Biosensor', data = data )
plt . grid(True)
plt . title('This is Stress_Level_Biosensor outliers graph ')
plt . show ()


   # the heart_rate need IQR method and last 2 required the repalce method ...

q1_01 = data['Heart_Rate'] . quantile(0.25)
q3_01 = data['Heart_Rate']. quantile(0.75)
IQR_01 = q3_01 - q1_01
min_range_01 = q1_01 - (1.5 * IQR_01 )
max_range_01 = q3_01 + (1.5 * IQR_01 )

data = data[(data['Heart_Rate'] >= min_range_01) & (data['Heart_Rate'] <= max_range_01)]

print(data['Blood_Pressure_Systolic'].unique().max())
print(data['Blood_Pressure_Diastolic'].unique().max())
 
   # lets replace the data 

data['Blood_Pressure_Systolic']  = data['Blood_Pressure_Systolic'] . replace(165.9292045,125.4423)
data['Blood_Pressure_Diastolic'] = data['Blood_Pressure_Diastolic']. replace(107.6597961 , 70.7766)
print(data.describe())


   # lets encode the data in bainary fomate -->
from sklearn.preprocessing import LabelEncoder 
encoder_01 = LabelEncoder()
data_columns_01  = data.select_dtypes(['object']).columns
output_01 = pd . DataFrame()
for obj in data_columns_01 : 
    output_01[obj + '_encoded']  = encoder_01.fit_transform(data[obj])
    print(output_01)

from sklearn.preprocessing import MinMaxScaler
scalor_01 = MinMaxScaler()
data_columns_02 = data.select_dtypes(['float64']).columns
output_02 = pd . DataFrame()
for col in data_columns_02 : 
  output_02[col + '_encoded'] = scalor_01.fit_transform(data[[col]]).flatten()

  print(output_02)

data_concat_01 = pd .concat([output_01, output_02], axis =1 )

from sklearn.preprocessing import StandardScaler
scalor_02 = StandardScaler()
data_columns_03 = data.select_dtypes(['int64']).columns
output_03 = pd . DataFrame()
for int2 in data_columns_03 : 
   output_03[int2 + 'encoded']   = scalor_02.fit_transform(data[[int2]]).flatten()
   print(output_03)

final_concat = pd.concat([data_concat_01, output_03], axis = 1 )
print(final_concat)


final_concat.to_csv('student_health_data_binary.csv', index=False)

# Load the processed dataset
processed_data = pd.read_csv('student_health_data_binary.csv')

# Train-Test Split
X = processed_data[['Project_Hours_encoded']]  # Replace with actual column names
y = processed_data[['Study_Hours_encoded']]    # Replace with actual column names
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.42, random_state=42)

# Display train-test split
print("Training Data:\n", X_train, y_train)
print("Testing Data:\n", X_test, y_test)

