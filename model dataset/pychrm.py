    # Hello world this is mezbah khan from backend developer 
    # Lets prepaired the dataset for data science model 
    # lets do this with proper code ....

import numpy as np 
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 
import os , time , functools

    # lets load the data --> 
data = pd . read_csv('student_data_01.csv')
print(data)

   # frist lets check the data structure and data info 
   # for analysis the data we need to checkout datas catagories .....

data_info = data.info ()
print('This is data information ' , data_info)

   # There are [6607 rows x 20 columns] and int64(7), object(13)
   # it loook like the are somne null values in dataset in [Teacher_Quality , Parental_Education_Level , Distance_from_Home ]
   
   # lets check the outliers and remove the duplicated rows 
   # lets do this with grpah and Iqr method --> 
   
data_describe = data. describe()
print(data_describe) 

   # Its seems some columns have outliers like [Hours_Studied ,Tutoring_Sessions , Physical_Activity ,]
   # lets view and remove --> 

plt . figure(figsize=(5,10))
sns . boxenplot(y='Hours_Studied', data = data)
plt . title('This is Hours_Studied outliers ')
plt . grid(True)
plt . show()


plt . figure(figsize=(5,10))
sns . boxenplot(y='Tutoring_Sessions', data= data)
plt . title('This is Tutoring_Sessions outliers')
plt . grid(True)
plt . show()


plt . figure(figsize=(5,10))
sns . boxenplot(y='Physical_Activity', data = data ) 
plt . title('This is Physical_Activit outlier')
plt . grid(True)
plt . show()

q1_01 = data['Hours_Studied'] . quantile(0.25)
q3_01   = data['Hours_Studied'] . quantile(0.75)
IQR_01 = q3_01 - q1_01
min_range_01 = q1_01 - (1.5 * IQR_01)
max_range_01 = q3_01 + (1.5 * IQR_01)
data = data[(data['Hours_Studied'] > min_range_01)  & ( data['Hours_Studied'] < max_range_01) ]

q1_02 = data['Tutoring_Sessions']. quantile(0.25)
q3_02 = data['Tutoring_Sessions']. quantile(0.75)
IQR_02 = q3_02 - q1_02
min_range_02 = q1_02 - (1.5 * IQR_02)
max_range_02 = q3_02 + (1.5 * IQR_02)
data = data [(data['Tutoring_Sessions'] > min_range_02)  & (data['Tutoring_Sessions'] < max_range_02) ]

data_recheck = data . describe()
print(data_recheck)
  
    # lets remvoe the duplicated data or rows formj dataset 
data = data.drop_duplicates()
print(data)

    #lets fill the nulls values and then use encoding for model 
    # frist fill the value then encode 
    
print(data.isnull().sum())
data['Teacher_Quality'] . fillna(data['Teacher_Quality'].mode()[0], inplace=True) 
data['Parental_Education_Level']. fillna(data['Parental_Education_Level'].mode()[0], inplace=True) 
data['Distance_from_Home']  . fillna(data['Distance_from_Home'].mode()[0], inplace= True) 
print(data.isnull().sum()) 

   # every process is done and lets encode the data 
   # first lets check the data object types and convert them into bainary formate ....
   
   

from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# Label Encoding for object columns
encoder = LabelEncoder()
columns_name_01 = data.select_dtypes(['object']).columns

encoded_data = pd.DataFrame()  # Store encoded columns

for col in columns_name_01:
    output_01 = encoder.fit_transform(data[col])
    encoded_data[col] = output_01  # Save encoded data as DataFrame
    print(output_01)

# Standard Scaling for integer columns
scaler = StandardScaler()
columns_name_02 = data.select_dtypes(['int64']).columns

scaled_data = pd.DataFrame()  # Store scaled columns

for col in columns_name_02:
    output_02 = scaler.fit_transform(data[[col]])
    scaled_data[col] = output_02.flatten()  # Save scaled data as DataFrame
    print(output_02)

# Concatenate scaled and encoded data along the column axis (axis=1)
concat_data = pd.concat([scaled_data, encoded_data], axis=1)
print(concat_data)

datas = concat_data.to_excel('model_dataset.xlsx', index=False)
print(datas)

   # the object type data is encoded into labelencode 
   # the int64 type data is encoded into scalorencode 
   # the dataset is ready for prepaired algorithoms and data science model 
   