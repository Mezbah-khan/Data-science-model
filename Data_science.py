   # hello wrold 
   # lets prepaired the dataset 

import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
import time , os 
from pathlib import Path
from icecream import ic 

   # lets work on dataset 
data_path = Path("data.csv")
if data_path.exists() : 
    data = pd. read_csv(data_path) 
    ic (data.head(3))
else: 
    FileNotFoundError(f'The file path{data_path} is not founded .....')

    # lets ork on big dataset ---->
    # lets check the datas info and structure 
ic (data.info()) 
ic (data.isnull().sum())

    # there are  float64(4), object(4)
    # theere are nulls in every columns 
    # lets check the outliers in dataset -->
ic (data.describe()) 

    # No coluns have outliers , So lets delete the columns [Unnamed: 7]
data.drop(columns=['Unnamed: 7'], inplace=True)

   # lets fill the NaN values 
data_fill_obj = data.select_dtypes(['object']).columns
for col in data_fill_obj:
    data[col].fillna(data[col].mode()[0], inplace=True)

data_fill_num = data.select_dtypes(['float64']).columns
for col in data_fill_num:
    data[col].fillna(data[col].mean(), inplace=True)

ic (data.isnull().sum())

    # lets encode the data 

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder() 
output_01 = pd .DataFrame() 
data_columns_01 = data.select_dtypes(['object']).columns
for cal in data_columns_01 :   
    output_01[cal+"_end"] = label.fit_transform(data[cal])
    ic (output_01)

from sklearn.preprocessing import MinMaxScaler
scalor = MinMaxScaler() 
data_columns_02 = data.select_dtypes(['float64']).columns
output_02 = pd .DataFrame() 
for sal in data_columns_02 : 
    output_02[sal + "_end"] = scalor.fit_transform(data[[sal]]).flatten()
    ic(output_02)

   # lets concat the outputs x1 + x2 
final_concat = pd .concat([output_01, output_02], axis = 1)
final_concat.to_csv('deep_learning.csv', index=False) 


datas = pd .read_csv('deep_learning.csv')
ic (datas.head(5))

                     # Lets use the machine learning algorithoms 
                     # lets do this 

    # first lets select the features 
    # x --> averageRating_end , numVotes_end
    # y --> releaseYear_end  .. Now lets build this model 
    # first lets check there are any linearity or not -->

from sklearn.model_selection import train_test_split
x = datas[['averageRating_end','numVotes_end']]
y = datas['releaseYear_end']
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.2 , random_state=42)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_train['averageRating_end'], y=y_train)
plt.title("Scatterplot: 'averageRating_end' vs 'releaseYear_end'")
plt.xlabel('Average Rating')
plt.ylabel('Release Year')
plt.show()

# 2. Correlation coefficient to quantify linearity
correlation = x_train['averageRating_end'].corr(y_train)
print(f"Correlation Coefficient: {correlation}")

   # there are no relation between x and y colummns 
   # lets move for polynomal regression 
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x_train)  # Transform only the features (x_train)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_poly, y_train)  # Fit the model on transformed features and target

# Check the model accuracy
model_accuracy = model.score(x_poly, y_train) * 100  # Use transformed x_train for accuracy
print(f'This is model accuracy: {model_accuracy}%')  



