# helllo wrold this is mezbah kham from bacckend developer 
# lets create pratice set oon data science and lets build the product 
# lets od this with code ...> 

from flask.testing import FlaskClient
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn as skn

     # lers load the dataset as data
data = pd . read_excel ('database.xlsx')
     
     # lets check the info and nulls in dataseet 
data_check_info = data.info ()
data_check_nullls = data.isnull().sum()
print('this is data info->\n', data_check_info, 'this is data nulls --->',data_check_nullls)

    # lets check the outliers aand removve it frist 
data_check_describe = data.describe()
print(data_check_describe)

sns . boxplot(x='bill_length_mm'  , data=data)
plt.figure(figsize=(12,6))


sns.boxplot(x='bill_depth_mm' , data=data)
plt.figure(figsize=(12,6))


     # lets remvoe the outliers frsit ---> 
q1_01 = data['bill_length_mm'] . quantile(0.25)
q3_01 = data['bill_length_mm'].  quantile(0.75)
IQR_1 = q3_01 - q1_01

min_range_01 = q1_01 - (1.5 * IQR_1)
max_range_01 = q3_01 + (1.5 * IQR_1)

data = data[(data['bill_length_mm'] >= min_range_01) & (data['bill_length_mm'] <= max_range_01)] 
print('this is datas nature: ',data['bill_depth_mm'].describe())

q1_02 = data['bill_depth_mm']. quantile(0.25)
q3_02 = data['bill_length_mm'].quantile(0.75)
IQR_2  = q3_02 -q1_02

min_range_02 = q1_02 - (1.5 * IQR_2)
max_range_02 = q3_02 + (1.5 * IQR_2)


data =  data[(data['bill_length_mm'] >= min_range_02) & (data['bill_length_mm'] <= max_range_02)]
print('this is datas nature: ',data['bill_length_mm'].describe()) 

    # outlier was removed now lets use the minmaxscaling -->
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data['flipper_length_mm_scaled'] = scaler.fit_transform(data[['flipper_length_mm']])
print(data) 

    #lets fill the missing value
data['species']   .ffill(inplace=True)
data['island']    .ffill(inplace=True)
data['sex']       . bfill(inplace=True)
data['Index']     .fillna(data['Index'].mode()[0], inplace=True) 
data['bill_depth_mm']. fillna(data['bill_depth_mm'].mode()[0], inplace=True) 
data['bill_length_mm']. fillna(data['bill_length_mm'].mode()[0], inplace=True) 
data['flipper_length_mm_scaled']  . fillna(data['flipper_length_mm_scaled'].mode()[0],inplace=True) 
data['body_mass_g']. fillna(data['body_mass_g'].mode()[0], inplace=True) 
    
    # lets recheck the data 
data_recheck_nulls = data.isnull().sum()
print(data_recheck_nulls)

   # lets encoded the data into bainary 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

encoder_01 = OneHotEncoder(sparse_output=False)
data_fit_01 = encoder_01.fit_transform(data[['sex']]) 
output_01 = pd . DataFrame(data_fit_01, columns=encoder_01.get_feature_names_out(['sex']))
print(output_01)

encoder_02 = LabelEncoder()
data_fit_02 = encoder_02.fit_transform(data['species'])
data_fit_03 = encoder_02.fit_transform(data['island'].astype(str))
data_fit_04 = encoder_02.fit_transform(data['bill_depth_mm'])
data_fit_05=  encoder_02.fit_transform(data['bill_length_mm'])
data_fit_06 = encoder_02.fit_transform(data['body_mass_g'])

output_2= pd . DataFrame( { 
                        'species' : data_fit_02 ,
                        'island'  : data_fit_03, 
                        'bill_depth_mm'  : data_fit_04,
                        'bill_length_mm'  : data_fit_05,
                        'body_mass_g'   : data_fit_06   }
                         )
print(output_2)


    # lets concctinate the data 
data_concate = pd.concat([output_2, output_01], axis=1)
data_concate.to_excel('database01.xlsx', index=False)
datas = pd.read_excel('database01.xlsx')
print(datas)
