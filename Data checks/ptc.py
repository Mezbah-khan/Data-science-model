    # hello wrold this is mezbah khan from backend devloper 
    # lets pratice on dataset and make a good actions 
    # lets do this with proper code 
    
from operator import index
import pprint
import numpy as np 
import  matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
import sklearn as skn 

    # lets move and load the dataset as data --> 
data = pd . read_excel('database.xlsx')
   
   # lets check the datas info and datas structure 
data_check_info = data.info ()
print(data_check_info)

   # i think there are some  nnulls in  data 
   # lets checkout --> 
   
data_check_nulls = data .isnull().sum()
print(data_check_nulls)

   # lets fill the nulls in dataset 
data['Index']   .fillna(data['Index'].mode()[0], inplace=True) 
data['species'] . ffill(inplace=True)
data['island']  . bfill(inplace=True)
data['bill_length_mm']    . fillna(data['bill_length_mm'].mode()[0], inplace=True) 
data['bill_depth_mm']     . fillna(data['bill_depth_mm'].mode()[0], inplace=True) 
data['flipper_length_mm'] . fillna(data['flipper_length_mm'].mode()[0], inplace=True) 
data['body_mass_g']       . fillna(data['body_mass_g'].mode()[0], inplace=True) 
data['sex']  . ffill(inplace=True)
    
    
    # lets recheck the data and visualized the data --> 
data_check_recheck = data . isnull().sum()
print(data_check_recheck) 

     # lets convert the fresh data into another file (datas)
data. to_excel('database1.xlsx')
datas = pd. read_excel('database1.xlsx')
print(datas) 

     # lets check the datas info
     
datas_check_info = datas . info ()
print(datas_check_info)

     # lets convert the data into bainary formate 
     #  fist lets find the intem of columns 
     
datas_unique_01 = datas['species'] .unique()   
print(datas_unique_01)  # ['Adelie' 'Chinstrap' 'Gentoo' 'Biscoe' 'Dream']

datas_unique_02 = datas['island'] .unique()
print(datas_unique_02)  # ['Torgersen' 'Biscoe' 'Dream' nan]

datas_unique_03 = datas['bill_length_mm'] .unique()

datas_unique_04 = datas['bill_depth_mm'] .unique()

datas_unique_05 = datas['flipper_length_mm'] .unique()

datas_unique_06 = datas['body_mass_g'] .unique()

datas_unique_07 = datas['sex'] .unique()
print(datas_unique_07) # ['sex' , 'female']


      # lets encoded the data 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

     # les encode the basic one 
     
encoder_01 = OneHotEncoder(sparse_output=False)
save_file_01 = encoder_01.fit_transform(datas[['sex']]) 
output_01 = pd . DataFrame(save_file_01 , columns=encoder_01.get_feature_names_out(['sex']))
print(output_01)


     # lets try the label encdoing for (specoies, island)  
     # lets do this 

encoder_02 = LabelEncoder()
encode_species = encoder_02.fit_transform(datas['species'])
encode_island = encoder_02 . fit_transform(datas['island'].astype(str))
output_02 = pd . DataFrame(
    {'species_datas' : encode_species ,
          'island_datas' : encode_island   ,
                            })

print(output_02)
encoder_df = pd .concat([output_01, output_02], axis=1) 
    # the data is encoded 
    # here are a little problem, we cant move other colums for odinal encoding 
    # we must go for minmaxscelor ()
    
from sklearn.preprocessing import MinMaxScaler
columns_to_normalize = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
min_max_scaler = MinMaxScaler()
scaled_data = min_max_scaler.fit_transform(datas[columns_to_normalize])
scaled_df = pd.DataFrame(scaled_data, columns=columns_to_normalize)
print(scaled_df)

final_df = pd. concat([encoder_df, scaled_df], axis=1) 
model_data= final_df.to_excel('database2.xlsx')
print("All columns have been encoded and saved to 'database2.xlsx'.")

    # Read and print the saved file
read_model_data = pd.read_excel('database2.xlsx')
print(read_model_data)


