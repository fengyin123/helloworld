
# coding: utf-8

# In[15]:



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing
import warnings
import datetime

warnings.filterwarnings("ignore")

import time

from utils import raw_data_path, dump_pickle

path = '../data/'
train_file = 'round1_ijcai_18_train_20180301.txt'
test_file = 'round1_ijcai_18_test_a_20180301.txt'

# def load_data():
#     train = pd.read_table(path + train_file, encoding='utf8', delim_whitespace=True)
#     test = pd.read_table(path + train_file, encoding='utf8', delim_whitespace=True)
#     df = pd.concat([train, test], axis=0, ignore_index=True)

def date_convert(data):
    # Transform into datetime format
    data['time'] = pd.to_datetime(data.context_timestamp, unit='s')
    
    # transform into Beijing datetime format
    data['realtime'] = data['time'].apply(lambda x: x + datetime.timedelta(hours=8))
    data['day'] = data['realtime'].dt.day
    data['hour'] = data['realtime'].dt.hour
    
    return data

def base_process(data):
    lbl = preprocessing.LabelEncoder()
    print("========================item==========================")
    # Divided into different category levels and LabelEncoder()
    for i in range(1, 3):
        data['item_category_list' + str(i)] = lbl.fit_transform(data['item_category_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))  
    for i in range(10):
        data['item_property_list' + str(i)] = lbl.fit_transform(data['item_property_list'].map(lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    for col in ['item_id', 'item_brand_id', 'item_city_id']:
        data[col] = lbl.fit_transform(data[col])
    
    # Fill none with mean
    data[data.item_sales_level==-1] = None;
    data['item_sales_level'].fillna(data['item_sales_level'].mean())
    
    
    print("========================user==========================")
    # user_gender_id and user_occupation_id should be handled with one-hot
    data[data.user_age_level==-1] = None;
    data['user_age_level'].fillna(data['user_age_level'].mean())
    data['user_age_level'] = data['user_age_level'].apply(lambda x: x%1000)
    
    data[data.user_star_level==-1] = None;
    data['user_star_level'].fillna(data['user_star_level'].mean())
    data['user_star_level'] = data['user_star_level'].apply(lambda x: x%3000)
    
   
    
    print("=====================context==========================")
    data = date_convert(data)
    
    for i in range(5):
        data['predict_category_property' + str(i)] = lbl.fit_transform(data['predict_category_property'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
        
    print("=====================shop===============================")
    data[data.shop_score_service==-1] = None;
    data['shop_score_service'].fillna(data['shop_score_service'].mean())
    
    data[data.user_age_level==-1] = None;
    data['shop_score_delivery'].fillna(data['shop_score_delivery'].mean())
    
    data[data.user_age_level==-1] = None;
    data['shop_score_description'].fillna(data['shop_score_description'].mean())
    
    return data
    

if __name__ == "__main__":
    start = time.time()
    print("Load Data")
    train = pd.read_table(path + train_file, encoding='utf8', delim_whitespace=True)
    test = pd.read_table(path + train_file, encoding='utf8', delim_whitespace=True)
    len_train = train.shape[0]
    df = pd.concat([train, test], axis=0, ignore_index=True)
    print("Start doing preprocessing")
    df = base_process(df)
    dump_pickle(df, path=raw_data_path + 'df.pkl')
    
    train = df[(df['day'] >= 18) & (df['day'] <= 23)]
    valid = df[(df['day'] == 24)]
    dump_pickle(train, path=raw_data_path + 'train.pkl')
    dump_pickle(valid, path=raw_data_path + 'valid.pkl')
    
    test = df.iloc[len_train:]
    dump_pickle(test, path=raw_data_path + 'test.pkl')
    
    end = time.time()
    print("Preprocessing done and time elapsed %s" % (end-start))
    
    
    
     


# In[12]:


df.head()

