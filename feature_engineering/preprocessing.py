
# coding: utf-8

# In[12]:



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
    '''
    item_id, item_category_list, item_property_list, item_brand_id, item_city_id, 
    item_price_level, item_sales_level, item_collected_level, item_pv_level
    '''
    for i in range(1, 3):
        data['item_category_list' + str(i)] = lbl.fit_transform(data['item_category_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else 'missing'))
    del data['item_category_list'] 
        
    for i in range(10):
        data['item_property_list' + str(i)] = lbl.fit_transform(data['item_property_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    del data['item_property_list']
    # train_item_property = data['item_property_list'].str.split(';', expand=True).add_prefix('item_property_')
    # train_item_property.fillna('missing', inplace=True)
    # train_item_property = lbl.fit_transform(train_item_property)
    # data = pd.concat([data, train_item_property], axis=1)

    for col in ['item_id', 'item_brand_id', 'item_city_id']:
        data[col] = lbl.fit_transform(data[col])
    
    # Fill none with mean
    data['item_sales_level'][data.item_sales_level==-1] = None
    data['item_sales_level'].fillna(data['item_sales_level'].mean(), inplace=True)
    
    
    print("========================user==========================")
    # user_gender_id and user_occupation_id should be handled with one-hot
    data[data.user_age_level==-1]['user_age_level'] = None
    data['user_age_level'].fillna(data['user_age_level'].mode())
    data['user_age_level'] = data['user_age_level'].apply(lambda x: x%1000)
    
    data[data.user_star_level==-1]['user_star_level'] = None
    data['user_star_level'].fillna(data['user_star_level'].mean())
    data['user_star_level'] = data['user_star_level'].apply(lambda x: x%3000)
    
    
    print("=====================context==========================")
    data = date_convert(data)
    
    for i in range(5):
        data['predict_category_property' + str(i)] = lbl.fit_transform(data['predict_category_property'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    del data['predict_category_property'] 
        
    print("=====================shop===============================")
    data['shop_score_service'][data.shop_score_service==-1] = None
    data['shop_score_service'].fillna(data['shop_score_service'].mean(), inplace=True)
    
    data['user_age_level'][data.user_age_level==-1] = None
    data['shop_score_delivery'].fillna(data['shop_score_delivery'].mean(), inplace=True)
    
    data['shop_score_description'][data.shop_score_description==-1] = None
    data['shop_score_description'].fillna(data['shop_score_description'].mean(), inplace=True)
    
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
    print(df.day.unique())
    # train = df[(df['day'] >= 18) & (df['day'] <= 23)]
    # valid = df[(df['day'] == 24)]
    # dump_pickle(train, path=raw_data_path + 'train.pkl')
    # dump_pickle(valid, path=raw_data_path + 'valid.pkl')
    #
    # test = df.iloc[len_train:]
    # dump_pickle(test, path=raw_data_path + 'test.pkl')
    
    end = time.time()
    print("Preprocessing done and time elapsed %s" % (end-start))

