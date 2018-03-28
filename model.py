#encoding:utf-8
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

from utils import __convert_timestamp_to_datetime


def convert_data(df):
    """
    按hour和day分离时间特征
    构建 user_query_day和user_query_day_hour特征
    :param data:
    :return:
    """
    df['time'] = df.context_timestamp.apply(__convert_timestamp_to_datetime)
    df['day'] = df.time.apply(lambda x: int(x[8:10]))
    df['hour'] = df.time.apply(lambda x: int(x[11:13]))

    user_query_day = df.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
    df = pd.merge(df, user_query_day, 'left', on=['user_id', 'day'])

    user_query_day_hour = df.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    df = pd.merge(df, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])
    print("========> Convert time Success!")

    return df


def load_data(file='sample.txt'):
    """
    将训练集和测试集一起加载 方便做特征处理
    :return:
    """
    print("========> Start load data!")
    path = './data/'
    # train = pd.read_table(path+'round1_ijcai_18_train_20180301.txt',encoding='utf8',delim_whitespace=True)
    data = pd.read_table(path + file, encoding='utf8', delim_whitespace=True)
    data = data.dropna()

    print("========> Load Data Success!")
    return data


def XgbModel(train, test, features, target):
    """
    xgboost模型
    :param train:
    :param test:
    :param features:
    :param target:
    :return:
    """
    train_data = train[features]
    train_target = train[target]
    test_data = test[features]

    dtrain = xgb.DMatrix(train_data, label=train_target)
    dtest = xgb.DMatrix(test_data)

    num_round = 100
    param = {'max_depth': 6,
             'eta': 0.1,
             'silent': 1,
             'n_estimators': 100,
             'objective': 'binary:logistic',
             "eval_metric": "logloss",
             'min_child_weight': 10,
             "colsample_bytree": 0.3}

    bst = xgb.train(param, dtrain, num_round)
    test['predicted_score'] = bst.predict(dtest)
    return test


if __name__=='__main__':
    online = True

    features = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
                'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                'context_page_id', 'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level',
                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                ]
    target = ['is_trade']

    df = load_data()
    df = convert_data(df)


    if online == False:
        train = df[df.day < 24]  # 18,19,20,21,22,23,24
        test = df[df.day == 24]  # 暂时先使用第24天作为验证集
    else:
        train = df.copy()
        test = load_data('test_sample.txt')
        test = convert_data(test)
    result = XgbModel(train, test, features, target)
    if online == False:
        # 测试训练准确率
        print(log_loss(test[target], test['predicted_score']))
    else:
        # 保存在线提交结果
        result[['instance_id', 'predicted_score']].to_csv('xgboost_baseline.csv', index=False, sep=' ')


