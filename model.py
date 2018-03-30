#encoding:utf-8
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from feature_convert import load_data, convert_data

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

    df = load_data()
    df = convert_data(df)
    features_use = df.columns.tolist()
    no_use = ['instance_id', 'is_train', 'is_trade']
    features_use = [i for i in features_use if i not in no_use]
    #print(features_use)
    target = ['is_trade']

    train_data = df[df.is_train==1].copy()
    if online == False:
        train = train_data[train_data.day < 24]  # 18,19,20,21,22,23,24
        test = train_data[train_data.day == 24]  # 暂时先使用第24天作为验证集
    else:
        train = train_data.copy()
        test = df[df.is_train==0].copy()
	print test.columns
    result = XgbModel(train, test, features_use, target)
    if online == False:
        # 测试训练准确率
        print(log_loss(test[target], test['predicted_score']))
    else:
        # 保存在线提交结果
        result[['instance_id', 'predicted_score']].to_csv('xgboost_baseline.csv', index=False, sep=' ')



