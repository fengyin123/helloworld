from utils import BayesianSmoothing, load_pickle, dump_pickle, raw_data_path
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

train = load_pickle('../data/train.pkl')
test = load_pickle('../data/test.pkl')
df = load_pickle('../data/df.pkl')

print(test.day.value_counts())
# train = pd.concat([train, test])

iter_num = 200
epsilon = 0.001
'''
1. 定义需要计算平滑点击率的变量
2. 对于每一天，找出在这之前的所有点击行为
3. 统计该变量的点击次数和购买次数
'''
# smooth_cols = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
smooth_cols = ['user_id']

smooth_train = df[smooth_cols + ['day']]
smooth_test = df[smooth_cols + ['day']]
for col in tqdm(smooth_cols):
    col_I = '{}_I'.format(col)
    col_C = '{}_C'.format(col)
    col_smooth_rate = '{}_smooth_rate'.format(col)
    train[col_smooth_rate] = -1
    smooth_all = pd.DataFrame({'day': train.day, '{}'.format(col): train[col]})
    CVR_all = None
    for day in tqdm(range(19, 26)):
        I = train[train.day<day].groupby(col)['is_trade'].count().reset_index()
        I.columns = [col, col_I]
        C = train[train.day<day].groupby(col)['is_trade'].sum().reset_index()
        C.columns = [col, col_C]
        CVR = pd.concat([I, C[col_C]], axis=1)
        CVR['day'] = day

        smooth = BayesianSmoothing(1, 1)
        smooth.update(CVR[col_I].values, CVR[col_C].values, iter_num, epsilon)
        alpha = smooth.alpha
        beta = smooth.beta
        CVR[col_smooth_rate] = (CVR[col_C] + alpha) / (CVR[col_I] + alpha + beta)
        CVR_all = pd.concat([CVR_all, CVR], axis=0)
        # print(CVR.head())
        # smooth_all[col_smooth_rate] = -1
        # print((pd.merge(train[train.day == day], CVR[[col, col_smooth_rate]], on=col, how='inner')).columns[-1])
        # smooth_all[col_smooth_rate][smooth_all.day == day] = (pd.merge(train[train.day == day], CVR[[col, col_smooth_rate]], on=col, how='left')).iloc[:, -1].values

    # smooth_all = pd.concat([smooth_all, smooth_feat], axis=1)
    # print(smooth_all.columns)
    smooth_train = pd.merge(smooth_train, CVR_all[[col, 'day', col_smooth_rate]], on=[col, 'day'], how='left')
    smooth_test = pd.merge(smooth_test, CVR_all[[col, 'day', col_smooth_rate]], on=[col, 'day'], how='left')

# smooth_all = pd.concat([smooth_train, smooth_test], axis=1)
# smooth_all.drop(smooth_cols + ['day'], axis=1, inplace=True)
dump_pickle(smooth_train, path='../data/train_feature/102_smooth_features.pkl')
dump_pickle(smooth_test, path='../data/test_feature/102_smooth_features.pkl')






