from utils import BayesianSmoothing, load_pickle, dump_pickle, raw_data_path
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

# train = pd.load_pickle('../data/train.pkl')
# test = pd.load_pickle('../data/test.pkl')
df = load_pickle('../data/df.pkl')
# df = pd.concat([train, test])

iter_num = 200
epsilon = 0.001
'''
1. 定义需要计算平滑点击率的变量
2. 对于每一天，找出在这之前的所有点击行为
3. 统计该变量的点击次数和购买次数
'''
smooth_cols = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']


for col in tqdm(smooth_cols):
    col_I = '{}_I'.format(col)
    col_C = '{}_C'.format(col)
    col_smooth_rate = '{}_smooth_rate'.format(col)
    df[col_smooth_rate] = -1
    smooth_all = None
    for day in tqdm(range(19, 26)):
        I = df[df.day<day].groupby(col)['is_trade'].count().reset_index()
        I.columns = [col, col_I]
        C = df[df.day<day].groupby(col)['is_trade'].sum().reset_index()
        C.columns = [col, col_C]
        CVR = pd.concat([I, C[col_C]], axis=1)


        smooth = BayesianSmoothing(1, 1)
        smooth.update(CVR[col_I].values, CVR[col_C].values, iter_num, epsilon)
        alpha = smooth.alpha
        beta = smooth.beta
        CVR[col_smooth_rate] = (CVR[col_C] + alpha) / (CVR[col_I] + alpha + beta)
        smooth_feat = df.loc[df.day == day, ['instance_id', col]]
        smooth_feat[col_smooth_rate] = -1
        # print((pd.merge(df[df.day == day], CVR[[col, col_smooth_rate]], on=col, how='inner')).columns[-1])
        smooth_feat[col_smooth_rate] = (pd.merge(df[df.day == day], CVR[[col, col_smooth_rate]], on=col, how='left')).iloc[:, -1].values

    smooth_all = pd.concat([smooth_all, smooth_feat], axis=1)
dump_pickle(smooth_all, path=raw_data_path+'102_smooth_features.pkl')





