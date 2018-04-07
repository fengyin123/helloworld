# 计算出至今为止用户买的产品数
from utils import BayesianSmoothing, load_pickle, dump_pickle, raw_data_path
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

df = load_pickle('../data/df.pkl')

data = df.copy()


# ========================= 当天搜索次数和当前小时搜索次数 =========================================
time_features = ['hour', 'day']
user_features = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']

user_time_df = pd.DataFrame()
user_time_df['instance_id'] = data['instance_id']

for user_feature in tqdm(user_features):
    for time_feature in time_features:
        search_group = data.groupby([user_feature, time_feature]).count().reset_index()
        tmp_df = data[[user_feature, time_feature]]
        user_day_search = pd.merge(tmp_df, search_group, on=[user_feature, time_feature], how='left').iloc[:, -1]
        user_time_df['{}_{}_search'.format(user_feature, time_feature)] = user_day_search

dump_pickle(user_time_df, path=raw_data_path+'103_user_time_count.pkl')


# ======================== 用户当前搜索距离上次的时间 ================================================
df_tmp = df[['instance_id', 'user_id',  'context_timestamp']].copy()
df_tmp.sort_values(['instance_id', 'user_id', 'context_timestamp'], inplace=True)

df_tmp['t-1_context_timestamp'] = df_tmp.groupby('user_id')['context_timestamp'].shift(1)
df_tmp['time_diff_last_query'] = np.log1p(df_tmp['context_timestamp'] - df_tmp['t-1_context_timestamp'])

final_feat = df_tmp[['instance_id', 'time_diff_last_query']]

dump_pickle(final_feat, path=raw_data_path+'103_feature_last_query.pkl')


# ========================= 当日用户当前搜索距离上次的时间(商品，商店，商标) ==========================================
final_feat = pd.DataFrame()
final_feat['instance_id'] = df['instance_id']
cols = ['item_id', 'shop_id', 'item_brand_id']
for col in tqdm(cols):
    df_select = df[['user_id', col,'day','context_timestamp']]
    df_group =  df_select.groupby(['user_id', col,'day'])
    group_max = df_group['context_timestamp'].transform('max')
    group_min = df_group['context_timestamp'].transform('min')
    print(len(group_max))
    final_feat['diff_maxtime_{}'.format(col)] = df['context_timestamp'].values - group_max
    final_feat['diff_mintime_{}'.format(col)] = df['context_timestamp'].values -group_max

dump_pickle(final_feat, path=raw_data_path + '103_diff_max_min.pkl')




