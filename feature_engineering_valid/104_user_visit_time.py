from utils import BayesianSmoothing, load_pickle, dump_pickle, raw_data_path
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

df = load_pickle('../data/df.pkl')
train = load_pickle('../data/train.pkl')
test = load_pickle('../data/test.pkl')


def user_visit_time(df, mode):
    final_feat = pd.DataFrame({'instance_id': df.instance_id, 'user_id': df.user_id})

    # 将时间离散成早中晚，并且计算早中晚的浏览总次数
    def time_discrete(hour):
        if 0 < hour <= 6:
            return 'midnight'
        elif 6 < hour <= 11:
            return 'morning'
        elif 11 < hour <= 15:
            return 'noon'
        elif 15 < hour <= 18:
            return 'afternoon'
        elif 18 < hour <= 24:
            return 'evening'


    final_feat['time_discrete'] = df['hour'].map(time_discrete)
    time_discrete_visit_count = pd.crosstab(index=final_feat['user_id'], columns=final_feat['time_discrete']).add_suffix('_visit_count').reset_index()
    final_feat = pd.merge(final_feat, time_discrete_visit_count, on='user_id', how='left')
    dump_pickle(final_feat, path=raw_data_path + '{}_feature/'.format(mode) +'104_user_visit_time.pkl')

user_visit_time(train, 'train')
user_visit_time(test, 'test')
