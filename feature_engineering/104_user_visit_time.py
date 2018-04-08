# 计算出至今为止用户买的产品数
from utils import BayesianSmoothing, load_pickle, dump_pickle, raw_data_path
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

df = load_pickle('../data/df.pkl')
final_feat = pd.DataFrame({'instance_id': df.instance_id, 'user_id': df.user_id})


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

time_discrete_visit_count = pd.crosstab(index=final_feat['user_id'], columns=final_feat['time_discrete']).add_suffix('visit_count')
time_discrete_trade_count = pd.crosstab(index=final_feat['user_id'], columns=final_feat['time_discrete']).add_suffix('visit_count')
print(time_discrete_count)