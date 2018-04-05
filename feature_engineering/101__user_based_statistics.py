import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

train = pd.read_pickle('../data/train.pkl')
test = pd.read_pickle('../data/test.pkl')
df = pd.concat([train, test])

# ============================ statistic feature ============================
user_cols = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
id_cols = ['instance_id', 'item_id', 'user_id', 'context_id', 'shop_id']

grade_cols = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'shop_review_num_level'
              'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']

# count features
for user_col in user_cols:
    for id_col in id_cols:
        df['%s_groupby_%s_cnt' % (user_col, id_cols)] = df.groupby(user_col)[id_cols].nunique()

# statistic feature

def get_stats_target(df, group_column, target_column, drop_raw_col=False):
    df_old = df.copy()
    grouped = df_old.groupby(group_column)
    the_stats = grouped[target_column].agg(['mean', 'median', 'max', 'min', 'std', 'skew']).reset_index()
    the_stats.columns = [ group_column,
                            '__%s__groupby__%s__mean' % (target_column, group_column),
                            '__%s__groupby__%s__median' % (target_column, group_column),
                            '__%s__groupby__%s__max' % (target_column, group_column),
                            '__%s__groupby__%s__min' % (target_column, group_column),
                            '__%s__groupby__%s__std' % (target_column, group_column),
                            '__%s__groupby__%s__skew' % (target_column, group_column)
                        ]
    df_old = pd.merge(df_old, the_stats, on=group_column)
    if drop_raw_col:
        df_old.drop(group_column, axis=1, inplace=True)
    return df_old

category_df = df[['user_id']+['item_category_list{}'.format(i) for i in range(1, 3)]]

for user_col in tqdm(user_cols):
    for grade_col in grade_cols:
        df = get_stats_target(df, user_col, grade_col, drop_raw_col=False)

# category and property columns
    category_cnt = category_df.groupby('user_id')['item_category_list1'].nunique().to_frame()
    category_cnt.columns = ['{category_groupby_{}_cnt' % user_col]










