import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

train = pd.read_pickle('../data/train.pkl')
test = pd.read_pickle('../data/test.pkl')
df = pd.concat([train, test])


user_cols = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
id_cols = ['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'shop_id']

# grade_cols = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'shop_review_num_level',
#               'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']
grade_cols = ['item_price_level']

# ============================= count features for id features =====================
for user_col in tqdm(user_cols):
    cnt_result = df.groupby(user_col)[id_cols].nunique()
    cnt_result = cnt_result.add_prefix(user_col + '_').add_suffix('_cnt')
    cnt_result = cnt_result.reset_index()
    df = pd.merge(df, cnt_result, on=user_col, how='left')

# ============================ statistic features for grade features ========================
def get_stats_target(df, group_column, target_column, drop_raw_col=False):
    df_old = df.copy()
    grouped = df_old.groupby(group_column)
    the_stats = grouped[target_column].agg(['mean', 'median', 'max', 'min', 'std', 'skew']).reset_index()
    the_stats.columns = [ group_column,
                            '_%s_groupby_%s_mean' % (target_column, group_column),
                            '_%s_groupby_%s_median' % (target_column, group_column),
                            '_%s_groupby_%s__max' % (target_column, group_column),
                            '_%s_groupby_%s__min' % (target_column, group_column),
                            '_%s_groupby_%s__std' % (target_column, group_column),
                            '_%s_groupby_%s__skew' % (target_column, group_column)
                        ]
    if drop_raw_col:
        df_old.drop(group_column, axis=1, inplace=True)

    return the_stats

for user_col in tqdm(user_cols):
    # statistic feature
    stats_list = []
    for grade_col in tqdm(grade_cols):
        the_stats = get_stats_target(df, user_col, grade_col, drop_raw_col=False)
        stats_list.append(the_stats)

    df = pd.merge(df, stats_list, on=user_col)


# ========================== count features for category features =========================
category_df = df[['user_id'] + ['item_category_list{}'.format(i) for i in range(1, 3)]]
for user_col in tqdm(user_cols):
    # category and property columns
    category_cnt = category_df.groupby('user_id')['item_category_list1'].nunique().to_frame()
    category_cnt.columns = ['{category_groupby_{}_cnt' % user_col]

    df = pd.merge([df, category_cnt], how='left', on='user_id')

    del category_cnt
    gc.collect()

# ======================== count features for instances ===================================
for user_col in tqdm(user_cols):
    if user_col == 'user_id':
        continue
    else:
        tmp_groupby_cnt = category_df.groupby(user_col)['user_id'].nunique().reset_index()
        tmp_groupby_cnt.columns = [user_col, 'id_groupby_{}_cnt'.format(user_col)]
        df = pd.merge([df, tmp_groupby_cnt], how='left', on=user_col)











