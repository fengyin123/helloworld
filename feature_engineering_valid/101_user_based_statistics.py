# %load 101_user_based_statistics.py
#encoding:utf-8
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import BayesianSmoothing, load_pickle, dump_pickle, raw_data_path

train = pd.read_pickle('../data/train.pkl')
test = pd.read_pickle('../data/test.pkl')
df = pd.read_pickle('../data/df.pkl')

train_feat = pd.DataFrame({'instance_id': train.instance_id})
test_feat = pd.DataFrame({'instance_id': test.instance_id})

user_cols = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
id_cols = ['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'shop_id']
df_cols = df.columns
# grade_cols = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'shop_review_num_level',
#               'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']
grade_cols = ['item_price_level']

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

def generate_basic_feats(df, feat_df):

    # ============================= 计算关于用户特征的id类特征的数量 =====================
    '''
    如：对于每个男性用户, 他所浏览的各种item_id的数量
    '''
    for user_col in tqdm(user_cols):
        cnt_result = df.groupby(user_col)[id_cols].nunique()
        cnt_result = cnt_result.add_prefix(user_col + '_').add_suffix('_cnt')
        cnt_result = cnt_result.reset_index()
        feat_df[user_col + '_count'] = pd.merge(df, cnt_result, on=user_col, how='left').iloc[:, -1].values

    # ============================ 计算关于用户特征的得分类特征的统计特征 ========================
    '''
    如：男性用户中item_price_level的平均数，中位数，最大值，最小值
    '''
    for user_col in tqdm(user_cols):
        # statistic feature
        for grade_col in tqdm(grade_cols):
            the_stats = get_stats_target(df, user_col, grade_col, drop_raw_col=False)
            #stats_list = pd.concat([stats_list, the_stats], axis=1)
            tmp_df = pd.merge(df, the_stats, on=user_col)
            feat_df = pd.concat([feat_df, tmp_df.drop(df_cols, axis=1)] , axis=1)

    # ========================== 计算每个用户下的category2的个数 =========================
    '''
    如：对于每个男性用户，他所浏览的各种category2的数量
    '''
    for user_col in tqdm(user_cols):
        # category and property columns
        category_cnt = df.groupby(user_col)['item_category_list2'].nunique().reset_index()
        category_cnt.columns = [user_col, 'category2_groupby_{}_cnt'.format(user_col)]
        tmp_df = pd.merge(df, category_cnt, how='left', on=user_col)
        feat_df = pd.concat([feat_df, tmp_df.drop(df_cols, axis=1)], axis=1)

        del category_cnt
        gc.collect()

    return feat_df
    # # ======================== 计算每个用户特征下分别有多少用户 ===================================
    # '''
    #     如：对于每个男性用户，他所浏览的各种category2的数量
    # '''
    # for user_col in tqdm(user_cols):
    #     if user_col == 'user_id':
    #         continue
    #     else:
    #         tmp_groupby_cnt = category_df.groupby(user_col)['user_id'].nunique().reset_index()
    #         tmp_groupby_cnt.columns = [user_col, 'id_groupby_{}_cnt'.format(user_col)]
    #         df = pd.merge([df, tmp_groupby_cnt], how='left', on=user_col)

train_feat = generate_basic_feats(train, train_feat)
test_feat = generate_basic_feats(test, test_feat)

dump_pickle(train, raw_data_path + 'train_feature/' + '101_user_based_statistics.pkl')
dump_pickle(test, raw_data_path + 'test_feature/' + '101_user_based_statistics.pkl')
