from glob import glob
from utils import BayesianSmoothing, load_pickle, dump_pickle, raw_data_path
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale

def read_pickles(path, col=None):
    if col is None:
        df = pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*.pkl')))])
    else:
        df = pd.concat([pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*.pkl')))])
    return df

def data_split():
    train = load_pickle('../data/train.pkl')
    test = load_pickle('../data/test.pkl')

    train_feats = read_pickles('../data/train/')
    test_feats = read_pickles('../data/test')

    train = train.merge(train_feats, on='instance_id', how='left')
    test = test.merge(test_feats, on='instance_id', how='left')

    drop_columns = ['time', 'realtime']
    train.drop(drop_columns, axis=1, inplace=True)
    test.drop(drop_columns, axis=1, inplace=True)


    train_df = train[train.day<24].copy()
    #train_df = handle_imbalance(train_df)
    valid_df = train[train.day==24].copy
    test_df = test

    dump_pickle(train_df, path='../data/train_feature/train_final.pkl')
    dump_pickle(valid_df, path='../data/valid_feature/valid_final.pkl')
    dump_pickle(test_df, path='../data/test_feature/test_final.pkl')

def data_onehot():

    train_data = load_pickle(path='../data/train_feature/train_final.pkl')
    cv_data = load_pickle(path='../data/valid_feature/valid_final.pkl')
    test_data = load_pickle(path='../data/test_feature/test_final.pkl')

    cols = ['user_gender_id', 'user_age_level', 'user_occupation_id'
        , 'second_cate', 'item_city_id', 'item_price_level'
        , 'context_page_id', 'shop_review_num_level']

    data = pd.concat([train_data, cv_data, test_data], axis=0)

    for col in cols:
        col_feature = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, col_feature], axis=1)

    X = minmax_scale(data.values)
    data = pd.DataFrame(data=X, columns=data.columns)

    train_data = data.loc[train_data.index]
    cv_data = data.loc[cv_data.index]
    test_data = data.loc[test_data.index]

    train_data.reset_index(inplace=True, drop=True)
    cv_data.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)

    dump_pickle(train_data, path='../data/train_feature/train_final_onehot.pkl')
    dump_pickle(cv_data, path='../data/valid_feature/valid_final_onehot.pkl')
    dump_pickle(test_data, path='../data/test_feature/test_final_onehot.pkl')






