from glob import glob
from utils import BayesianSmoothing, load_pickle, dump_pickle, raw_data_path
import pandas as pd
from tqdm import tqdm

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

    dump_pickle(train_df, path='../data/train/train_final.pkl')
    dump_pickle(valid_df, path='../data/valid/valid_final.pkl')
    dump_pickle(test_df, path='../data/test/test_final.pkl')

def data_onehot():





