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

train = load_pickle('../data/train.pkl')
test = load_pickle('../data/test.pkl')

train_feats = read_pickles('../data/train/')
test_feats = read_pickles('../data/test')

train = train.merge(train_feats, on='instance_id', how='left')
test = test.merge(test_feats, on='instance_id', how='left')

drop_columns = ['time', 'realtime']
train.drop(drop_columns, axis=1, inplace=True)
test.drop(drop_columns, axis=1, inplace=True)

def train_valid_split(data):
    train_df = data[data.day<24].copy()
    train_df = handle_imbalance(train_df)
    valid_df = data[data.day==24].copy

    dump_pickle(train_df, path='train_final.pkl')
    dump_pickle(valid_df, path='')



