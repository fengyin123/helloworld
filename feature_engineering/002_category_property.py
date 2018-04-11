#encoding:utf-8
import gc
import numpy as np
import pandas as pd
from utils import BayesianSmoothing, load_pickle, dump_pickle, raw_data_path
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

path = '../data/'
train_file = 'round1_ijcai_18_train_20180301.txt'
test_file = 'round1_ijcai_18_test_a_20180301.txt'

train = pd.read_table(path + train_file, encoding='utf8', delim_whitespace=True)
test = pd.read_table(path + test_file, encoding='utf8', delim_whitespace=True)
len_train = train.shape[0]
df = pd.concat([train, test], axis=0, ignore_index=True)

corpus = df.item_property_list.values.astype('U').tolist()
vectorizer = CountVectorizer()
vectorizer.fit(corpus)
countvector = vectorizer.transform(df.item_property_list)

pror_count = np.array(countvector.sum(axis=0))[0]
count = pd.Series(data=pror_count, index=np.arange(0, len(pror_count)))
selected_index = list(count.sort_values(ascending=False)[:100].index)
column_name = np.array(vectorizer.get_feature_names())[selected_index]
column_value = countvector[:, selected_index]
final_feat = pd.DataFrame(column_value.toarray(), columns=column_name)
final_feat['instance_id'] = df['instance_id']

train_feat = final_feat.iloc[:len(train), :]
test_feat = final_feat.iloc[len(train):, :]

dump_pickle(train_feat, path=raw_data_path + 'train_feature/' + '002_property_count.pkl')
dump_pickle(test_feat, path=raw_data_path + 'test_feature/' + '002_property_count.pkl')
