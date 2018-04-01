import pickle
import pandas as pd
import numpy as np
import scipy.stats as sps
from tqdm import tqdm
import math
import random

#file_path
raw_data_path = '../data/'
result_path = '../result/'
model_path = '../model/'

def load_pickle(path):
    with open(path,'rb') as f_t:
        return pickle.load(f_t)
def dump_pickle(obj, path, protocol=None,):
    with open(path,'wb') as f_t:
        return pickle.dump(obj,f_t,protocol=protocol)
