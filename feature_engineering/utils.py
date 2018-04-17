import pickle
import pandas as pd
import numpy as np
import scipy.stats as sps
from tqdm import tqdm
import math
import random
import scipy.special as special


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

class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        # 产生样例数据
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            # imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return pd.Series(I), pd.Series(C)

    def update(self, tries, success, iter_num, epsilon):
        # 更新策略
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        # 迭代函数
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        sumfenzialpha = (special.digamma(success + alpha) - special.digamma(alpha)).sum()
        sumfenzibeta = (special.digamma(tries - success + beta) - special.digamma(beta)).sum()
        sumfenmu = (special.digamma(tries + alpha + beta) - special.digamma(alpha + beta)).sum()

        return alpha * (sumfenzialpha / sumfenmu), beta * (sumfenzibeta / sumfenmu)
    
def cal_log_loss(predict_list, valid_list):
    if len(predict_list) != len(valid_list):
        return -1
    loss = 0
    for predict_label, valid_label in zip(predict_list, valid_list):
        if predict_label <= 0:
            predict_label = 0.00000000001
        if predict_label >= 1:
            predict_label = 0.99999999999
        loss += (valid_label*math.log(predict_label)+(1-valid_label)*math.log(1-predict_label))
    return -loss/(len(predict_list))


