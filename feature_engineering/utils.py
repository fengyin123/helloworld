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

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clicks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self._fixed_point_iter(imps, clicks, self.alpha, self.beta)
            if(i%50==0):
                print('iter_num:', i)
                print('difference of alpha is {}'.format(new_alpha - self.alpha))
                print('difference of beta is {}'.format(new_beta - self.beta))
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def _fixed_point_iter(self, imps, clicks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0
        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clicks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clicks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))


        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)
    
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


