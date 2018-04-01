#encoding:utf-8
import time
import pandas as pd

def __convert_timestamp_to_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df

def convert_data(df):
    """
    按hour和day分离时间特征
    构建 user_query_day和user_query_day_hour特征
    :param data:
    :return:
    """
    # 1. 类目列表特征
    listItem = ['item_category_list', 'item_property_list'] #, 'predict_category_property']

    # 2. 类别特征
    singleIntItem = ['item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
                     'item_brand_id']
    singleIntUser = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    singleIntContext = ['context_page_id']
    singleIntShop = ['shop_review_num_level', 'shop_star_level' , ]
    singleIntFeature = singleIntItem + singleIntUser + singleIntContext + singleIntShop

    # 3. 连续型特征
    singleDoubleShop = ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery',
                        'shop_score_description']
    singleDoubleShopDispersed = ['shop_review_positive_rate_dispersed', 'shop_score_service_dispersed',
                                 'shop_score_delivery_dispersed', 'shop_score_description_dispersed']

    # 4. ID列表
    idList = ['instance_id', 'item_id', 'user_id', 'context_id', 'shop_id']

    # 5. 目前还未用到的特征
    unsureList = ['context_timestamp']

    # 5 train label标记
    label = ['is_trade', 'is_train']


    featureSum = listItem + singleIntFeature + singleDoubleShop + idList + unsureList + label

    df = df[featureSum].copy()
    print("========>  Start 预处理")
    print("========>  item_category_list 广告商品的的类目列表，String类型；从根类目（最粗略的一级类目）向叶子类目（最精细的类目）依次排列")
    for i in range(3):
        df['category_%d' % (i)] = df['item_category_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else " "
        )
        df = one_hot(df,['category_%d' % (i)] )
        df = df.drop('category_%d' % (i), axis=1)
    del df['item_category_list']

    print('========>  item_property_list 广告商品的属性列表，String类型；各个属性没有从属关系; 数据拼接格式为 "property_0;property_1;property_2"')
    for i in range(3):
        df['property_%d' % (i)] = df['item_property_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else " "
        )
        df = one_hot(df, ['property_%d' % (i)])
        df = df.drop('property_%d' % (i), axis=1)
    del df['item_property_list']

    """
    print('========>  predict_category_property_ing 根据查询词预测的类目属性列表，String类型；')
    for i in range(3):
        df['predict_category_%d' % (i)] = df['predict_category_property'].apply(
            lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else " "
        )
        df = one_hot(df, ['predict_category_%d' % (i)])
        df = df.drop('predict_category_%d' % (i), axis=1)
    del df['predict_category_property']
    """

    print('========>  time处理')
    df['time'] = df.context_timestamp.apply(__convert_timestamp_to_datetime)
    df['day'] = df.time.apply(lambda x: int(x[8:10]))
    df['hour'] = df.time.apply(lambda x: int(x[11:13]))
    del df['context_timestamp']
    del df['time']

    user_query_day = df.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
    df = pd.merge(df, user_query_day, 'left', on=['user_id', 'day'])

    user_query_day_hour = df.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    df = pd.merge(df, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])

    print("========>  Convert Success!")

    return df


def load_data():
    path = '../data/'

    # 训练集
    # train = pd.read_table(path+'round1_ijcai_18_train_20180301.txt',encoding='utf8',delim_whitespace=True)
    train = pd.read_table(path + 'train.data', encoding='utf8', delim_whitespace=True)
    train['is_train'] = 1
    train = train.dropna()

    # 测试集
    # test = pd.read_table(path+'round1_ijcai_18_test_a_20180301.txt',encoding='utf8',delim_whitespace=True)
    test = pd.read_table(path + 'test.data', encoding='utf8', delim_whitespace=True)
    test['is_train'] = 0

    # 连接
    df = pd.concat([train, test])
    print("========> Load Data Success!")
    return df


if __name__=="__main__":
    df = load_data()
    df = convert_data(df)
    # print(df.columns)
