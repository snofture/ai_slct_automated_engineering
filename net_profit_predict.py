# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 09:41:04 2017

@author: limen
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
#import matplotlib.pyplot as plt
#from sklearn import metrics
import re
import os
import sys
import yaml
#sys.setdefaultencoding('utf-8')


#build MAPE function
def mean_absolute_percentage_error(y,p):   
    return np.mean(np.abs((y-p)/y))


#标准品牌映射
def jd_tmall_brand_mapping(a):
    jd_tmall_brand_mapping = params['worker']['dir']+'/input/'+params['EndDate']+'/'+ params['item_third_cate_cd']+'/jd_tmall_brand_mapping'
    brand_map = pd.read_table(jd_tmall_brand_mapping, header=0, sep='\t',names=['jd_std_brand','attr_value'],encoding='utf-8')
    tmall_brand = a[(a['attr_name']==u'品牌') & (a['web_id']==2)]   
    jd_brand = a[~a.index.isin(tmall_brand.index)]
    new_tmall_brand = pd.merge(tmall_brand,brand_map,on='attr_value',how='left')
    func = lambda x: x['attr_value'] if pd.isnull(x['jd_std_brand']) else x['jd_std_brand']
    new_tmall_brand['attr_value'] = new_tmall_brand.apply(func, axis =1)
    new_tmall_brand = new_tmall_brand.drop('jd_std_brand', axis = 1)
    a =pd.concat([jd_brand,new_tmall_brand],axis=0).drop_duplicates()
    return a


#删除缺失超过半数的样本数据
def del_rows_missing_values(att):
    number_of_columns = att.shape[1]
    t = int((number_of_columns-1)/2) +2
    drop_null_att = att.dropna(thresh = t)
    return drop_null_att


#fill nan with the most frequent one if the missing entries are less than 25% of the column
def fill_missing_values(a_web):
    for col in a_web.columns.difference(['sku_id','web_id']):
        if (np.asscalar(np.int16(pd.isnull(a_web[col]).sum())) / len(a_web[col])) * 100 < 25:
            a_web[col] = a_web[col].fillna(a_web[col].value_counts().index[0])
        else:
            a_web[col] = a_web[col].fillna(u'其它')
    return a_web


#cell信息提取
def value_selection(a_web):
        #handle abnormal value
        for col in a_web.columns.difference(['sku_id','web_id']):
            a_web[col] = a_web[col].apply(lambda x:x.replace(u'其他',u'其它'))
        
        #regular expression, select the last part      
        for col in a_web.columns.difference(['sku_id','web_id']):
            a_web[col] = a_web[col].apply(lambda x: re.sub('.*#\$','',x))
        
        #phrase split, select the last part
        for col in a_web.columns.difference(['sku_id','web_id']):
            a_web[col] = a_web[col].apply(lambda x:x.split('/')[0])
        
        return a_web


#仅使用有两条销售记录以上的价格数据
def fil_price_without_recods(gdm_m04_ord_det_sum):
    sale_count = gdm_m04_ord_det_sum.groupby(['sku_id']).agg({'sale_ord_tm':'count'})
    sale_count = sale_count.reset_index()
    sale_count['sku_id'] = sale_count['sku_id'].apply(lambda x: str(x))
    sale_count['count'] = sale_count['sale_ord_tm']
    sale_count.drop('sale_ord_tm', axis = 1, inplace = True)
   
    #use only the sku with sale records greater than 3
    valid_sale_count = sale_count[sale_count['count'] > 2]
    valid_sku_id = list(valid_sale_count['sku_id']) 
    gdm_m04_ord_det_sum = gdm_m04_ord_det_sum[gdm_m04_ord_det_sum['sku_id'].isin(valid_sku_id)]
    gdm_m04_ord_det_sum = gdm_m04_ord_det_sum.drop_duplicates()
    return gdm_m04_ord_det_sum


#处理类别特征和数值特征
def label_encoding_and_mean_normalization(jd_pop):
    #label encoder method to handle discrete/categorical features except continuous features
    le = preprocessing.LabelEncoder()
    for attribute in jd_pop.columns.difference(['mean_price','sku_id','web_id']):
        jd_pop[attribute] = le.fit_transform(jd_pop[attribute])
        
    #normalize continuous features('mean_price')
    jd_pop['mean_price'] = jd_pop['mean_price'].apply(lambda x: 
        (x-jd_pop['mean_price'].mean())/(jd_pop['mean_price'].std()))
    
    return jd_pop


#自营sku利润率
def cal_self_sku_net_profit(net_profit):
    #extract only self sku profit
    self_sku_net_profit = net_profit.groupby('item_sku_id').agg({'net_profit':'mean'})
    self_sku_net_profit = self_sku_net_profit.reset_index()
    self_sku_net_profit['item_sku_id'] =self_sku_net_profit['item_sku_id'].apply(lambda x: int(x))
    self_sku_net_profit['item_sku_id'] =self_sku_net_profit['item_sku_id'].apply(lambda x: str(x))
    return self_sku_net_profit


#训练模型
def train_model(final_net_profit):
    from sklearn.model_selection import train_test_split
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(final_net_profit.drop('net_profit',
                                                                                axis=1), 
                                                        final_net_profit['net_profit'], 
                                                        test_size=0.30,
                                                        random_state = 101)
    from hyperopt import fmin, tpe, hp, partial
    from sklearn.ensemble import RandomForestRegressor
    def objective(args):
        rfr = RandomForestRegressor(  n_estimators = int(args['n_estimators']), 
                                      max_features = 'auto',
                                      max_depth = int(args['max_depth']),
                                      min_samples_leaf = int(args['min_samples_leaf']),
                                      min_samples_split = int(args['min_samples_split']),
                                      oob_score=True,
                                      n_jobs=-1,
                                      bootstrap = True)
        rfr.fit(X_train, y_train)
        predictions = rfr.predict(X_test)
        return mean_absolute_percentage_error(y_test, predictions)    
    space = {'n_estimators':hp.quniform('n_estimators',10, 500,1),
             'max_depth':hp.quniform('max_depth',1,10,1),
             'min_samples_leaf':hp.quniform('min_samples_leaf', 1, 4,1),
             'min_samples_split':hp.quniform('min_samples_split',2, 8,1)}
    
    algo = partial(tpe.suggest,n_startup_jobs=10)
    best = fmin(objective,space,algo = algo,max_evals=50)
    
    
    best['n_estimators'] = int(best['n_estimators'])
    best['max_depth'] = int(best['max_depth'])
    best['min_samples_leaf'] = int(best['min_samples_leaf'])
    best['min_samples_split'] = int(best['min_samples_split'])
    print (best)
    print objective(best)
    rf = RandomForestRegressor(**best)
    rf.fit(X_train,y_train)

    return rf


def main(params):
    #import jd attributes table
    app_ai_slct_attributes = params['worker']['dir']+'/input/'+params['EndDate']+'/'+ params['item_third_cate_cd']+'/app_ai_slct_attributes'
    a =  pd.read_table(app_ai_slct_attributes,sep = '\t', encoding = 'utf-8')
    a = a[['sku_id','attr_name','attr_value','web_id']]
    a = a.drop_duplicates()
    
    web_id = a[['sku_id','web_id']]
    web_id = web_id.drop_duplicates()
    web_id['sku_id'] = web_id['sku_id'].apply(lambda x: str(x))
        
    attrs = jd_tmall_brand_mapping(a)        
        
    #transform original table to pivot_table 
    att = pd.pivot_table(attrs, index=['sku_id'], columns=['attr_name'],
                        values=['attr_value'],fill_value = np.nan, aggfunc='max')
    att.columns = att.columns.droplevel(level=0)
    att = att.reset_index(drop=False)
    att = att.drop_duplicates()
    att['sku_id'] = att['sku_id'].apply(lambda x: str(x))
    
    
    drop_null_att = del_rows_missing_values(att)
    
    #to add web_id information
    a_web = pd.merge(drop_null_att,web_id,how='inner',on='sku_id')
    a_web = a_web.drop_duplicates()
    a_web['sku_id'] = a_web['sku_id'].apply(lambda x: str(x))
    
    #fill nan for brand with u'其它'
    a_web[u'品牌'] = a_web[u'品牌'].fillna(u'缺失')
       
    a_web = fill_missing_values(a_web)
    
    a_web = value_selection(a_web)    
        
    #import sku_price table
    gdm_m04_ord_det_sum = params['worker']['dir']+'/input/'+params['EndDate']+ '/'+params['item_third_cate_cd']+'/gdm_m04_ord_det_sum'
    gdm_m04_ord_det_sum = pd.read_table(gdm_m04_ord_det_sum, sep='\t',encoding='utf-8')
    gdm_m04_ord_det_sum['sku_id'] = gdm_m04_ord_det_sum['item_sku_id']
    gdm_m04_ord_det_sum['sku_id'] = gdm_m04_ord_det_sum['sku_id'].apply(lambda x: str(x))

    
    gdm_m04_ord_det_sum = fil_price_without_recods(gdm_m04_ord_det_sum)
    
    #calculate the mean_price
    gdm_m04_ord_det_sum = gdm_m04_ord_det_sum.groupby(['sku_id']).agg({'before_prefr_amount':'sum','sale_qtty':'sum'})
    gdm_m04_ord_det_sum = gdm_m04_ord_det_sum.reset_index()
    gdm_m04_ord_det_sum['sku_id'] = gdm_m04_ord_det_sum['sku_id'].apply(lambda x: str(x))
    gdm_m04_ord_det_sum['mean_price'] = gdm_m04_ord_det_sum['before_prefr_amount']/gdm_m04_ord_det_sum['sale_qtty']
    gdm_m04_ord_det_sum.drop(['before_prefr_amount','sale_qtty'], axis = 1, inplace = True)


    #import tmall sku_price
    tm_sku_price = params['worker']['dir']+'/input/'+params['EndDate']+ '/'+params['item_third_cate_cd']+'/tm_sku_price'
    tm_sku_price =  pd.read_table(tm_sku_price, sep='\t',encoding='utf-8')
    tm_sku_price['sku_id'] = tm_sku_price['sku']
    tm_sku_price.drop(['sku'], axis = 1, inplace = True)
    tm_sku_price['sku_id'] = tm_sku_price['sku_id'].apply(lambda x: str(x)) 
    
    
    #concat jd_pop and tmall price
    all_price = pd.concat([gdm_m04_ord_det_sum, tm_sku_price], axis = 0).drop_duplicates()
    
    #merge jd_pop_attrs table with sku_price table based on sku_id 
    jd_pop = pd.merge(a_web, all_price, how = 'inner', on = 'sku_id')
    jd_pop = jd_pop.drop_duplicates()
    jd_pop['mean_price'] = jd_pop['mean_price'].apply(lambda x: int(x))
    jd_pop['sku_id'] = jd_pop['sku_id'].apply(lambda x: str(x))


    jd_pop = label_encoding_and_mean_normalization(jd_pop)
    
    jd_pop['sku_id'] = jd_pop['sku_id'].apply(lambda x: str(x))    

        
    jd = jd_pop[jd_pop['web_id']==0]
    pop_tmall = jd_pop[jd_pop['web_id'] != 0]


    #import profit table
    app_cfo_profit_loss_b2c_det = params['worker']['dir']+'/input/'+params['EndDate']+ '/'+params['item_third_cate_cd']+'/app_cfo_profit_loss_b2c_det'
    net_profit = pd.read_table(app_cfo_profit_loss_b2c_det, sep = '\t', encoding = 'utf-8')


    net_profit.drop(['dt','gmv','item_third_cate_name','cost_tax','income','grossfit','gross_sales','rebate_amunt_notax','adv_amount','store_fee','deliver_fee'], axis = 1, inplace = True)
    net_profit['item_sku_id'] = net_profit['item_sku_id'].apply(lambda x:str(x))
    
    
    #extract self_sku_profit rate 
    self_sku_net_profit = cal_self_sku_net_profit(net_profit)


    #extract the mean sku_id profit table
    average_net_profit = net_profit.groupby('item_sku_id').agg({'net_profit':'mean'})
    average_net_profit.reset_index(inplace=True)
    average_net_profit['item_sku_id'] = average_net_profit['item_sku_id'].apply(lambda x: int(x))
    average_net_profit['item_sku_id'] = average_net_profit['item_sku_id'].apply(lambda x: str(x))
    average_net_profit= average_net_profit.rename(columns={'item_sku_id':'sku_id'})
    
    
    #merge attributes table and mean profit table based on sku_id
    final_net_profit = pd.merge(jd,average_net_profit, how = 'inner', on = 'sku_id')
    final_net_profit = final_net_profit.drop_duplicates()
    final_net_profit.drop(['sku_id','web_id'],axis = 1, inplace = True)

    
    #implement the profit_prediction algorihtm on pop_tmall skus          
    pop_tmall_sku_id = pop_tmall['sku_id']
    pop_tmall_sku_id = pop_tmall_sku_id.reset_index()
    pop_tmall_sku_id.drop('index',axis=1,inplace=True)
    pop_tmall_sku_id = pop_tmall_sku_id.rename(columns={'sku_id':'item_sku_id'})
    pop_tmall_sku_id = pop_tmall_sku_id.drop_duplicates()
    
    
    pop_tmall.drop(['sku_id','web_id'], axis = 1, inplace = True)
    pop_tmall_predictions = train_model(final_net_profit).predict(pop_tmall)


    #self_predictions = train_model(final_net_profit).predict(X_test)

    
    profit_tmall_predict = pd.DataFrame({'net_profit': pop_tmall_predictions.tolist()})
    net_profit_predict = pd.concat([pop_tmall_sku_id, profit_tmall_predict],axis = 1, ignore_index = True)
    net_profit_predict.columns = ['item_sku_id','net_profit']
    net_profit_predict['item_sku_id'] = net_profit_predict['item_sku_id'].apply(lambda x: str(x))
    
    
    
    #concat self sku profit and predicted pop/tmall sku profit
    net_profit_predict = pd.concat([net_profit_predict,self_sku_net_profit],axis=0)
    
    
    
    #filter result sku
    app_ai_slct_sku = params['worker']['dir']+'/input/'+params['EndDate']+ '/'+params['item_third_cate_cd']+'/app_ai_slct_sku'
    circle = pd.read_table(app_ai_slct_sku, sep='\t',encoding='utf-8',dtype = {'sku_id':str})
    circle = circle.rename(columns = {'sku_id':'item_sku_id'})['item_sku_id'].drop_duplicates()
    circle = circle.to_frame()
    all_net_profit_predict=pd.merge(net_profit_predict,circle,on='item_sku_id',how='inner')

    

    #save to file
    out_path = params['worker']['dir']+'/output/'+params['EndDate']+'/'+ params['scope_id']
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    all_net_profit_predict.to_csv(out_path+'/all_net_profit_predict',header='infer',sep='\t',encoding='utf-8',index=False)


if __name__ == '__main__':
    #read command line arguments
    n = len(sys.argv) - 1
    if n < 1:
        print('Usage:\n    pyhton profit_rate_predict.py param_file\n')
        sys.exit()
    else:
        param_file = sys.argv[1]
        print('[INFO] net_profit_predict started')
    
    #read parameters
    params = yaml.load(file('params/default.yaml','r'))
    user_params = yaml.load(file(param_file,'r'))
    for key, value in user_params.items():
        params[key] = value
    
    main(params)
    print('[INFO] net_profit_predict completed')


















































































