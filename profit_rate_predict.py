# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 15:42:10 2017

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
sys.setdefaultencoding('utf-8')


#build MAPE function
def mean_absolute_percentage_error(y,p):   
    return np.mean(np.abs((y-p)/y))

def main(params):
    #import jd attributes table
    app_ai_slct_attributes = params['worker']['dir']+'/input/'+params['EndDate']+\ +'/'
    + params['item_third_cate_cd']+'/app_ai_slct_attributes'
    attrs =  pd.read_table('app_ai_slct_attributes',sep = '\t', encoding = 'utf-8')
    attrs0 = attrs[attrs['web_id'] == 0] 
    attrs1 = attrs[attrs['web_id'] == 1] 
    attrs = pd.concat([attrs0,attrs1],ignore_index=True)
    attrs = attrs[['sku_id','attr_name','attr_value','web_id']]
    attrs['sku_id'] = attrs['sku_id'].apply(lambda x: int(x))
    web_id = attrs[['sku_id','web_id']]
    web_id['sku_id'] = web_id['sku_id'].apply(lambda x: int(x))
    web_id = web_id.drop_duplicates()
    
    
    #transform original table to pivot_table 
    a = pd.pivot_table(attrs, index=['sku_id'], columns=['attr_name'],
                        values=['attr_value'],fill_value = np.nan, aggfunc='max')
    a.columns = a.columns.droplevel(level=0)
    a = a.reset_index(drop=False)
    a = a.drop_duplicates()
    
    #to add web_id information
    a_web = pd.merge(a,web_id,how='inner',on='sku_id')
    a_web = a_web.drop_duplicates()
    
    #fill nan with the most frequent entry
    a_web = a_web.apply(lambda x: x.fillna(x.value_counts().index[0]))
    
    
    #regular expression, select the last part      
    for col in a_web.columns.difference(['sku_id','web_id']):
        a_web[col] = a_web[col].apply(lambda x: re.sub('#\$.*','',x))
    
    #phrase split, select the last part
    for col in a_web.columns.difference(['sku_id','web_id']):
        a_web[col] = a_web[col].apply(lambda x:x.split('/')[0])
    
    #handle abnormal value
    for col in a_web.columns.difference(['sku_id','web_id']):
        a_web[col] = a_web[col].apply(lambda x:x.replace(u'其他',u'其它'))
    
        
    #import sku_price table
    gdm_m04_gdm_m04 = params['worker']['dir']+'/input/'+params['EndDate']+\ + '/'
    +params['item_third_cate_cd']+'/gdm_m04_ord_det_sum'
    gdm_m04_ord_det_sum = pd.read_table('gdm_m04_ord_det_sum', sep='\t',encoding='utf-8')
    gdm_m04_ord_det_sum['sku_id'] = gdm_m04_ord_det_sum['item_sku_id']
    
    
    sale_count = gdm_m04_ord_det_sum.groupby(['sku_id']).agg({'sale_ord_tm':'count'})
    sale_count = sale_count.reset_index()
    sale_count['count'] = sale_count['sale_ord_tm']
    sale_count.drop('sale_ord_tm', axis = 1, inplace = True)
    
    
    valid_sale_count = sale_count[sale_count['count'] > 3]
    valid_sku_id = list(valid_sale_count['sku_id']) 
    gdm_m04_ord_det_sum = gdm_m04_ord_det_sum[gdm_m04_ord_det_sum['sku_id'].isin(valid_sku_id)]
    
    
    
    gdm_m04_ord_det_sum = gdm_m04_ord_det_sum.groupby(['sku_id']).agg({'before_prefr_amount':'sum','sale_qtty':'sum'})
    gdm_m04_ord_det_sum = gdm_m04_ord_det_sum.reset_index()
    gdm_m04_ord_det_sum['mean_price'] = gdm_m04_ord_det_sum['before_prefr_amount']/gdm_m04_ord_det_sum['sale_qtty']
    gdm_m04_ord_det_sum.drop(['before_prefr_amount','sale_qtty'], axis = 1, inplace = True)
    
    
    '''
    #import sku_price table
    sku_price = params['worker']['dir']+'/input/'+params['EndDate']+'/'+params['item_third_cate_cd']+'/sku_price'
    sku_price = pd.read_table('sku_price', sep='\t',encoding='utf-8')
    cols = ['sku_id','item_first_cate_cd','item_second_cate_cd','item_third_cate_cd','mean_price']
    sku_price.columns = cols
    
    sku_price.drop(['item_first_cate_cd','item_second_cate_cd',
                    'item_third_cate_cd'], axis = 1, inplace = True)
       
    sku_price = sku_price[~np.isnan(sku_price['mean_price'])] 
        
    sku_price['sku_id'] = sku_price['sku_id'].apply(lambda x:int(x))
    '''    
        
    #merge jd_pop_attrs table with sku_price table based on sku_id 
    jd_pop = pd.merge(a_web, gdm_m04_ord_det_sum, how = 'inner', on = 'sku_id')
    jd_pop = jd_pop.drop_duplicates()   
    jd_pop['mean_price'] = jd_pop['mean_price'].apply(lambda x: int(x))
    
    
    s = jd_pop.groupby(u'品牌').agg({'sku_id':'count'})
    s = s.reset_index()
    s['brand_count'] = s['sku_id']
    s.drop('sku_id',axis = 1, inplace=True)
    
    
    
    d = list(s[s['brand_count'] < 100][u'品牌'])
    f = list(jd_pop[u'品牌'])
    for i in d:
        jd_pop.loc[jd_pop[u'品牌'] == i, u'品牌'] = u'其它'

    
    #label encoder method to handle discrete/categorical features except continuous features
    for attribute in jd_pop.columns.difference(['mean_price','sku_id','web_id']):
        le = preprocessing.LabelEncoder()
        jd_pop[attribute] = le.fit_transform(jd_pop[attribute])
    
    
    #normalize continuous features('mean_price')
    jd_pop['mean_price'] = jd_pop['mean_price'].apply(lambda x: 
        (x-jd_pop['mean_price'].mean())/(jd_pop['mean_price'].std()))
    
        
    '''    
    #handle high cardinality of brand feature using kmeans clustering
    from sklearn.cluster import KMeans
    X = jd_pop[['mean_price',u'品牌']]
    kmeans = KMeans(n_clusters = 11, random_state = 0).fit(X)
    jd_pop[u'品牌'] = kmeans.labels_
    
    
    #use elbow method to find the best number of clusters
    c = range(10,50)
    ks = [KMeans(n_clusters = i) for i in c]
    score = [ks[i].fit(X).score(X) for i in range(len(ks))]
    plt.scatter(c,score)
    '''
    
    
    jd = jd_pop[jd_pop['web_id']==0]
    pop = jd_pop[jd_pop['web_id'] == 1]
    
    #import profit table
    sku_profit = pd.read_table('app_cfo_profit_loss_b2c_det', sep = '\t', encoding = 'utf-8')
    sku_profit['sku_id'] = sku_profit['item_sku_id']
    sku_profit.drop(['dt','item_third_cate_name','item_sku_id','cost_tax','income','grossfit','gross_sales','rebate_amunt_notax','adv_amount','store_fee','deliver_fee'], axis = 1, inplace = True)
    sku_profit['sku_id'] = sku_profit['sku_id'].apply(lambda x:int(x))
    
    
    
    #filter sku_profit table
    sku_profit = sku_profit[sku_profit['gmv'] > 1 ]
    
    
    #create the profit_rate column
    sku_profit['profit_rate'] = (sku_profit['net_profit']/sku_profit['gmv'])*100
    sku_profit = sku_profit[sku_profit['net_profit'] < sku_profit['gmv']]
    sku_profit.drop(['net_profit','gmv'],axis =1,inplace = True)
    sku_profit = sku_profit[sku_profit['profit_rate'] > -200]
    
    
    ave = np.mean(sku_profit['profit_rate'])
    std = np.std(sku_profit['profit_rate'])
    upper = ave + 2.5*std
    lower = ave - 2.5*std
    
    #filter results
    sku_profit = sku_profit[sku_profit['profit_rate'] > lower]
    sku_profit = sku_profit[sku_profit['profit_rate'] < upper]
    
    
    #calculate the profit_rate records per sku_id
    sku_count =  sku_profit.groupby('sku_id').count()
    sku_count = sku_count.reset_index()
    sku_count['count'] = sku_count['profit_rate']
    sku_count.drop('profit_rate',axis = 1, inplace = True)
    
    
    #filter profit rate for every sku_id, keep the sku_id with records less than 4
    col = ['sku_id','profit_rate']
    p = pd.DataFrame(columns = col)
    
    fewer_sku_count = sku_count[sku_count['count'] <= 4] 
    unique_sku_id = list(fewer_sku_count['sku_id'])
    for sku_id in unique_sku_id:
        duplicate_sku_id = sku_profit[sku_profit['sku_id']==sku_id].sort_values('profit_rate', ascending=False)
        unique = duplicate_sku_id.iloc[:]
        p = pd.concat([p,unique],axis = 0)
    p['sku_id'] = p['sku_id'].apply(lambda x: int(x))
    
    
    
    #filter profit rate for every sku_id, drop the max2 and min2 profit rate for sku_id with records greater than 4
    q = pd.DataFrame(columns = col)
    greater_sku_count = sku_count[sku_count['count'] > 4]
    greater_sku_count = greater_sku_count[greater_sku_count['count'] <= 12]
    unique_sku_id2 = list(greater_sku_count['sku_id'])
    
    for sku_id in unique_sku_id2:
        duplicate_sku_id2 = sku_profit[sku_profit['sku_id']==sku_id].sort_values('profit_rate', ascending=False)
        unique2 = duplicate_sku_id2.iloc[1:-1]
        q = pd.concat([q,unique2],axis = 0)
    q['sku_id'] = q['sku_id'].apply(lambda x: int(x))
    
    p_q = pd.concat([p,q],axis = 0)
    
    
    o = pd.DataFrame(columns = col)
    most_sku_count = sku_count[sku_count['count'] > 12]
    unique_sku_id3 = list(most_sku_count['sku_id'])
    
    for sku_id in unique_sku_id3:
        duplicate_sku_id3 = sku_profit[sku_profit['sku_id']==sku_id].sort_values('profit_rate', ascending=False)
        unique3 = duplicate_sku_id3.iloc[3:-3]
        o = pd.concat([o,unique3],axis = 0)
    o['sku_id'] = o['sku_id'].apply(lambda x: int(x))
    
    
    sku_profit = pd.concat([p_q,o],axis = 0)
    
    #extract the mean sku_id profit table
    average_profit = sku_profit.groupby('sku_id').agg({'profit_rate':'mean'})
    average_profit.reset_index(inplace=True)
    
    
    #merge attributes table and mean profit table based on sku_id
    net_profit_percent = pd.merge(jd,average_profit, how = 'inner', on = 'sku_id')
    
    
    net_profit_percent.drop(['sku_id','web_id'],axis = 1, inplace = True)
    
    
           
    #train_test_split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(net_profit_percent.drop('profit_rate',axis=1),+ \
                                                        net_profit_percent['profit_rate'], + \
                                                        test_size=0.30, + \
                                                        random_state = 101)
    
    
    #optimize algotirhm and tune parameter with GridSearchCV
    from sklearn.grid_search import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(  n_estimators = 500, 
                                  max_features = 'auto',
                                  max_depth=8,
                                  min_samples_leaf=4,
                                  min_samples_split=8,
                                  oob_score=True,
                                  #random_state = 42,
                                  n_jobs=-1,
                                  criterion = 'mae',
                                  bootstrap = True)
    param_grid = {
    'n_estimators':[100,200,500],       
    'max_depth':[3,5,8],
    'min_samples_leaf':[2,4,6],
    'min_samples_split':[4,8,10]
    }
    
    CV_rfr= GridSearchCV(estimator=rfr, param_grid=param_grid, cv=5)
    CV_rfr.fit(X_train, y_train) 
    #print (CV_rfr.best_params_)
    
    best_model = CV_rfr.best_estimator_
    #implement RandomForestregressor to solve regression problem    
    #from sklearn.ensemble import RandomForestRegressor
    #rfr = RandomForestRegressor(  n_estimators = CV_rfr.best_params_['n_estimators'], 
    #                              max_features = 'auto',
    #                              max_depth=CV_rfr.best_params_['max_depth'],
    #                              min_samples_leaf=CV_rfr.best_params_['min_samples_leaf'],
    #                              min_samples_split=CV_rfr.best_params_['min_samples_split'],
    #                              oob_score=True,
    #                              #random_state = 42,
    #                              criterion = 'mae',
    #                              n_jobs=-1,
    #                              bootstrap = True)
    #                              #warm_start=False,
    #                              #max_leaf_nodes = 30)
    #rfr.fit(X_train, y_train)
    '''
    predictions = best_model.predict(X_test)
           
    #plt.scatter(y_test,predictions)
    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    print('MAPE:', mean_absolute_percentage_error(y_test,predictions))
    
       
    #subplots method of matplotlib 
    fig, axes = plt.subplots(nrows = 2, ncols = 1)
    axes[0].scatter(y_test, predictions)
    plt.sca(axes[1]) #Use the pyplot interface to change just one subplot
    plt.xticks(range(X_train.shape[1]),X_train.columns, color='r')
    axes[1].bar(range(X_train.shape[1]),rfr.feature_importances_, color= 'b',align = 'center')
    '''
    
    #implement the profit_prediction algorihtm on pop skus
    pop.drop(['sku_id','web_id'], axis = 1, inplace = True)
    pop_predictions = best_model.predict(pop)
    pop_predicitons.index.name = 'profit_rate'
    
    #save to file
    out_path = params['worker']['dir']+'/output/'+params['EndDate']+'/'+ params['item_third_cate_cd']
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    pop_predictions.to_csv(out_path+'/profit_rate.txt',header=False,sep='\t',encoding='utf-8',index=False)


if __name__ == '__main__':
    #read command line arguments
    n = len(sys.argv) - 1
    if n < 1:
        print('Usage:\n    pyhton profit_rate_predict.py param_file\n')
        sys.exit()
    else:
        param_file = sys.argv[1]
        print('[INFO] profit__rate_predict started')
    
    #read parameters
    params = yaml.load(file('params/default.yaml','r'))
    user_params = yaml.load(file(param_file,'r'))
    for key, value in user_params.items():
        params[key] = value
    
    main(params)
    print('[INFO] profit_rate_predict completed')








