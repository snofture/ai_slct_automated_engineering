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
#sys.setdefaultencoding('utf-8')


#build MAPE function
def mean_absolute_percentage_error(y,p):   
    return np.mean(np.abs((y-p)/y))

def main(params):
    #import jd attributes table
    app_ai_slct_attributes = params['worker']['dir']+'/input/'+params['EndDate']+'/'+ params['item_third_cate_cd']+'/app_ai_slct_attributes'
    a =  pd.read_table(app_ai_slct_attributes,sep = '\t', encoding = 'utf-8')
    a = a[['sku_id','attr_name','attr_value','web_id']]
    a = a.drop_duplicates()
    
    web_id = a[['sku_id','web_id']]
    web_id = web_id.drop_duplicates()
    web_id['sku_id'] = web_id['sku_id'].apply(lambda x: str(x))
    
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
    attrs = jd_tmall_brand_mapping(a)
        
        
    #transform original table to pivot_table 
    att = pd.pivot_table(attrs, index=['sku_id'], columns=['attr_name'],
                        values=['attr_value'],fill_value = np.nan, aggfunc='max')
    att.columns = att.columns.droplevel(level=0)
    att = att.reset_index(drop=False)
    att = att.drop_duplicates()
    att['sku_id'] = att['sku_id'].apply(lambda x: str(x))
    
    
    #delete the rows with nan values more than half of total columns
    number_of_columns = att.shape[1]
    t = int((number_of_columns-1)/2) +1
    drop_null_att = att.dropna(thresh = t)
    
    #to add web_id information
    a_web = pd.merge(drop_null_att,web_id,how='inner',on='sku_id')
    a_web = a_web.drop_duplicates()
    a_web['sku_id'] = a_web['sku_id'].apply(lambda x: str(x))
    
    #fill nan for brand with u'其它'
    a_web[u'品牌'] = a_web[u'品牌'].fillna(u'缺失')
    
    
    #fill nan with the most frequent one if the missing entries are less than 25% of the column
    for col in a_web.columns.difference(['sku_id','web_id']):
        if (np.asscalar(np.int16(pd.isnull(a_web[col]).sum())) / len(a_web[col])) * 100 < 25:
            a_web[col] = a_web[col].fillna(a_web[col].value_counts().index[0])
        else:
            a_web[col] = a_web[col].fillna(u'其它')
    
   
    #handle abnormal value
    for col in a_web.columns.difference(['sku_id','web_id']):
        a_web[col] = a_web[col].apply(lambda x:x.replace(u'其他',u'其它'))
    
    #regular expression, select the last part      
    for col in a_web.columns.difference(['sku_id','web_id']):
        a_web[col] = a_web[col].apply(lambda x: re.sub('.*#\$','',x))
    
    #phrase split, select the last part
    for col in a_web.columns.difference(['sku_id','web_id']):
        a_web[col] = a_web[col].apply(lambda x:x.split('/')[0])
    
        
    #import sku_price table
    gdm_m04_ord_det_sum = params['worker']['dir']+'/input/'+params['EndDate']+ '/'+params['item_third_cate_cd']+'/gdm_m04_ord_det_sum'
    gdm_m04_ord_det_sum = pd.read_table(gdm_m04_ord_det_sum, sep='\t',encoding='utf-8')
    gdm_m04_ord_det_sum['sku_id'] = gdm_m04_ord_det_sum['item_sku_id']
    gdm_m04_ord_det_sum['sku_id'] = gdm_m04_ord_det_sum['sku_id'].apply(lambda x: str(x))

    
    #calculate the sale records per sku
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
    
    #calculate the mean_price
    gdm_m04_ord_det_sum = gdm_m04_ord_det_sum.groupby(['sku_id']).agg({'before_prefr_amount':'sum','sale_qtty':'sum'})
    gdm_m04_ord_det_sum = gdm_m04_ord_det_sum.reset_index()
    gdm_m04_ord_det_sum['sku_id'] = gdm_m04_ord_det_sum['sku_id'].apply(lambda x: str(x))
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
    
    '''
    s = jd_pop.groupby(u'品牌').agg({'sku_id':'count'})
    s = s.reset_index()
    s['brand_count'] = s['sku_id']
    s.drop('sku_id',axis = 1, inplace=True)
    
        
    d = list(s[s['brand_count'] < 3][u'品牌'])
    for i in d:
        jd_pop.loc[jd_pop[u'品牌'] == i, u'品牌'] = u'其它'
    '''
    
    #label encoder method to handle discrete/categorical features except continuous features
    le = preprocessing.LabelEncoder()
    for attribute in jd_pop.columns.difference(['mean_price','sku_id','web_id']):
        jd_pop[attribute] = le.fit_transform(jd_pop[attribute])
    
    
    #normalize continuous features('mean_price')
    jd_pop['mean_price'] = jd_pop['mean_price'].apply(lambda x: 
        (x-jd_pop['mean_price'].mean())/(jd_pop['mean_price'].std()))
    
    jd_pop['sku_id'] = jd_pop['sku_id'].apply(lambda x: str(x))    
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
    pop_tmall = jd_pop[jd_pop['web_id'] != 0]
    
    #import profit table
    app_cfo_profit_loss_b2c_det = params['worker']['dir']+'/input/'+params['EndDate']+ '/'+params['item_third_cate_cd']+'/app_cfo_profit_loss_b2c_det'
    sku_profit = pd.read_table(app_cfo_profit_loss_b2c_det, sep = '\t', encoding = 'utf-8')
    sku_profit['sku_id'] = sku_profit['item_sku_id']
    sku_profit.drop(['dt','item_third_cate_name','item_sku_id','cost_tax','income','grossfit','gross_sales','rebate_amunt_notax','adv_amount','store_fee','deliver_fee'], axis = 1, inplace = True)
    sku_profit['sku_id'] = sku_profit['sku_id'].apply(lambda x:str(x))
    
        
    #filter sku_profit table
    sku_profit = sku_profit[sku_profit['gmv'] > 1 ]
    sku_profit = sku_profit[sku_profit['net_profit'] < sku_profit['gmv']]
    
    #create the profit_rate column
    sku_profit['profit_rate'] = (sku_profit['net_profit']/sku_profit['gmv'])*100
    sku_profit.drop(['net_profit','gmv'],axis =1,inplace = True)
    sku_profit = sku_profit[sku_profit['profit_rate'] > -150]
    sku_profit = sku_profit[sku_profit['profit_rate'] != 0]
    
    #set the criteria for upper and lower bound dynamically
    number_of_profit = sku_profit.shape[0]
    global number_of_profit
    ave = np.mean(sku_profit['profit_rate'])
    std = np.std(sku_profit['profit_rate'])
    
    
    if number_of_profit < 500:
        upper = ave + 2.0 * std
        lower = ave - 1.0 * std
    elif (number_of_profit >= 500 and number_of_profit < 15000):
        upper = ave + 2.25*std
        lower = ave - 1.25*std
    elif (number_of_profit >= 15000 and number_of_profit < 35000):
        upper = ave + 2.75*std
        lower = ave - 1.75*std
    else:
        upper = ave + 2.75*std
        lower = ave - 2.0*std
    
    #filter results
    sku_profit = sku_profit[sku_profit['profit_rate'] > lower]
    sku_profit = sku_profit[sku_profit['profit_rate'] < upper]
    
    def control_profit_records(sku_profit):
        if number_of_profit > 3000:
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
        else:
            sku_profit = sku_profit
        return sku_profit
    
    sku_profit = control_profit_records(sku_profit)
    
    #extract the mean sku_id profit table
    average_profit = sku_profit.groupby('sku_id').agg({'profit_rate':'mean'})
    average_profit.reset_index(inplace=True)
    average_profit['sku_id'] = average_profit['sku_id'].apply(lambda x: str(x))
    
    
    #merge attributes table and mean profit table based on sku_id
    net_profit_percent = pd.merge(jd,average_profit, how = 'inner', on = 'sku_id')
    
    
    net_profit_percent.drop(['sku_id','web_id'],axis = 1, inplace = True)
    net_profit_percent = net_profit_percent[net_profit_percent['profit_rate'] != 0]
    
           
    #train_test_split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(net_profit_percent.drop('profit_rate',axis=1),
                                                        net_profit_percent['profit_rate'],
                                                        test_size=0.30,
                                                        random_state = 101)
    
    #find the best parameter for randomforest regressor model using hyperopt
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
    best = fmin(objective,space,algo = algo,max_evals=100)
        
    best['n_estimators'] = int(best['n_estimators'])
    best['max_depth'] = int(best['max_depth'])
    best['min_samples_leaf'] = int(best['min_samples_leaf'])
    best['min_samples_split'] = int(best['min_samples_split'])
    print (best)
    print objective(best)
    rf = RandomForestRegressor(**best)
    rf.fit(X_train,y_train)

    
    #implement the profit_prediction algorihtm on pop_tmall skus
    pop_tmall_sku = pop_tmall['sku_id']
    pop_tmall_sku = pop_tmall_sku.reset_index()
    pop_tmall_sku.drop('index',axis=1,inplace=True)
    pop_tmall.drop(['sku_id','web_id'], axis = 1, inplace = True)
    pop_predictions = rf.predict(pop_tmall)
    profit_predict = pd.DataFrame({'profit_rate': pop_predictions.tolist()})
    profit_rate_predict = pd.concat([pop_tmall_sku,profit_predict],axis = 1, ignore_index = True)
    profit_rate_predict.columns = ['item_sku_id','profit_rate']
    '''
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
    'max_depth':[5,8,10],
    'min_samples_leaf':[2,4,6],
    'min_samples_split':[4,8,10]
    }
    
    CV_rfr= GridSearchCV(estimator=rfr, param_grid=param_grid, cv=5)
    CV_rfr.fit(X_train, y_train) 
    #print (CV_rfr.best_params_)
    
    best_model = CV_rfr.best_estimator_
    
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

    
    #save to file
    out_path = params['worker']['dir']+'/output/'+params['EndDate']+'/'+ params['scope_id']
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    profit_rate_predict.to_csv(out_path+'/profit_rate_predict',header='infer',sep='\t',encoding='utf-8',index=False)


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








