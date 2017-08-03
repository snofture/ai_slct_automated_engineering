# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:41:06 2017

@author: limen
"""

import sys
import os
import numpy as np
import pandas as pd
import yaml
#os.chdir('E:/code/cid3/6223')



def main(params):
    # read JD skus
    sku_file = params['worker']['dir'] + '/input/' + params['EndDate'] + '/' + \
               params['item_third_cate_cd'] + '/gdm_m03_item_sku_da'
    jd_items = pd.read_table(sku_file, quotechar='\0', dtype={'item_sku_id':str})
    jd_items = jd_items[jd_items['sku_status_cd']==3001]
    jd_items = jd_items['item_sku_id'].values

    
    # construct a dataframe contains self, pop and tmall skus
    skulist_file = params['worker']['dir'] + '/input/' + params['EndDate'] + '/' + \
                   params['item_third_cate_cd'] + '/app_ai_slct_sku'
    skulist = pd.read_table(skulist_file, quotechar='\0', dtype={'sku_id':str})
    skulist = skulist[skulist['dt']==params['EndDate']]
    df_3in1 = skulist[['web_id','sku_id','sku_name','item_third_cate_cd']].drop_duplicates(subset=['sku_id'],keep='last')
    df_3in1.rename(columns={'sku_id':'item_sku_id'}, inplace=True)
    ind = df_3in1['item_sku_id'].apply(lambda x: x in jd_items)
    df_3in1 = df_3in1[(ind==True) | (df_3in1['web_id']==2)]
    df_3in1.index = df_3in1['item_sku_id']
    
    
    # filter item fourth category
    if params['scope_type']=='lvl4':
        cid4_file = params['worker']['dir'] + '/temp/' + params['EndDate'] + '/' + \
                    params['item_third_cate_cd'] + '/item_fourth_cate'
        cid4 = pd.read_table(cid4_file, quotechar='\0', dtype={'sku_id':str})
        sku4 = cid4[cid4['attr_value']==params['scope_desc'].encode('utf-8')]['sku_id'].values
        df_3in1 = df_3in1[df_3in1['item_sku_id'].isin(sku4)]
    
    
    
    # add sales
    sale_file = params['worker']['dir'] + '/temp/' + params['EndDate'] + '/' + \
                    params['item_third_cate_cd'] + '/sale_summary'
    sale_summary = pd.read_table(sale_file, quotechar='\0', dtype={'item_sku_id':str})
    sale_summary.set_index('item_sku_id', inplace=True)
    df_3in1['average_net_sales'] = sale_summary['average_net_sales']
    df_3in1['total_net_sales'] = sale_summary['total_net_sales']
    df_3in1['first_day_of_sale'] = sale_summary['first_day_of_sale']
    df_3in1['predicted_average_net_sales'] = sale_summary['predicted_sales']
    df_3in1['average_net_sales'] = df_3in1['average_net_sales'].fillna(value=0)
    df_3in1['total_net_sales'].fillna(value=0,inplace=True)
    df_3in1['first_day_of_sale'].fillna(value=params['StartDate'],inplace=True)
    df_3in1['predicted_average_net_sales'].fillna(value=0,inplace=True)
    
    
    #add attributes
    attrs_cg_file = params['worker']['dir'] + '/temp/' + params['EndDate'] + '/' + \
                        params['item_third_cate_cd'] + '/attributes_categorical'
    attrs_nm_file = params['worker']['dir'] + '/temp/' + params['EndDate'] + '/' + \
                    params['item_third_cate_cd'] + '/attributes_numerical'
    attrs_cg = pd.read_table(attrs_cg_file, quotechar='\0', dtype={'item_id':str})
    attrs_nm = pd.read_table(attrs_nm_file, quotechar='\0', dtype={'item_id':str})
    attrs_cg.drop_duplicates(subset=['item_id','attr_key'], keep='last', inplace=True)
    attrs_nm.drop_duplicates(subset=['item_id','attr_key'], keep='last', inplace=True)
    attrs_cg_matrix = attrs_cg.pivot(index='item_id', columns='attr_key', values='attr_val')
    attrs_nm_matrix = attrs_nm.pivot(index='item_id', columns='attr_key', values='attr_val')
    df_3in1 = pd.concat([df_3in1, attrs_cg_matrix, attrs_nm_matrix], axis=1, join_axes=[df_3in1.index])
    
    
    # add self switching
    switch_file = params['worker']['dir'] + '/output/' + params['EndDate'] +\
                  '/' + params['scope_id'] + '/switching_prob.txt'
    switch = pd.read_table(switch_file, header=None, quotechar='\0')
    switch.columns=['scope_id','src_item_id','dst_item_id','switching_prob','model']
    switch['src_item_id'] = switch['src_item_id'].apply(str)
    switch['dst_item_id'] = switch['dst_item_id'].apply(str)
    selfswitch = switch[switch['src_item_id']==switch['dst_item_id']]
    selfswitch = selfswitch[selfswitch['switching_prob'] > 0]
    selfswitch.set_index('src_item_id', inplace=True)
    df_3in1['selfswitching'] = selfswitch['switching_prob']
    
    
    
    # add predicted self switching
    pred_switch_file = params['worker']['dir'] + '/output/' + params['EndDate'] + '/' + \
                       params['scope_id'] + '/predicted.txt'
    pred_switch = pd.read_table(pred_switch_file, header=None, quotechar='\0')
    pred_switch.columns = ['scope_id','web','item_sku_id','switchprob_predicted']
    pred_switch['item_sku_id'] = pred_switch['item_sku_id'].apply(str)
    pred_switch.set_index('item_sku_id', inplace=True)
    df_3in1['predicted_selfswitching'] = pred_switch['switchprob_predicted']
    
    
    # add final_switching
    df_3in1['final_switching'] = df_3in1['selfswitching']
    ind = pd.isnull(df_3in1['final_switching'])
    df_3in1.loc[ind,'final_switching'] = df_3in1.loc[ind,'predicted_selfswitching']
    df_3in1['final_switching'] = df_3in1['final_switching'].apply(lambda x: x**2)
    df_3in1['final_switching'] = df_3in1['final_switching'].fillna(value=0)
    
    
    '''
    # add margin
    margin_file = params['worker']['dir'] + '/input/' + params['EndDate'] + '/' + \
                  params['item_third_cate_cd'] + '/app_cfo_profit_loss_b2c_det'
    margin = pd.read_table(margin_file, quotechar='\0', dtype={'item_sku_id':str})
    margin_rate = margin.groupby(['item_sku_id'])[['net_profit','gmv']].sum().reset_index()
    margin_rate['margin_rate'] = margin_rate['net_profit'] / margin_rate['gmv']
    margin_rate.fillna(value=0, inplace=True)
    margin_rate.set_index('item_sku_id', inplace=True)
    df_3in1['margin_rate'] = margin_rate['margin_rate']
    mean_margin_rate = margin_rate['net_profit'].sum() / margin['gmv'].sum()
    print 'mean margin rate is %s' % (mean_margin_rate,)
    df_3in1['margin_rate'].replace([np.nan, np.inf, -np.inf], value=mean_margin_rate, inplace=True)
    #df_3in1['average_net_margin'] = df_3in1['margin_rate'] * df_3in1['average_net_sales']
    #df_3in1['total_net_margin'] = df_3in1['margin_rate'] * df_3in1['total_net_sales']
    '''
    
    
    #add margin
    profit_rate_predict = params['worker']['dir']+'/output/'+params['EndDate']+'/'+ params['scope_id']+ '/profit_rate_predict'
    profit_rate = pd.read_table(profit_rate_predict, sep='\t',encoding='utf-8',dtype={'item_sku_id':str})
    profit_rate.set_index('item_sku_id', inplace=True)
    
    
    
    #margin_rate = margin.groupby(['item_sku_id'])[['net_profit','gmv']].sum().reset_index()
    #margin_rate['margin_rate'] = margin_rate['net_profit'] / margin_rate['gmv']
    
    df_3in1['profit_rate'] = profit_rate['profit_rate']
    self_mean = np.mean(df_3in1[df_3in1['web_id']==0]['profit_rate'])
    pop_mean = np.mean(df_3in1[df_3in1['web_id']==1]['profit_rate'])
    tmall_mean = np.mean(df_3in1[df_3in1['web_id']==2]['profit_rate'])

    print(self_mean)
    print(pop_mean)
    print(tmall_mean)
    df_3in1.loc[(df_3in1['web_id']==0) & (df_3in1['profit_rate'].isnull()),'profit_rate'] = self_mean
    df_3in1.loc[(df_3in1['web_id']==1) & (df_3in1['profit_rate'].isnull()),'profit_rate'] = pop_mean
    df_3in1.loc[(df_3in1['web_id']==2) & (df_3in1['profit_rate'].isnull()),'profit_rate'] = tmall_mean
    #df_3in1[df_3in1['web_id']==0]['profit_rate'] = df_3in1[df_3in1['web_id']==0]['profit_rate'].fillna(self_mean)
    #df_3in1[df_3in1['web_id']==1]['profit_rate'] = df_3in1[df_3in1['web_id']==1]['profit_rate'].fillna(pop_mean)
    #df_3in1[df_3in1['web_id']==2]['profit_rate'] = df_3in1[df_3in1['web_id']==2]['profit_rate'].fillna(tmall_mean)
    #df_3in1['profit_rate'].replace([np.nan, np.inf, -np.inf], value=mean_profit_rate, inplace=True)
    df_3in1['average_net_margin'] = (df_3in1['profit_rate']/100.00) * df_3in1['average_net_sales']
    df_3in1['total_net_margin'] = (df_3in1['profit_rate']/100.00) * df_3in1['total_net_sales']
    
    '''
    #add margin
    all_net_profit_predict = params['worker']['dir']+'/output/'+params['EndDate']+'/'+ params['scope_id']+ '/all_net_profit_predict_no_zero'
    net_profit = pd.read_table(all_net_profit_predict, sep='\t',encoding='utf-8',dtype={'item_sku_id':str})
    net_profit=net_profit.drop_duplicates()
    net_profit.set_index('item_sku_id', inplace=True)
    df_3in1['net_profit'] = net_profit['net_profit']
    '''
    
    
    # save to file
    temp_path = params['worker']['dir'] + '/temp/' + params['EndDate'] + '/' + \
                params['scope_id']
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    df_3in1.to_csv(temp_path+'/selection_input_trial_profit_rate.txt', sep='\t', header=True, index=False, quoteing=None, encoding='utf-8')




if __name__ == '__main__':
    # read command line arguments
    n = len(sys.argv) - 1
    if n < 1:
        print 'Usage: \n    python utility.py param_file\n'
        sys.exit()
    else:
        param_file = sys.argv[1]
        print("[INFO] utility started")

    # read parameters
    params = yaml.load( file('params/default.yaml', 'r') )
    user_params = yaml.load( file(param_file, 'r') )
    for key, value in user_params.items():
        params[key] = value

    main(params)
    print("[INFO] utility completed")


