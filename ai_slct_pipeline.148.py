# -*- coding: utf-8 -*-
"""
Created on Mon Jul 03 16:45:33 2017

@author: limen
"""
import numpy as np
import pandas as pd
import os
import sys
import commands
import re
import yaml
reload(sys)
sys.setdefaultencoding('utf-8')

os.chdir('E:/code/product_selection_automated_engineering')

def add_time_frame(params):
    #calculate time frame and add them into params if needed
    if params.has_key('EndDate'):
        end_date = datetime.datetime.strptime(params['EndDate'],'%Y-%m-%d')
    else:
        today = datetime.date.today()
        end_date = today - datetime.timedelta(days=today.day)
        params['EndDate'] = end_date.strftime('%Y-%m-%d')
    if params.has_key('StartDate'):
        pass
    else:
        year = end_date.year - params['nYears']
        month = end_date.month + 1
        if month > 12:
            month = month - 12
            year = year + 1
        start_date = datetime.date(year, month, 1)
        params['StartDate'] = start_date.strftime('%Y-%m-%d')
    return params



def add_category_information(cate3s,params):    
    #construct category information in dict form
    file = open(params['category_tree'])
    line = file.readline().strip('\n')
    header = line.split('\t')
    cgs = dict()
    while 1:
        line=file.readline().strip('\n')
        if not line:
            break
        cg = dict()
        record = line.split('\t')
        for i in range(len(record)):
            cg[header[i]] = record[i]
        cgs[cg['item_third_cate_cd']] = cg
    
    #query full category information
    cate1s = list()
    cate2s = list()
    descs = list()
    
    for cate in cate3s.split('-'):
        if cgs.has_key(cate):
            cate1 = cgs[cate]['item_first_cate_cd']
            cate2 = cgs[cate]['item_second_cate_cd']
            desc = cgs[cate]['item_third_cate_name']
        else:
            cate1 = '0'
            cate2 = '0'
            desc = 'None'
            
        cate1s.append(cate1)
        cate2s.append(cate2)
        descs.append(desc)
        
    params['item_first_cate_cd'] = '-'.join(cate1s)
    params['item_second_cate_cd'] = '-'.join(cate2s)
    params['item_third_cate_cd'] = cate3s
    params['scope_desc'] = '-'.join(descs)
    
    return params


def add_scope_id(params):
    params['scope_id'] = '%s_%s_%s_%s_%s_%s_%s' %
    (params['item_first_cate_cd'],
     params['item_second_cate_cd'],
     params['item_third_cate_cd'],
     params['self_pop'],
      params['scope_type'],
      params['scope_desc'],
      params['item_type'])
    
    return params

def add_log_file(params):
    params['log_file'] = params['worker']['dir'] + '/logs/'
    + params['EndDate'] + '/' + params['item_third_cate_cd']
    + '.log'
    
    return params

def create_local_path(params):
    local_path = '/input/' + params['EndDate'] + '/' + params['item_third_cate_cd']
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    log_path = params['wordker']['dir'] + '/logs/' + params['EndDate']
    if not os.path.exists(log_path):
        os.makedirs(log_path)












def fetch_app_cfo_profit_loss_b2c_det(params):
    local_path = 'input/' + params['end_date'] + '/' + params['third_cate_item_cd'] 
    query = '''
    set hive.cli.print.header=true; 
    select * from app.app_cfo_profit_loss_b2c_det 
    where dt >= '%s' and dt <= '%s' and item_third_cate_name in (%s);
    ''' % (params['StartDate'],params['EndDate'],"'" + re.sub("-","','", params['scope_desc']) + "'")
    
    query = query.replace('\n','')
    cmd = 'hive -e "%s" > %s/app_cfo_profit_loss_b2c_det' % (query, local_path)
    print(cmd)
    (status,output) = commands.getstatusoutput(cmd)
    if status == 0:
        print('download app_cfo_profit_loss_b2c_det success!')
    else:
        print('download app_cfo_profit_loss_b2c_det failed! \n' + output + '\n')
    return status


def fetch_app_ai_slct_attributes(params):
    local_path = 'input/' + params['end_date'] +'/' +params['item_third_cate_cd']
    query = '''
    set hive.cli.print.header=true;
    select * from app.app_ai_slct_attributes
    where dt = '%s' and item_third_cate_cd in (%s);
    ''' % (params['EndDate'],str(params['item_third_cate_cd']).replace('-',','))
    
    query = query.replace('\n','')
    cmd = 'hive -e "%s" > %s/app_ai_slct_attributes' % (query, local_path)
    print(cmd)
    (status,output) = commands.getstatusoutput(cmd)
    if status == 0:
        print('download app_ai_slct_attributes success!')
    else:
        print('download app_ai_slct_attributes failed! \n' + output + '\n')
    return status


def fetch_gdm_m04_ord_det_sum(params):
    local_path = 'input/'+params['EndDate']+'/'+params['item_third_cate_cd']
    query = '''
    set hive.cli.print.header=true;
    select user_id, parent_sale_ord_id, item_sku_id, sale_ord_id, sale_ord_tm,
        sale_qtty, after_prefr_amount, before_prefr_amount
        from gdm.gdm_m04_ord_det_sum
        where dt >= '%s' and sale_ord_dt >= '%s' and sale_ord_dt <= '%s' and 
        item_third_cate_cd in (%s) and sale_ord_valid_flag=1;
    ''' % (params['StartDate'],params['StartDate'],params['EndDate'],
    str(params['item_third_cate_cd']).replace('-',','))
    
    query = query.replace('\n','')
    cmd = 'hive -e "%s" |grep -v \'^WARN:\' > %s/gdm_m04_ord_det_sum' % (query, local_path)
    print(cmd)
    (status,output) = commands.getstatusoutput(cmd)
    if status == 0:
        print('download gdm_m04_ord_det_sum success!')
    else:
        print('download gdm_m04_ord_det_sum failed! \n' + output + '\n')
    return status
        

def fetch_gdm_m03_item_sku_da(params):
    local_path = 'input/' + params['EndDate'] + '/'+ params['item_third_cate_cd']
    query = '''
    set hive.cli.print.header = true;
    select sku.item_sku_id, sku.sku_name, sku.sku_status_cd, sku.wt, sku.spu_id
           sku.item_third_cate_cd, sku.dt
           case when getDataTypeBySkuId(cast(sku.item_sku_id as bigint))=10
                     then 'self'
                when getDataTypeBySkuId(cast(sku.item_sku_id as bigint)) in (1,2,3,4,5,6,7,8,9)
                     then 'pop'
                else 'other'
           end
    from gdm.gdm_m03_item_sku_da sku
    where sku.dt = '%s' and sku.item_third_cate_cd in (%s);
    ''' % (params['EndDate'], str(params['item_third_cate_cd']).replace('-',','))
    
    query = query.replace('\n','')
    cmd = 'hive -e "%s" > %s/gdm_m03_item_sku_da' % (query, local_path)
    print(cmd)
    (status,output) = commands.getstatusoutput(cmd)
    if status == 0:
        print('download gdm_m03_item_sku_da success!')
    else:
        print('download gdm_m03_item_sku_da failed! \n' + output + '\n')
    return status


def fetch_app_ai_slct_sku(params):
    local_path = 'input/' + params['EndDate'] + '/' + params['item_third_cate_cd']
    query = '''
    set hive.cli.print.header.true;
    select * from app.app_ai_slct_sku
    where dt <= '%s' and item_third_cate_cd in (%s)
    ''' % (params['EndDate'], str(params['item_third_cate_cd']).replace('-',','))
    
    query = query.replace('\n','')
    cmd = 'hive -e "%s" > %s/app_ai_slct_sku' % (query, local_path)
    print(cmd)
    (status,output) = commands.getstatusoutput(cmd)
    if status == 0 :
        print('download app_ai_slct_sku success')
    else:
        print('download app_ai_slct_sku failed \n' + output + '\n')
    return status

def fetch_app_ai_slct_gmv(params):
    local_path = 'input/' + params['EndDate'] + '/' + params['item_thid_cate_cd']
    query = '''
    set hive.cli.print.header=true;
    select * from app.app_ai_slct_gmv
    where dt <= '%s' and item_third_cate_cd in (%s)
    ''' % (params['EndDate'], str(params['item_third_cate_cd']).replace('-',','))
    
    query = query.replace('\n','')
    cmd = 'hive -e "%s" > %s/app_ai_slct_gmv ' % (query, local_path)
    print(cmd)
    (status, output) = commands.getstatusoutput(cmd)
    if status == 0:
        print('download app_ai_slct_gmv success!')
    else:
        print('download app_ai_slct_gmv failed \n' + output + '\n')
    return status

def fetch_app_ai_slct_match(params):
    local_path = 'input/' + params['EndDate'] + '/' + params['item_third_cate_cd']
    query = '''
    set hive.cli.print.header=true;
    select * from app.app_ai_slct_match
    where dt = '%s' and item_third_cate_cd in (%s)
    ''' % (params['EndDate'], str(params['item_third_cate_cd']).replace('-',','))
    
    query = query.replace('\n','')
    cmd = 'hive -e "%s" > %s/app_ai_slct_match' % (query, local_path)
    print(cmd)
    (status, output) = commands.getstatusoutput(cmd)
    if status == 0:
        print('download app_ai_slct_match success!')
    else:
        print('download app_ai_slct_match failed! \n' + output + '\n')
    return status











def fetch_app_ai_slct_sku_price(params):
    local_path = 'input/' + params['EndDate']+'/'+params['item_third_cate_cd']
    query = '''
    hive set.cli.print.header=true;
    select a.item_sku_id,a.item_first_cate_cd,a.item_second_cate_cd,
    a.item_third_cate_cd,b.mean_price as price 
    from (select item_sku_id,item_first_cate_cd,item_second_cate_cd,item_third_cate_cd 
    from gdm.gdm_m03_item_sku_da where dt = '%s' and item_third_cate_cd in (%s))a 
    join (select sku_id, avg(jd_prc)as mean_price 
    from gdm.gdm_m03_item_sku_price_da group by sku_id) b on a.item_sku_id ==b.sku_id
    ''' % (params['EndDate'],str(params['item_third_cate_cd']).replace('-',','))
    query = query.replace('\n','')
    cmd = 'hive -e "%s" > %s/sku_price' % (query,local_path)
    print(cmd)
    (status,output) = commands.getstatusoutput()
    if status == 0:
        print('download sku_price success!')
    else:
        print('download sku_price failed! \n' + output + '\n')
    return status
    
    
    
    
   
    





def fetch_jd_tmall_brand_mapping(params):
    local_path = 'input/'+params['EndDate']+'/'+params['item_third_cate_cd']
    query = '''
    set hive.cli.print.header=true;
    select jdstdbrandname, tmstdbrandname from app.app_sys_determined_jd_tmall_mapping
    where dt = '%s' and jdcategoryid3 in (%s)
    and status == 1;
    ''' % (params['EndDate'], str(params['item_third_cate_cd']).replace('-',','))
    
    query = query.replace('\n','')
    cmd = 'hive -e "%s" > %s/jd_tmall_brand_mapping' % (query, local_path)
    print(cmd)
    (status, output) = commands.getstatusoutput(cmd)
    if status == 0:
        print('download jd_tmall_brand_mapping success!')
    else:
        print('download jd_tmall_brand_mapping failed! \n' + output +'\n')   
    return status

def fetch_app_aicm_jd_std_brand_da(params);:
    local_path = 'input/' + params['EndDate']
    query = '''
    set hive.cli.print.header=true;
    select * from app.app_aicm_jd_std_brand_da 
    where dt = '%s'
    ''' % (params['EndDate'])
    query = query.replace('\n','')
    cmd = 'hive -e "%s" > %s/app_aicm_jd_std_brand_da' % (query, local_path)
    print(cmd)
    (status, output) = commands.getstatusoutput(cmd)
    if status == 0:
        print('download aicm_jd_std_brand_da success!')
    else:
        print('download aicm_jd_std_brand_da failed! \n' + output + '\n')
    return status



def new_month(params):
    local_file = 'input/' + params['EndDate'] +'/app_aicm_jd_std_brand_da' 
    if os.path.isfile(local_file):
        return False
    else:
        return True
    

def create_param(params):
    local_path = 'params/' + 'sku_' + params['item_third_cate_cd']+'.yaml'
    stream = file(local_path, 'w')
    yaml.safe_dump(params, stream, allow_unicode=True, default_flow_style=False)
    
    return local_path

def create_param_brand(params):
    local_path = 'params/' + 'brand_' + params['item_third_cate_cd'] + '.yaml'
    params['scope_id'] = params['scope_id'].replace('_sku','_brand')
    params['item_type'] = 'brand'
    stream = file(local_path, 'w')
    yaml.safe_dump(params,stream, allow_unicode=True, default_flow_style=False)
    
    return local_path



def run_task_local(params):
    work_dir = params['worker']['dir']
    item_third_cate_cd = params['item_third_cate_cd']
    lvl = params['scope_type']
    
    
    #本机执行shell命令
    cmd = 'python %s/computation_pipe.py %s %s' % (work_dir, item_third_cate_cd, lvl)
    (status, output) = commands.getstatusoutput(cmd)
    if status == 0:
        print('computation task done!')
    else:
        print('computation task failed: \n' + output + '\n')
        
    return params











































