# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 17:05:06 2017

@author: limen
"""

import os
import sys
import yaml
import csv
from pyspark.sql import HiveContext
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark import SparkContext, SparkConf
import pandas as pd
from datetime import datetime,timedelta
import commands
reload(sys)
sys.setdefaultencoding('utf-8')

n = len(sys.argv) - 1
if n < 1:
    print 'Usage: \n  python switch_log.py param_file\n'
    sys.exit()
else:
    param_file = sys.argv[1]
    print 'param_file:', param_file, '\n'

# read parameters
#os.chdir('/data0/data_dir/czq/ai_slct/')
#param_file = './params/sku_9888.yaml'

params = yaml.load( file('params/default.yaml', 'r') )
user_params = yaml.load( file(param_file, 'r') )
for key, value in user_params.items():
    params[key] = value

for item in params:
    print item,':',params[item]

sc = SparkContext(appName='caculate switch')
hc = HiveContext(sc)

dt = str(datetime.now().date())
begin_dt = str(datetime.now().date()-timedelta(days=91))

# 读取点击数据和sku数据
log_query = ''' select  sku_id as item_sku_id,user_log_acct,date(request_tm) as request_dt, 
                        concat(session_id,bs) as session_id  
                from dev.gdm_m14_online_log_item_d_op_full 
                where  dt >= "%s" and dt < "%s" and item_third_cate_id = "%s" 
                and session_id is not null and bs in ("1","13","8","311210")'''%(begin_dt,dt,params['item_third_cate_cd'])

data = hc.sql(log_query).coalesce(1000)

sku_file = params['worker']['dir'] + '/input/' + params['EndDate'] + '/' \
           + params['item_third_cate_cd'] + '/gdm_m03_item_sku_da'

m03 = pd.read_table(sku_file,header='infer',sep='\t',quote=csv.QUOTE_NONE)

# 过滤自营
if params['self_pop'] == 'self':
    self_sku = m03[m03.sku_type == 'self']['item_sku_id'].tolist()
    self_sku = [str(x) for x in self_sku]
    data = data[data.item_sku_id.isin(self_sku)]
elif params['self_pop'] == 'pop':
    pop_sku = m03[m03.sku_type == 'pop']['item_sku_id'].tolist()
    pop_sku = [str(x) for x in pop_sku]
    data = data[data.item_sku_id.isin(pop_sku)]
else:
    all_sku = m03[m03.sku_type.isin('pop','self')]['item_sku_id'].tolist()
    all_sku = [str(x) for x in all_sku]
    data = data[data.item_sku_id.isin(all_sku)]

# 过滤四级分类范围的sku
if params['scope_type'] == 'lvl4':
    cid4_file = params['worker']['dir'] + '/temp/' + params['EndDate'] + '/' \
                + params['item_third_cate_cd'] + '/item_fourth_cate'
    cid4 = pd.read_table(cid4_file,header='infer',sep='\t',dtype={'sku_id':str,'attr_value':str},encoding='utf-8')
    sku4 = cid4[cid4.attr_value == params['scope_desc']]['sku_id'].tolist()
    sku4 = [str(x) for x in sku4]
    data = data[data.item_sku_id.isin(sku4)]


#### 将sku 映射到标准品牌
if params['item_type'] == 'brand':
    attr_file = params['worker']['dir'] + '/input/' + params['EndDate'] + '/' \
                + params['item_third_cate_cd'] + '/app_ai_slct_attributes'
    brand_file = params['worker']['dir'] + '/input/' + params['EndDate'] + '/app_aicm_jd_std_brand_da'
    attr = pd.read_table(attr_file,header='infer',sep='\t',dtype={'sku_id':str})
    brand = pd.read_table(brand_file, header='infer', sep='\t',dtype={'jd_brand_id':str})
    brand = brand[['jd_brand_id','jd_brand_name']].drop_duplicates().dropna(how='any')
    sku_brand = attr[(attr.web_id < 2) & (attr.attr_name == '品牌')][['sku_id','attr_value']].drop_duplicates()
    sku_brand = sku_brand.merge(brand,left_on='attr_value',right_on='jd_brand_name',how = 'inner')
    sku_brand['scope_id'] = params['scope_id']
    sku_brand['item_type'] = 'brand'
    sku_brand['model'] = 'log'
    sku_brand = sku_brand.drop('attr_value',axis=1)
    output_dir = params['worker']['dir'] + '/output/' + params['EndDate']+ '/' + params['scope_id']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sku_brand = sku_brand[['scope_id','sku_id','jd_brand_id','jd_brand_name','item_type','model']]
    sku_brand.to_csv(output_dir + '/sku_in_scope_test.txt',header=False,sep='\t',index=False,encoding='utf-8')
    sku_brand = hc.createDataFrame(sku_brand[['sku_id','jd_brand_id']])
    sku_brand = sku_brand.withColumnRenamed('sku_id','item_sku_id')
    data = data.join(sku_brand,'item_sku_id','inner')
    data = data.withColumn('item_sku_id',data.jd_brand_id).drop('jd_brand_id')

##### 计算替代性
data = data.withColumnRenamed('item_sku_id', 'src_item_id')
tmp = data.withColumnRenamed('src_item_id', 'dst_item_id')

switch = data.join(tmp,['user_log_acct','session_id','request_dt'],'inner')
switch = switch.withColumn('page_views', F.lit(1))

switch = switch.groupby('src_item_id', 'dst_item_id').agg(F.sum('page_views').alias('page_views_switch'))
from_sku_total_switch = switch.groupby('src_item_id').agg(F.sum('page_views_switch').alias('from_sku_total_switch'))

new_column = from_sku_total_switch.src_item_id.cast("string")
from_sku_total_switch = from_sku_total_switch.withColumn('src_item_id',new_column)
switch = switch.join(from_sku_total_switch,'src_item_id','inner')
switch = switch.withColumn('switching_prob', switch.page_views_switch/switch.from_sku_total_switch)
#switch = switch[['from_sku_id','to_sku_id','page_views_switch','from_sku_total_switch','switch_prob']]
switch = switch.withColumn('model',F.lit('log')).withColumn('scope_id',F.lit(params['scope_id']))
switch = switch.select('scope_id','src_item_id','dst_item_id','switching_prob','model')

# 保存结果
output_path = params['worker']['dir'] + '/output/' + params['EndDate'] + '/' + params['scope_id']
switch.toPandas().to_csv(output_path+'/switching_prob.txt',header=False,index=False,sep='\t',encoding='utf-8')

print '[PIPE]',params['scope_id'],'\n'
print '[INFO] switch_log completed at',datetime.now(),'\n'