# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 17:02:52 2017

@author: limen
"""

import os
import time
import sys
from datetime import datetime,timedelta
from pyspark.sql import HiveContext
from pyspark import SparkContext, SparkConf
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import *
''' 
n = len(sys.argv) - 1
if n < 1:
    print 'enter date to start calculation \n'
    sys.exit()
else :
    dt = sys.argv[1]
'''
sc = SparkContext(appName='caculate pop vender switch')
hc = HiveContext(sc)
#dt = '2017-06-01'
dt = str(datetime.now().date())
begin_dt = datetime.strptime(dt, '%Y-%m-%d') - timedelta(days=91)
begin_dt = str(begin_dt.date())
# 获取店铺信息
top_pop_query = ''' select vender_id as pop_vender_id, shop_name from dev.op_pop_shop_gmv_top_N '''
top_pop_id = hc.sql(top_pop_query).distinct().coalesce(50).cache()
top_pop_id = F.broadcast(top_pop_id)
log_query = ''' select  sku_id as item_sku_id,user_log_acct, date(request_tm) as request_dt, 
                        concat(session_id,bs) as session_id,pop_vender_id,item_first_cate_cd,
                        item_second_cate_cd,item_third_cate_cd
                from dev.gdm_m14_online_log_item_d_op_pop 
                where  dt >= "%s" and dt < "%s" 
                and session_id is not null and bs in ("1","13","8","311210")'''%(begin_dt,dt)
data = hc.sql(log_query).coalesce(1200)
data = top_pop_id.join(data,'pop_vender_id','inner')
data = data.withColumnRenamed('pop_vender_id', 'from_shop_id').withColumnRenamed('shop_name', 'from_shop_name')
tmp = data.withColumnRenamed('from_shop_id', 'to_shop_id').withColumnRenamed('item_third_cate_cd', 'to_cid3')\
          .withColumnRenamed('from_shop_name', 'to_shop_name')
# 过滤同一天的session 才计算替代性
switch = data.join(tmp,['session_id','request_dt'],'inner')
switch = switch.filter(switch.item_third_cate_cd == switch.to_cid3)
switch = switch.withColumn('page_views', F.lit(1))
switch = switch.groupby('from_shop_id','from_shop_name','to_shop_id','to_shop_name')\
               .agg(F.sum('page_views').alias('page_views_switch'))
from_shop_total_switch = switch.groupby('from_shop_id').agg(F.sum('page_views_switch').alias('from_shop_total_switch'))
new_column = from_shop_total_switch.from_shop_id.cast("string")
from_shop_total_switch = from_shop_total_switch.withColumn('from_shop_id',new_column)
switch = switch.join(from_shop_total_switch,'from_shop_id','inner')
switch = switch.withColumn('switch_prob', switch.page_views_switch/switch.from_shop_total_switch)
switch = switch[['from_shop_id','from_shop_name','to_shop_id','to_shop_name',
                 'page_views_switch','from_shop_total_switch','switch_prob']]
# 添加店铺url信息
shop_inf = hc.sql('select vender_id,shop_url from gdm.gdm_m01_vender_da where dt = "%s" '%(dt))
a1 = shop_inf.withColumnRenamed('vender_id','from_shop_id').withColumnRenamed('shop_url','from_shop_url')
a2 = shop_inf.withColumnRenamed('vender_id','to_shop_id').withColumnRenamed('shop_url','to_shop_url')
result = switch.join(a1,'from_shop_id','left').join(a2,'to_shop_id','left')
result= result[['from_shop_id','from_shop_name','from_shop_url','to_shop_id','to_shop_name',
                'to_shop_url','page_views_switch','from_shop_total_switch','switch_prob']]
# 保存结果
hc.registerDataFrameAsTable(result, "table1")
insert_sql = '''insert overwrite table dev.dev_open_pricing_pop_similarity_replacement partition(dt="%s") 
                select * from table1'''%(dt)
hc.sql(insert_sql)
'''
create table dev.dev_open_pricing_pop_similarity_replacement(
    from_shop_id string comment '被替代的pop商家的pop_vender_id',
    from_shop_name string comment '被替代的pop商家的对应的店铺名称',
    from_shop_url string comment '被替代的pop商家的对应的店铺url',
    to_shop_id string comment '替代from_shop_id商家的pop_vender_id',
    to_shop_name string comment '替代from_shop_id商家的店铺名称',
    to_shop_url string comment '替代from_shop_id商家的店铺url',
    page_views_switch bigint comment '从from_shop_id到to_shop_id的跳转次数',
    from_shop_total_switch bigint comment '从from_shop_id到其他商家总的跳转次数',
    switch_prob double comment '替代概率'
) 
PARTITIONED BY ( 
  `dt` string)
ROW FORMAT DELIMITED  
  FIELDS TERMINATED BY '\t'  
stored as orc;
'''