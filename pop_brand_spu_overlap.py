# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 17:04:03 2017

@author: limen
"""

import os
import time
import sys
from datetime import datetime,timedelta
from pyspark.sql import HiveContext
from pyspark.sql.functions import *
from pyspark.sql import Window
from pyspark import SparkContext, SparkConf
'''
n = len(sys.argv) - 1
if n < 1:
    print 'enter date to start calculation \n'
    sys.exit()
else :
    dt = sys.argv[1]
'''
#dt = '2017-06-14'
# 每天计算昨天的
dt = str(datetime.now().date()-timedelta(days=1))
sc = SparkContext(appName='caculate pop vender switch')
hc = HiveContext(sc)
################### 基于spu的重合度计算，匹配关系取最新的分区的数据
last_dt = hc.sql('select max(dt) from  dev.dev_pop_vender_sku_match ').collect()[0][0]
vender_spu_match = hc.sql('''select sku_id1,main_sku_id1,shop_name1,pop_vender_id1,sku_id2,main_sku_id2,shop_name2,pop_vender_id2
                             from dev.dev_pop_vender_sku_match where dt in ("2017-06-10","%s") '''%(last_dt)).coalesce(100).cache()
tmp_spu_match = hc.sql('''select sku_id2 as sku_id1,main_sku_id2 as main_sku_id1,shop_name2 as shop_name1,pop_vender_id2 as pop_vender_id1,
                                 sku_id1 as sku_id2,main_sku_id1 as main_sku_id2,shop_name1 as shop_name2,pop_vender_id1 as pop_vender_id2  
                            from dev.dev_pop_vender_sku_match where dt in ("2017-06-10","%s") '''%(last_dt)).coalesce(100).cache()
all_match = vender_spu_match.union(tmp_spu_match)
all_match = all_match.distinct()
# 提取所有商家的sku,spu 关系
# sku_pop_query = '''select item_sku_id as sku_id1,main_sku_id as main_sku_id1 from gdm.gdm_m03_pop_item_sku_da
#                     where dt = "%s" and length(pop_vender_id) > 0
#                     and item_first_cate_cd in ('6994','6196','1316','1620','9847',
#                     '12259','1672','1319','12473','6728','1320','6233','9192','1318') '''%('2017-06-01')
#
# sku_pop_query = '''select item_sku_id as sku_id1,main_sku_id as main_sku_id1 from gdm.gdm_m03_pop_item_sku_da
#                     where dt = "%s" and length(pop_vender_id) > 0
#                     and item_first_cate_cd in ("1320") '''%('2017-06-01')
#
# # sku_pop_query = '''select sku_id as sku_id1,main_sku_id as main_sku_id1 from dev.valid_pop_sku_da
# #                     where dt >= "%s" and dt < "%s" '''%('2017-05-01','2017-06-01')
#
# all_sku_spu = hc.sql(sku_pop_query).coalesce(200).distinct().cache()
#
# # 提取过去一个月有gmv 的sku,关联spu 信息
# top_sku = hc.sql('''select vender_id as pop_vender_id1,sku_id as sku_id1,gmv from dev.op_pop_sku_gmv_da where dt>='2017-05-01' and dt < '2017-06-01' ''').cache()
# top_sku_spu = top_sku.join(all_sku_spu,'sku_id1','inner')
# # 取累计销量在前80%的spu
# spu_gmv = top_sku_spu.groupby(['pop_vender_id1','main_sku_id1']).agg(sum('gmv').alias('spu_gmv'))
# gmv_rank = Window.partitionBy('pop_vender_id1').orderBy(spu_gmv.spu_gmv.desc())
# spu_gmv = spu_gmv.withColumn('gmv_rank',rank().over(gmv_rank))
#
# window = Window.partitionBy('pop_vender_id1').orderBy(spu_gmv.gmv_rank.asc()).rowsBetween(-sys.maxsize, 0)
# spu_gmv = spu_gmv.select('*',sum('spu_gmv').over(window).alias('cum_gmv'))
# total_gmv = spu_gmv.agg(max('cum_gmv')).first()[0]
# spu_gmv = spu_gmv.withColumn('gmv_percent',spu_gmv.cum_gmv/total_gmv)
# spu_gmv = spu_gmv.filter(spu_gmv.gmv_percent<=0.8)
# spu_gmv = spu_gmv.select('pop_vender_id1','main_sku_id1')
# # 查看总体的sku 数目
# top_spu = spu_gmv.join(top_sku_spu,['pop_vender_id1','main_sku_id1'],'inner')
# top_pop_query = ''' select vender_id  as pop_vender_id1 from dev.op_pop_shop_gmv_top_20 where cid1 in ('1320') '''
# top_pop_id = hc.sql(top_pop_query).distinct().coalesce(50).cache()
# shop_spu_num = top_spu.join(top_pop_id,'pop_vender_id1','inner')
# shop_spu_num = shop_spu_num.groupby('pop_vender_id1').agg(countDistinct('main_sku_id1').alias('spu_num'))
# # 过滤sku匹配范围
# top_spu_match = all_match.join(top_sku,['pop_vender_id1','sku_id1'],'inner')
# 汇总spu匹配数目
spu_match = all_match.groupby('pop_vender_id1','shop_name1','pop_vender_id2','shop_name2').agg(countDistinct(all_match.main_sku_id1).alias('macth_spu_num')).cache()
spu_match = spu_match.sort('pop_vender_id1','macth_spu_num')
# 计算每个店铺有匹配关系的spu数目
shop_spu_num = all_match.groupby('pop_vender_id1').agg(countDistinct('main_sku_id1').alias('spu_num'))
# 计算重合度
spu_match = spu_match.join(shop_spu_num,'pop_vender_id1','inner')
spu_match = spu_match.withColumn('overlap_ratio',spu_match.macth_spu_num/spu_match.spu_num)
#  关联url 信息
shop_inf = hc.sql('select vender_id,shop_url from gdm.gdm_m01_vender_da where dt = "%s" '%(dt))
a1 = shop_inf.withColumnRenamed('vender_id','pop_vender_id1').withColumnRenamed('shop_url','shop_url_1')
a2 = shop_inf.withColumnRenamed('vender_id','pop_vender_id2').withColumnRenamed('shop_url','shop_url_2')
result = spu_match.join(a1,'pop_vender_id1','left').join(a2,'pop_vender_id2','left')
result= result[['pop_vender_id1','shop_name1','shop_url_1','pop_vender_id2','shop_name2',
                'shop_url_2','macth_spu_num','spu_num','overlap_ratio']]
# 保存到hive表
hc.registerDataFrameAsTable(result, "table1")
insert_sql = '''insert overwrite table dev.dev_open_pricing_pop_similarity_spu_overlap_da partition(dt="%s") 
                select * from table1'''%(dt)
hc.sql(insert_sql)
''' 
create table dev.dev_open_pricing_pop_similarity_spu_overlap_da(
    pop_vender_id1 string comment '第一个pop_vender_id',
    shop_name1 string comment '第一个商家店铺名称',
    shop_url_1 string comment '第一个商家店铺url',
    pop_vender_id2 string comment '第二个pop_vender_id',
    shop_name2 string comment '第二个商家店铺名称',
    shop_url_2 string comment '第二个商家店铺url',
    macth_spu_num bigint comment '商家1中sku和商家2中匹配上的spu数目',
    valid_spu_number bigint comment '商家1中过去一个月有销量的spu数目',
    overlap_ratio double comment '重合度'
) 
PARTITIONED BY ( 
  `dt` string)
ROW FORMAT DELIMITED  
  FIELDS TERMINATED BY '\t'  
stored as orc;
'''