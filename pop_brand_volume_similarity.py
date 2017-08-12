# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 17:01:46 2017

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
else:
    dt = sys.argv[1]
'''
sc = SparkContext(appName='caculate volumn similarity')
hc = HiveContext(sc)
#dt = '2017-05-31'
dt = str(datetime.now().date()-timedelta(days=1))
begin_dt = datetime.strptime(dt, '%Y-%m-%d') - timedelta(days=31)
begin_dt = begin_dt.strftime('%Y-%m-%d')
top_pop_query = ''' select vender_id from dev.op_pop_shop_gmv_top_N  where cid1 in ('1318','6728','1672','9570','1320',
                        '6196','1316','12259','1319','6233','6994','9192','9847','12473','9570') and gmv > 0 '''
top_pop_id = hc.sql(top_pop_query).coalesce(50).distinct().cache()
#### 计算每个商家在期gmv占比最高的三级分类下的相似商家
# 获取top 商家在每个三级分类下过去一个月的 gmv
cid3_query = ''' select * from dev.op_pop_cid3_gmv_da  where dt >= '%s' and dt <= '%s' '''%(begin_dt,dt)
vender_cid3_gmv = hc.sql(cid3_query)
vender_cid3_gmv = vender_cid3_gmv.groupby(['vender_id','cid3']).agg(sum('gmv').alias('gmv'))
top_vender_cid3 = top_pop_id.join(vender_cid3_gmv,'vender_id','inner')
#  计算所有商家在所有三级分类下的排名
gmv_rank = Window.partitionBy('cid3').orderBy(top_vender_cid3.gmv.desc())
top_vender_cid3 = top_vender_cid3.filter(top_vender_cid3.gmv > 0).withColumn('gmv_rank',rank().over(gmv_rank))
# 计算每个商家gmv最高的三级分类
cid3_gmv_rank = Window.partitionBy('vender_id').orderBy(top_vender_cid3.gmv.desc())
vender_cid3 = top_vender_cid3.withColumn('cid3_gmv_rank', rank().over(cid3_gmv_rank))
vender_top_cid3 = vender_cid3.filter(vender_cid3.cid3_gmv_rank == 1).select('vender_id','cid3')
#  计算每个商家gmv 最高的三级分类在所有商家中的排名
vender_top_cid3_rk = vender_top_cid3.join(top_vender_cid3,['vender_id','cid3'],'inner').select('vender_id','cid3','gmv_rank')
top_vender_cid3 = top_vender_cid3[['vender_id','cid3','gmv_rank']].withColumnRenamed('vender_id','similar_vender_id')\
                        .withColumnRenamed('gmv_rank','similar_vender_gmv_rank')
result = vender_top_cid3_rk.join(top_vender_cid3,'cid3','inner')
result = result.filter((result.gmv_rank >= result.similar_vender_gmv_rank-10) & (result.gmv_rank <= result.similar_vender_gmv_rank+10))
result = result.withColumnRenamed('cid3','group_id').withColumn('mode',lit('cid3'))
result = result[['vender_id','gmv_rank','similar_vender_id','similar_vender_gmv_rank','group_id','mode']]
#### 计算每个商家在期gmv占比最高的品牌下的相似商家
# 获取top 商家在每个品牌下的gmv
brand_query = ''' select * from dev.op_pop_brand_gmv_da  where dt >= '%s' and dt <= '%s' '''%(begin_dt,dt)
vender_brand_gmv = hc.sql(brand_query)
vender_brand_gmv = vender_brand_gmv.groupby(['vender_id','brand_code']).agg(sum('gmv').alias('gmv'))
top_vender_brand = top_pop_id.join(vender_brand_gmv,'vender_id','inner')
#  计算所有商家在所有品牌下的排名
gmv_rank = Window.partitionBy('brand_code').orderBy(top_vender_brand.gmv.desc())
top_vender_brand = top_vender_brand.filter(top_vender_brand.gmv > 0).withColumn('gmv_rank',rank().over(gmv_rank))
# 计算每个商家gmv最高的品牌
brand_gmv_rank = Window.partitionBy('vender_id').orderBy(top_vender_brand.gmv.desc())
vender_brand = top_vender_brand.withColumn('brand_gmv_rank', rank().over(brand_gmv_rank))
vender_top_brand = vender_brand.filter(vender_brand.brand_gmv_rank == 1).select('vender_id','brand_code')
#  计算每个商家gmv最高的品牌的gmv在所有商家中的排名
vender_top_brand_rk = vender_top_brand.join(top_vender_brand,['vender_id','brand_code'],'inner')\
                                    .select('vender_id','brand_code','gmv_rank')
top_vender_brand = top_vender_brand[['vender_id','brand_code','gmv_rank']]\
                                .withColumnRenamed('vender_id','similar_vender_id')\
                                .withColumnRenamed('gmv_rank','similar_vender_gmv_rank')
result_brand = vender_top_brand_rk.join(top_vender_brand,'brand_code','inner')
result_brand = result_brand.filter((result_brand.gmv_rank >= result_brand.similar_vender_gmv_rank-10) &
                                   (result_brand.gmv_rank <= result_brand.similar_vender_gmv_rank+10))
result_brand = result_brand.withColumnRenamed('brand_code','group_id').withColumn('mode',lit('brand'))
result_brand = result_brand[['vender_id','gmv_rank','similar_vender_id','similar_vender_gmv_rank','group_id','mode']]
# 合并并保存结果
result = result.union(result_brand)
result = result[result.vender_id != result.similar_vender_id]
hc.registerDataFrameAsTable(result, "table1")
insert_sql = '''insert overwrite table dev.dev_open_pricing_pop_similarity_volumn_da  
                partition(dt='%s') select * from table1'''%(dt)
hc.sql(insert_sql)
'''
create table dev.dev_open_pricing_pop_similarity_volumn_da(
     vender_id string comment '商家id' ,
     gmv_rank bigint comment '商家在三级分类或者品牌维度的gmv排名',
     similar_vender_id string comment '相似商家id',
     similar_vender_gmv_rank bigint comment '相似商家在相同模式下的gmv排名',
     group_id string comment '在mode为cid3时对应item_third_cate_cd,mode为brand时对应brand_code',
     mode string comment '两种相似模式,cid3对应三级分类，brand对应品牌'
) 
PARTITIONED BY ( 
  `dt` string)
ROW FORMAT DELIMITED  
  FIELDS TERMINATED BY '\t'  
stored as orc;
'''