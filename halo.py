# -*- coding: utf-8 -*-
""" 
Created on Tue Dec 13 15:13:06 2016
@author: chenzhiquan
"""

import os  
import sys 
import yaml 
from pyspark.sql import HiveContext 
import pyspark.sql.functions as F
from pyspark.sql import Window 
from pyspark import SparkContext, SparkConf
import pandas as pd
from datetime import datetime,timedelta
import commands
import item_fourth_cate as ifc

#read command line arguments
n = len(sys.argv) - 1
if n < 1:
    print 'Usage: \n  python halo.py param_file\n'
    sys.exit()
else:
    param_file = sys.argv[1]
    print 'param_file:', param_file, '\n'

# read parameters
#os.chdir('/data0/data_dir/xia/ai_slct/')
#param_file = './params/sku_1594-5021.yaml'
params = yaml.load( file('params/default.yaml', 'r') )
user_params = yaml.load( file(param_file, 'r') )
for key, value in user_params.items():
    params[key] = value

# one year time frame
end_date = datetime.strptime(params['EndDate'],'%Y-%m-%d')
start_date = datetime.strptime(params['EndDate'],'%Y-%m-%d') - timedelta(days=365)
params['begin_date'] = start_date.strftime('%Y-%m-%d')
params['end_date'] = end_date.strftime('%Y-%m-%d')
halo_params = params.copy()

specify_first_cate = str(halo_params['item_first_cate_cd'])
specify_third_cate = str(halo_params['item_third_cate_cd'])
specify_second_cate = str(halo_params['item_second_cate_cd'])

path =  halo_params['hdfs_path']
ord_file = halo_params['ord_file']
sc = SparkContext(appName='caculate halo')
hc = HiveContext(sc) 
####读取数据订单数据
####################查看是否有订单数据
dt =  halo_params['end_date']
start_date = params['begin_date']
hdfs_path = '/user/mart_cis/czq/halo/%s/'%dt
cid1s = specify_first_cate.split('-')
if len(set(['1316', '1319', '1320', '12259','911']).union(cid1s)) == 5 :
   file_name = '1316_1319_1320_12259_911'
   first_cates = ('1316', '1319', '1320', '12259','911')
else:
   file_name = '_'.join(set(cid1s))
   first_cates = "('" + ",'".join(set(cid1s)) + "')"

hdfs_file = hdfs_path+file_name
local_path = '/data0/data_dir/czq/halo/input/%s/'%(dt)
local_file =  local_path + file_name

if not os.path.exists(local_path):
    os.makedirs(local_path)
    print 'create local path %s'%(local_path)
    
if  os.system(('hadoop fs -test -e %s')%hdfs_path) != 0:
    os.system(('hadoop fs -mkdir %s')%hdfs_path)
    print 'create hdfs file dir %s'%(hdfs_path)


query = '''
select b.*,a.item_first_cate_cd,a.item_second_cate_cd,a.item_third_cate_cd 
from
        (select item_sku_id,item_first_cate_cd,item_second_cate_cd,item_third_cate_cd
        from gdm.gdm_m03_item_sku_da  where dt = '%s' and item_first_cate_cd in %s) a 
join    (select 
                user_id,parent_sale_ord_id,item_sku_id,sale_ord_id,sale_ord_tm,
                sale_qtty,after_prefr_amount from gdm.gdm_m04_ord_det_sum
         where  dt  > '%s'   and sale_ord_valid_flag   = '1' and sale_ord_tm > '%s'
                and sale_ord_tm <= '%s' and length(pop_vender_id) = 0 
                and item_first_cate_cd  in  %s)b  
on a.item_sku_id == b.item_sku_id ''' %('2017-04-18',first_cates,start_date,start_date,dt,first_cates)

query = query.replace('\n','')
cmd = 'hive -e "%s" > %s'%(query,local_file)
#a = os.system(cmd)

if  os.system(('hadoop fs -test -e %s'%hdfs_file)) != 0:
    a = os.system(cmd)
    if a == 0:
       print 'load order data into %s successfully '%(local_file)
       b = os.system(('hadoop fs -put %s %s')%(local_file,hdfs_path))
       if b == 0:
          print 'put order data into hdfs path %s'%hdfs_path
       else:
          print 'faild to put order data into hdfs'
          os.system(('hadoop fs -rm %s ')%(hdfs_file))
    else :
       os.system(('rm %s')%local_file)
       os.system(('hadoop fs -rm %s ')%(hdfs_file))
       print 'faild to load order data'

################## 
rdd = sc.textFile(hdfs_file)\
        .map(lambda x:x.replace(chr(0),'s').split('\t'))\
        .filter(lambda x:len(x) == 10)
columns = ['user_id','parent_sale_ord_id','item_sku_id','sale_ord_id',
           'sale_ord_tm','sale_qtty','after_prefr_amount','item_first_cate_cd',
           'item_second_cate_cd','item_third_cate_cd']
ord = hc.createDataFrame(rdd,schema=columns)        
ord = ord.withColumn('sale_ord_tm',F.to_date(ord.sale_ord_tm))

## 合并相同的三级分类
cid3s = specify_third_cate.split('-')
cid2s = specify_second_cate.split('-')
cid1s = specify_first_cate.split('-')
if len(cid3s)>1:
    for item in cid3s :
        ord = ord.replace([item],[specify_third_cate],'item_third_cate_cd')
    for item in cid2s :
        ord = ord.replace([item],[specify_second_cate],'item_second_cate_cd')
    for item in cid1s :
        ord = ord.replace([item],[specify_first_cate],'item_first_cate_cd')

#订单数据去重
ord = ord.dropDuplicates(['user_id','parent_sale_ord_id','item_sku_id','sale_ord_id','sale_ord_tm'])

####计算用户(2015-09-01--2016-08-31)在FMCG下的总消费，总消费排名；选取累计消费额在前80%的用户作为FMCG专注用户；
#  时间范围获取:
ord = ord.filter((ord.sale_ord_tm >= halo_params['begin_date'])&(ord.sale_ord_tm <= halo_params['end_date']))

#计算每个用户在FMCG下的总消费额(考虑滤除消费额为负的情况)
user_sales = ord.groupby('user_id').agg(F.sum('after_prefr_amount').alias('total_net_sales'))

#计算用户在FMCG下的总消费额排名
user_sales = user_sales.withColumn('sales_rank', F.rank().over(Window.partitionBy().orderBy(user_sales.total_net_sales.desc())))

#对所有用户计算累加消费,选取前80%的用户作为FMCG专注用户
window = Window.partitionBy().orderBy(user_sales.sales_rank.asc()).rowsBetween(-sys.maxsize, 0)
user_sales = user_sales.select('*',F.sum('total_net_sales').over(window).alias('cum_netSales'))
total_Sales = user_sales.agg(F.max('cum_netSales')).first()[0]
user_sales = user_sales.withColumn('sales_percent',user_sales.cum_netSales/total_Sales)
FMCG_primary_user = user_sales.filter(user_sales.sales_percent<=0.8).select('user_id')

####计算FMCG专注用户在FMCG下对每个sku的总消费，购物篮次数，交易次数(由于produckey是合并后的，所以交易次数可能会大于购物篮次数)；
temp_ord = ord.filter(ord.item_third_cate_cd==specify_third_cate)
temp_ord = hc.createDataFrame(temp_ord.rdd,schema=temp_ord.columns)
primary_ord = FMCG_primary_user.join(temp_ord,"user_id","inner")
primary_basket = primary_ord.groupby('user_id','item_sku_id').agg(F.sum('after_prefr_amount').alias('sku_net_sales'),\
                                     F.countDistinct('parent_sale_ord_id').alias('sku_basket_num'),\
                                     F.countDistinct('sale_ord_id').alias('sku_order_num'))

####计算每个FMCG专注用户在指定三级品类下的消费总额；并关联其在总的FMCG下的消费总额；
#选取指定的三级品类的订单；
primary_level3 =  primary_ord.groupby('item_third_cate_cd','user_id')\
                             .agg(F.sum('after_prefr_amount').alias('level3_netSales'))

primary_level3 = hc.createDataFrame(primary_level3.rdd,schema=primary_level3.columns)
primary_level3 = primary_level3.join(user_sales.select('user_id','total_net_sales'),"user_id","inner")
primary_level3 = primary_level3.withColumn('salesPercent_level3',primary_level3.level3_netSales/primary_level3.total_net_sales).cache()

####计算在FMCG下的专注用户在三级分类下消费占总消费的百分比的均值avg，方差std;
primary_level3 =  primary_level3.withColumn('rank',F.rank().over(Window.partitionBy().orderBy(primary_level3.salesPercent_level3.desc())))
all_user_num = primary_level3.count()
user_focus =  primary_level3.filter(primary_level3.rank <= 0.05*all_user_num).select('user_id').withColumn('user_class',F.lit('focus'))
user_primary =  primary_level3.filter(primary_level3.rank > 0.05*all_user_num)\
                              .select('user_id').withColumn('user_class',F.lit('primary'))

temp_user = user_focus.unionAll(user_primary)
user_all= user_sales.select('user_id').join(temp_user,"user_id","left_outer").na.fill({"user_class":'other'})

####对该三级品类计算每一个sku在三种用户（focus,primary,non-primary）的总销售额，总用户数:
level3_sku_sales = temp_ord.join(user_all,'user_id','inner')\
                           .groupby('item_third_cate_cd','item_sku_id','user_class')\
                           .agg(F.sum('after_prefr_amount').alias('total_sales'),F.countDistinct('user_id').alias('user_number'))
level3_sku_sales = F.broadcast(level3_sku_sales)
level3_sku_sales.cache()

####计算每个三级品类下的sku缺失时的被迫替代概率，以及采用三种用户销售额占比加权计算的平均被迫替代性；
lvl = halo_params['scope_type']
if lvl == 'lvl4':
    pnames = ifc.main(halo_params)
    p = yaml.load(file(pnames[0], 'r'))
    switch_file = p['worker']['dir'] + '/output/' + p['EndDate'] + \
                  '/' + p['super_scope_id'] + '/switching_prob.txt'
else:
    switch_file = halo_params['worker']['dir'] + '/output/' + halo_params['EndDate'] + \
                  '/' + halo_params['scope_id'] + '/switching_prob.txt'

df = pd.read_table(switch_file, header=None)
df.columns = ['scop_id','from_item_sku_id','item_sku_id','switch_prob','type'] 
switch = hc.createDataFrame(df) \
           .select('item_sku_id','from_item_sku_id','switch_prob') \
           .filter('item_sku_id = from_item_sku_id')
           
switch = switch.withColumn('self_switch_prob',switch.switch_prob * switch.switch_prob)
switch = F.broadcast(switch)

tmp = switch.join(level3_sku_sales,'item_sku_id','inner')
tmp = tmp.withColumn('weight_switch_prob',tmp.self_switch_prob*tmp.total_sales)
tmp = tmp.groupby().sum('total_sales','weight_switch_prob').collect()
avgSelfSwitch =  tmp[0][1]/tmp[0][0]

####计算每个品类下的专注用户对此品类下的每个sku的总消费和损失概率，该品类的总消费;
####计算对专注用户其专注品类下的sku损失金额占该用户在品类下的占比p:
user_focus = F.broadcast(user_focus)
step1 = switch.join(primary_basket,'item_sku_id','right_outer')\
              .join(user_focus,'user_id','inner')\
              .join(primary_level3,'user_id','inner')\
              .na.fill({"self_switch_prob":avgSelfSwitch})

step1 = step1.withColumn('p',step1.self_switch_prob*step1.sku_net_sales/step1.level3_netSales)
####计算每个三级品类专注用户对于每个sku的效用f:也就是没有该sku时该用户在完全流逝和不完全流失情况下的损失;计算denominator：每个用户对每个sku的消费额*损失概率 
step1 =  step1.withColumn('f',step1.p*step1.total_net_sales+(1-step1.p)*step1.sku_net_sales*step1.self_switch_prob)\
              .withColumn('denominator',step1.sku_net_sales*step1.self_switch_prob)

####汇总每个三级品类的专注用户的效用函数;计算每个三级品类下每个专注用户的总销售额损失和占总消费额损失的最大概率；
step2 =  step1.groupby('item_third_cate_cd','user_id')\
              .agg(F.sum('f').alias('user_cate_f'),\
                   F.sum('denominator').alias('sum_net_sales_switch'),\
                   (F.sum('denominator')/F.sum('sku_net_sales')).alias('max_prob_total_loss'))

####对三级品类下的sku计算其专注用户用f分配新的权重后的损失；                   
step2 = hc.createDataFrame(step2.rdd,schema=step2.columns)
step1 = hc.createDataFrame(step1.rdd,schema=step1.columns)

step4 = step2.join(step1,['item_third_cate_cd','user_id'],'inner')
step4 = step4.withColumn('prob_allocated_to_product',step4.f/step4.user_cate_f)
step4 = step4.withColumn('expected_loss',step4.max_prob_total_loss*step4.prob_allocated_to_product*step4.total_net_sales)

####对每个sku汇总所有专注用户的expectedLoss计算alpha;并将alpha限定在大于1的范围；
step5 = step4.groupby('item_third_cate_cd','item_sku_id').agg((F.sum('expected_loss')/F.sum('denominator')).alias('alpha'))
#选取所有的sku_list
skulist = level3_sku_sales.select('item_sku_id').distinct()
skulist = F.broadcast(skulist)
skulist.cache()
#对于没有alpha值得
step6 = skulist.join(step5,'item_sku_id','left_outer').na.fill({'alpha':1}).cache()
####计算每个sku在三种不同用户的用户数，消费额，平均消费额，总消费额，以及光环效应
sku_all =  level3_sku_sales.groupby('item_sku_id').pivot('user_class',['focus','primary','other']).sum('total_sales','user_number')

sku_all = sku_all.withColumnRenamed('focus_sum(`total_sales`)','focus_user_total_sales').withColumnRenamed('focus_sum(`user_number`)','focus_user_num')\
                 .withColumnRenamed('primary_sum(`total_sales`)','primary_user_total_sales').withColumnRenamed('primary_sum(`user_number`)','primary_user_num')\
                 .withColumnRenamed('other_sum(`total_sales`)','other_user_total_sales').withColumnRenamed('other_sum(`user_number`)','other_user_num')

sku_all = sku_all.withColumn('avg_focus_user_sales',sku_all.focus_user_total_sales/sku_all.focus_user_num)\
                 .withColumn('avg_primary_user_sales',sku_all.primary_user_total_sales/sku_all.primary_user_num)\
                 .withColumn('avg_other_user_sales',sku_all.other_user_total_sales/sku_all.other_user_num)\
                 .withColumn('sku_total_sales',sku_all.focus_user_total_sales+sku_all.primary_user_total_sales+sku_all.other_user_total_sales)
sku_all = F.broadcast(sku_all)
#计算weightedsales:将alpha与专注用户的 weightsales,以及halo
#halo逻辑：对于一个sku 如果销售额大于0，专注用户消费额大于0，则计算其halo,其他的情况光环取为1；
step7 = sku_all.join(step6,['item_sku_id'],'inner')
halo_sku = step7.filter((step7.alpha>1)&(step7.focus_user_total_sales>0)&(step7.sku_total_sales>0)).select('item_sku_id')
halo_sku = step7.join(halo_sku,'item_sku_id','inner')\
                .withColumn('halo_factor',(step7.alpha*step7.focus_user_total_sales+step7.primary_user_total_sales+step7.other_user_total_sales)/step7.sku_total_sales)\
                .select('item_sku_id','halo_factor')
result = step7.join(halo_sku,'item_sku_id','left_outer').na.fill({'halo_factor':1})

dt =  halo_params['EndDate']
result = result.withColumn('halo_cover_cd',F.lit(halo_params['halo_cover_cd']))\
               .withColumn('halo_cover_name',F.lit(halo_params['halo_cover_name']))\
               .withColumn('item_first_cate_cd',F.lit(specify_first_cate))\
               .withColumn('item_second_cate_cd',F.lit(specify_second_cate))
               
columns = ['halo_cover_cd','halo_cover_name','item_sku_id','halo_factor','alpha','focus_user_num',
'focus_user_total_sales','avg_focus_user_sales','primary_user_num','primary_user_total_sales',
'avg_primary_user_sales', 'other_user_num','other_user_total_sales','avg_other_user_sales',
'sku_total_sales','item_first_cate_cd','item_second_cate_cd']

result = result[columns]
##保存结果
if lvl == 'lvl4':
    pnames = ifc.main(halo_params)
    p = yaml.load(file(pnames[0], 'r'))
    out_path = p['worker']['dir'] + '/output/' + p['EndDate'] + \
               '/' + p['super_scope_id']
else:
    out_path = halo_params['worker']['dir'] + '/output/' + halo_params['EndDate'] + \
               '/' + halo_params['scope_id']

if not os.path.exists(out_path):
    os.makedirs(out_path)
result.toPandas().to_csv(out_path+'/halo.txt',header=False,sep='\t',encoding='utf-8',index=False)
#result.toPandas().to_csv('/data0/data_dir/czq/halo_9888.txt',header=False,sep='\t',encoding='utf-8',index=False)

# 保存到hive表
hc.registerDataFrameAsTable(result, "table1")
insert_sql = '''insert overwrite table app.app_ai_slct_loyal_halo partition(dt="%s",item_third_cate_cd="%s") 
                select * from table1'''%(dt,specify_third_cate)
hc.sql(insert_sql)
sql_clause = 'select productkey as item_sku_id,\
                     ExistingCustomerHaloOutsideFMCG_2 as old_user_halo_outside,\
                     ExistingCustomerButNewToFMCGHaloWithinFMCG_3A as old_user_new_halo_withIn,\
                     BrandNewCustomerHaloOutsideFMCG_3C as new_user_halo_outside,\
                     BrandNewCustomerHaloWithinFMCG_3B as new_user_halo_withIn \
              from app.fact_FinalHalo_Product_Halo '
other_halo = hc.sql(sql_clause)
loyal_halo = result.select('item_sku_id','halo_factor','item_first_cate_cd','item_second_cate_cd')

all_halo = loyal_halo.join(other_halo,'item_sku_id','left_outer')

all_halo = all_halo.fillna({'old_user_halo_outside':0,'old_user_new_halo_withIn':0,
                                    'new_user_halo_outside':0,'new_user_halo_withIn':0})

all_halo = all_halo.withColumnRenamed('halo_factor','loyal_halo')
all_halo = all_halo[['item_sku_id','loyal_halo','old_user_halo_outside','old_user_new_halo_withIn',
                     'new_user_halo_outside','new_user_halo_withIn','item_first_cate_cd','item_second_cate_cd']]    
    
hc.registerDataFrameAsTable(all_halo, "all_halo")
insert_sql = '''insert overwrite table app.app_cis_ai_slct_sku_halo_all partition(dt="%s",item_third_cate_cd="%s") 
                select * from all_halo'''%(dt,specify_third_cate)
hc.sql(insert_sql)
