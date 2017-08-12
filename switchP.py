# -*- coding:utf-8 -*-
__author__ = 'zhangzhan'

import os
import sys
import yaml
import datetime
import time
import commands
import re
import pandas as pd
import numpy as np
import math
reload(sys)
sys.setdefaultencoding('utf-8')

def mapping_to_item_id(item_sku_id,params):

    if params["item_type"] not in ["sku","brand"]:
        print "\nunsupport item_type:",params["item_type"]," program stopped\n"
        sys.exit()

    if params["item_type"] == "sku":
        #linux
        #sku_file = params["worker"]["dir"] + '/input/' + params["EndDate"] + '/' + params["item_third_cate_cd"]+'/gdm_m03_item_sku_da'
        sku_file = 'input/' + params["EndDate"] + '/' + params["item_third_cate_cd"] + '/gdm_m03_item_sku_da'
        sku = pd.read_table(sku_file,quotechar='\0',dtype={"item_sku_id":str},encoding="utf8")
        mapping = sku[sku["item_sku_id"].isin(item_sku_id)][["item_sku_id","sku_name"]].reset_index()
        mapping["item_id"] = mapping["item_sku_id"]
        mapping["scope_id"] = params["scope_id"]
        mapping.rename(columns={"sku_name":"item_name"})
        mapping["item_type"] = "sku"
        mapping = mapping.drop_duplicates();
        # linux
        #output_dir = params["worker"]["dir"] + '/output/' + params["EndDate"] + params["scope_id"]
        output_dir = 'output/' + params["EndDate"] + "/"+params["scope_id"]
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)
        mapping.to_csv(output_dir+"/sku_in_scope.txt",sep="\t",header=False,index=False,encoding="utf8")
        return item_sku_id
    elif params["item_type"] == "brand":

        #linux
        #attr_file = params["worker"]["dir"] + "/input/" + params["EndDate"] + '/' + params["item_third_cate_cd"]+'/app_ai_slct_attributes'
        attr_file = "input/" + params["EndDate"] + '/' + params["item_third_cate_cd"] + '/app_ai_slct_attributes'
        #linux
        #bid_file = params["worker"]["dir"] + '/input/' + params["EndDate"] + "/app_aicm_jd_std_brand_da"
        bid_file = 'input/' + params["EndDate"] + "/app_aicm_jd_std_brand_da"
        attr = pd.read_table(attr_file,quotechar='\0',encoding="utf8")
        bid = pd.read_table(bid_file,quotechar='\0',encoding="utf8")
        bid = bid.drop_duplicates("jd_brand_name")
        bid = bid[["jd_brand_id", "jd_brand_name"]]
        bid = bid.rename(columns={"jd_brand_name":"attr_value"})
        attr = attr[(attr["web_id"] < 2) & (attr["attr_name"] == "品牌")].reset_index()
        attr = attr.join(bid.set_index("attr_value"),on="attr_value",how="left")
        attr = attr.rename(columns={"jd_brand_id":"brand_id"})
        attr = attr.drop("index",1)
        attr = attr[["sku_id","attr_value","brand_id"]]
        attr = attr.drop_duplicates("sku_id")
        attr = attr.rename(columns={"sku_id": "item_sku_id"})
        attr["brand_id"] = attr["brand_id"].astype(str)
        item_sku_id = pd.DataFrame(item_sku_id)
        item_name_r = item_sku_id.join(attr.set_index("item_sku_id"),on="item_sku_id",how="left")
        item_name = item_name_r["attr_value"]
        item_id = item_name_r["brand_id"]

        mapping = pd.DataFrame(columns=["scope_id","item_sku_id","item_id","item_name","item_type"])
        mapping["item_sku_id"] = item_name_r["item_sku_id"]
        mapping["scope_id"] = params["scope_id"]
        mapping["item_id"] = item_id
        mapping["item_name"] = item_name
        mapping["item_type"] = "brand"
        mapping = mapping.drop_duplicates()
        #linux
        #output_dir = params["worker"]["dir"] + '/output/' + params["EndDate"] + '/' + params["scope_id"]
        output_dir =  'output/' + params["EndDate"] + '/' + params["scope_id"]
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)
        mapping.to_csv(output_dir+"/sku_in_scope.txt",sep="\t",index=False,header=False)
        return item_id
# def get_week(tt,st):
#
#     tt = datetime.datetime.strptime(str(tt),"Y%-%m-%d")
#     st = datetime.datetime.strptime(str(st),"Y%-%m-%d")
#     res = tt-st
#
#     return math.ceil((res.days+1)/float(7))

if __name__ == '__main__':

    #read command line arguments
    args_len = len(sys.argv) - 1

    # if args_len < 1:
    #     print "Usage: \n python switchP.py param_file\n"
    #     sys.exit()
    #
    # else :
    #     param_file = sys.argv[1]

    param_file = unicode("params/sku_1590_奶酪.yaml","utf8")

    params = yaml.load(file('params/default.yaml'))

    if os.path.isfile(param_file):
        user_params = yaml.load(file(param_file))
        for key in user_params.keys():
            params[key] = user_params[key]
    print params

    #linux
    # output_path = params["worker"]["dir"] + '/output/'+params["EndDate"] + '/'+params["scope_id"]

    output_path =  'output/' + params["EndDate"] + '/' + params["scope_id"]
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    # 读取订单数据
    # linux
    #ord_file = params["worker"]["dir"] + '/input/' + params["EndDate"] + '/' + params["item_third_cate_cd"] +"/gdm_m04_ord_det_sum"
    ord_file =  'input/' + params["EndDate"] + '/' + params["item_third_cate_cd"] + "/gdm_m04_ord_det_sum"
    ord = pd.read_table(ord_file,quotechar='\0')

    # 读取商品数据
    # linux
    # sku_file = params["worker"]["dir"] + '/input/' + params["EndDate"] +"/" + params["item_third_cate_cd"]+'/gdm_m03_item_sku_da'
    sku_file =  'input/' + params["EndDate"] + "/" + params["item_third_cate_cd"] + '/gdm_m03_item_sku_da'
    sku = pd.read_table(sku_file,quotechar='\0')

    # 过滤自营
    if params["self_pop"] == "self":
        self_sku = sku[sku["sku_type"] == "self"]["item_sku_id"]
        ord = ord[ord["item_sku_id"].isin(self_sku)]
    elif params["self_pop"] == "pop":
        pop_sku = sku[sku["sku_type"] == "pop"]["item_sku_id"]
        ord = ord[ord["item_sku_id"].isin(pop_sku)]
    else:
        all_sku = sku[sku["sku_type"].isin(["self","pop"])]["item_sku_id"]
        ord = ord[ord["item_sku_id"].isin(all_sku)]

    # 过滤四级分类
    if params["scope_type"] == 'lvl4':
        #linux
        #cid4_file = params["worker"]["dir"] + '/temp/'+params["EndDate"] + "/" + params["item_third_cate_cd"] + '/item_fourth_cate'
        #windows
        cid4_file =  'temp/' + params["EndDate"] + "/" + params["item_third_cate_cd"] + '/item_fourth_cate'
        cid4 = pd.read_table(cid4_file,quotechar='\0')
        sku4 = cid4[cid4["attr_value"] == params["scope_desc"]]["sku_id"]
        ord["item_sku_id"] = ord["item_sku_id"].astype(str)
        ord = ord[ord["item_sku_id"].isin(sku4)]

    # 只保留有效订单
    # ord = ord[ord["sale_ord_valid_flag"] == 1]

    #只保留订单时间在[*StartDate*, *EndDate*]内的订单数据
    #ord["sale_ord_dt"] = ord["sale_ord_tm"].map(lambda dt:time.strptime(dt,"%Y-%m-%d"))
    ord["sale_ord_dt"] = map(lambda s:s[:10],ord["sale_ord_tm"])
    ord = ord[(ord["sale_ord_dt"] >= params["StartDate"]) & (ord["sale_ord_dt"] <=params["EndDate"])]

    # 计算每位顾客的购物次数
    #customer = ord[["user_id","parent_sale_ord_id"]].groupby("user_id")["parent_sale_ord_id"].nunique().rename("ord_qtty").reset_index()
    customer = ord.groupby("user_id")["parent_sale_ord_id"].agg({"ord_qtty":"nunique"}).reset_index()

    # 只保留累计销量占比小于等于prodTail的sku的订单数据
    if params["prodTail"] < 1:
        sku = ord.groupby("item_sku_id")["sale_qtty"].agg({"sale_qtty":"sum"}).reset_index()
        sku = sku.sort_values("sale_qtty",ascending=False).reset_index()
        total_sale_qtty = np.sum(sku["sale_qtty"])
        sku["cum_ratio"] = sku["sale_qtty"].cumsum()/total_sale_qtty
        skulist = sku[sku["cum_ratio"] <= params["prodTail"]]["item_sku_id"]
        ord = ord[ord["item_sku_id"].isin(skulist)]

    # mapping to item id
    ord["item_id"] =mapping_to_item_id(ord["item_sku_id"],params)
    ord = ord[ord["item_id"].notnull()]

    # 计算总销售额大于*SwitchingMinSales*的item
    item_sum = ord.groupby("item_id")["before_prefr_amount"].agg({"sale_amount":"sum"}).reset_index()
    item_in_scope = item_sum[item_sum["sale_amount"] > params["SwitchingMinSales"]]["item_id"]

    # 计算有销售额周数大于等于*MinWeekswithSales*的item
    item_daily = ord[["item_id","sale_ord_dt","before_prefr_amount"]].groupby(["item_id","sale_ord_dt"])["before_prefr_amount"].agg({"sale_amount":np.sum}).reset_index()
    st = datetime.datetime.strptime(params["StartDate"],"%Y-%m-%d")
    item_daily["week"] = map(lambda v:math.ceil(((datetime.datetime.strptime(v,"%Y-%m-%d")-st).days + 1)/float(7)),item_daily["sale_ord_dt"])
    #item_weeknum = item_daily[item_daily["sale_amount"] > 0][["item_id","week"]].groupby("item_id")["week"].nunique().rename("weeknum").reset_index()
    item_weeknum = item_daily[item_daily["sale_amount"] > 0].groupby("item_id")["week"].agg({"weeknum":"nunique"}).reset_index()
    item_minwk = item_weeknum[item_weeknum["weeknum"] >= params["MinWeekswithSales"]]["item_id"]
    item_in_scope = pd.DataFrame(item_in_scope)
    item_in_scope = item_in_scope[item_in_scope["item_id"].isin(item_minwk)]

    # 只保留有ID并且订单量在[*MinBasketsPerYear*, *MaxBasketsPerYear*)内的顾客的订单数据
    customerlist = customer[ (customer["ord_qtty"] >= params["MinBasketsPerYear"]) & (customer["ord_qtty"] <= params["MaxBasketsPerYear"])]["user_id"]
    ord = ord[(ord["user_id"] != -1) & (ord["user_id"].isin(customerlist))]

    # 只保留销量大于0且折前销售额大于0的商品的订单数据
    ord = ord[(ord["sale_qtty"] > 0) & (ord["before_prefr_amount"] >0)]

    # 当*nonPromoOnly*==1时，只保留折后金额 >= 0.95 * 折后金额的订单数据
    if params["nonPromoOnly"] == 1:
        ord = ord[ord["after_prefr_amount"] >= 0.95 * ord["before_prefr_amount"]]

    ##############################
    ### numbered basket
    ##############################

    if params["basketByDay"] :
        rank = ord[["user_id","sale_ord_dt"]]
        rank = rank.drop_duplicates()
        rank = rank.sort_values(["user_id","sale_ord_dt"],ascending=[True,True])
        rank = rank.reset_index()
        rank["seqnum"] = rank.index+1
        rank = rank.drop("index",1)
        ord = ord.join(rank.set_index(["user_id","sale_ord_dt"]),on=["user_id","sale_ord_dt"],how="left")
    else :
        ord["sale_ord_tm"] = ord["sale_ord_tm"].map(lambda t:t[:-2])
        rank = ord[["user_id","sale_ord_tm","parent_sale_ord_id"]]
        rank = rank.drop_duplicates()
        rank = rank.sort_values(["user_id", "sale_ord_tm","parent_sale_ord_id"], ascending=[True, True,True])
        rank = rank.reset_index()
        rank["seqnum"] = rank.index + 1
        rank = rank.drop("index", 1)
        ord = ord.join(rank.set_index(["user_id","sale_ord_tm","parent_sale_ord_id"]),on=["user_id","sale_ord_tm","parent_sale_ord_id"],how="left")

    # 计算每位顾客在每个购物车中为每个item的消费的数量和金额 `A_010_TransProd_SeqNum
    #basket = ord[["user_id","seqnum","item_id","sale_qtty","before_prefr_amount"]].groupby(["user_id","seqnum","item_id"]).sum().reset_index()
    basket = ord.groupby(["user_id", "seqnum", "item_id"])["sale_qtty", "before_prefr_amount"].sum().reset_index()

    # 计算每个购物车的金额 `A_010b_TransTotals`
    basket_join = basket.groupby(["user_id","seqnum"])["before_prefr_amount"].agg({"basket_amount":"sum"}).reset_index()
    basket_join = basket_join.drop_duplicates()
    basket = basket.join(basket_join.set_index(["user_id","seqnum"]),on=["user_id","seqnum"],how="left")


    ##############################
    ### raw spendswitch
    ##############################

    # 计算每个购物车的销售额、购物车中每个商品的销量、销售额以及前一个购物车的销售额、前一个购物车中每个商品的销量、销售额 `A_030_TransProdFromToProd`
    from_basket = basket.copy(deep=True)
    from_basket["seqnum"] += 1
    from_basket.columns = from_basket.columns.map(lambda cn:"from_"+cn)
    from_basket = from_basket.rename(columns=({"from_user_id":"user_id","from_seqnum":"seqnum"}))
    basket2basket = basket.join(from_basket.set_index(["user_id","seqnum"]),on=["user_id","seqnum"],how="inner")

    # 按照概要中的公式计算替代性 `A_030b_TransProdFromToProd_SpendSwitch`
    basket2basket["spendswitch"] = (basket2basket["before_prefr_amount"]/basket2basket["basket_amount"]) * (basket2basket["from_before_prefr_amount"]/basket2basket["from_basket_amount"]) *(basket2basket["basket_amount"] + basket2basket["from_basket_amount"])

    # 对每对商品(A,B)进行汇总，计算行数、A商品的销售额、B商品的销售额、替代性总和 `B_010_ProdFromToProd_SpendSwitch`
    #spendswitch = basket2basket[["item_id","from_item_id","before_prefr_amount","from_before_prefr_amount","spendswitch"]].groupby(["item_id","from_item_id"]).sum().reset_index()
    spendswitch = basket2basket.groupby(["item_id", "from_item_id"])["before_prefr_amount", "from_before_prefr_amount", "spendswitch"].sum().reset_index()

    # 计算替代性大于零的的item
    item_switch = np.unique(spendswitch[spendswitch["spendswitch"] > 0]["item_id"])
    item_in_scope = item_in_scope[item_in_scope["item_id"].isin(item_switch)]

    ##############################
    ### 用户行为调整
    ##############################
    if params["adjustForEntropy"]:

        nb = len(item_in_scope)
        sales_per_item = basket.groupby(["user_id","item_id"])["before_prefr_amount"].agg({"sales":"sum"}).reset_index()
        sales_per_item = sales_per_item[sales_per_item["item_id"].isin(item_in_scope["item_id"])]

        # average sales over the scope for each user
        avg_sales_join = sales_per_item.groupby("user_id")["sales"].agg({"avg_sales":"sum"}).reset_index()
        avg_sales_join["avg_sales"] = avg_sales_join["avg_sales"]/nb
        sales_per_item = sales_per_item.join(avg_sales_join.set_index("user_id"),on="user_id",how="left")
        print sales_per_item.shape
        print sales_per_item.head()
        sales_per_item["item_score"] = (sales_per_item["sales"]/sales_per_item["avg_sales"]) * np.log(sales_per_item["sales"]/sales_per_item["avg_sales"])

        # normalized theil index
        theil_index = sales_per_item.groupby("user_id")["item_score"].agg({"t":"sum"}).reset_index()
        theil_index["t"] = theil_index["t"]/(nb*np.log(nb))

        # complementarity
        theil_index["comp_index"] = (1-2*theil_index["t"])/(2*(theil_index["t"]**2) - 2 * theil_index["t"] + 1)

        # within basket spendswitch
        from_basket = basket.copy(deep=True)
        from_basket.columns = from_basket.columns.map(lambda cn:"from_"+cn)
        from_basket = from_basket.rename(columns={"from_user_id":"user_id","from_seqnum":"seqnum"})
        within_basket = basket.join(from_basket.set_index(["user_id","seqnum"]),on=["user_id","seqnum"],how="inner")
        within_basket = within_basket[within_basket["item_id"] != within_basket["from_item_id"]]
        within_basket["spendswitch"] = (within_basket["before_prefr_amount"]/within_basket["basket_amount"]) *\
                                        (within_basket["from_before_prefr_amount"]/within_basket["from_basket_amount"]) * \
                                        (within_basket["basket_amount"] + within_basket["from_basket_amount"])
        within_spendswitch_ = within_basket[(within_basket["item_id"].isin(item_in_scope["item_id"])) & (within_basket["from_item_id"].isin(item_in_scope["item_id"]))]
        within_spendswitch = within_spendswitch_.groupby(["user_id","item_id","from_item_id"])["before_prefr_amount","from_before_prefr_amount","spendswitch"].sum().reset_index()

        # complementarity rate
        within_spendswitch_1 = within_spendswitch.join(theil_index.set_index("user_id"),on="user_id",how="inner")
        within_spendswitch_1["spend_comp_index"] = within_spendswitch_1["spendswitch"] * within_spendswitch_1["comp_index"]
        within_spendswitch_1["before_from"] = within_spendswitch_1["before_prefr_amount"] + within_spendswitch_1["from_before_prefr_amount"]
        within_spendswitch_2 = within_spendswitch_1.groupby(["item_id","from_item_id"])["spend_comp_index","before_from"].sum().reset_index()
        within_spendswitch_2["cr"] = within_spendswitch_2["spend_comp_index"]/within_spendswitch_2["before_from"]
        comp_rate = within_spendswitch_2[["item_id","from_item_id","cr"]]

        # complementarity adjust
        comp_rate["ind1"] = comp_rate["item_id"].apply(str) + " " + comp_rate["from_item_id"].apply(str)
        comp_rate = comp_rate[["ind1","cr"]]
        comp_rate = comp_rate.drop_duplicates("ind1")
        spendswitch["ind1"] = spendswitch["item_id"].apply(str) + " " + spendswitch["from_item_id"].apply(str)
        spendswitch = spendswitch.join(comp_rate.set_index("ind1"),on="ind1",how="left")
        spendswitch = spendswitch.drop("ind1",1)
        spendswitch.loc[spendswitch["cr"].isnull(),"cr"] = 0
        spendswitch["spendswitch_ca"] = spendswitch["spendswitch"] * (1 + spendswitch["cr"])
    else :
        spendswitch["cr"] = 0
        spendswitch["spendswitch_ca"] = spendswitch["spendswitch"]

    ##############################
    ### 互补性调整
    ##############################

    # 计算互补性指标 $$c= -(lift^2 - 1) / (lift^2 + 1) ,   lift = p(A,B)/(p(A)*p(B))$$ , 如果c<0, 则对替代性进行调整 spendswitch = spendswitch * (1 + c)  `D_050_compAdjustedTxnCountFromToAllProducts`
    basket_join = basket.groupby(["user_id","seqnum"])["item_id"].agg({"item_count":"count"}).reset_index()
    basket = basket.join(basket_join.set_index(["user_id","seqnum"]),on = ["user_id","seqnum"],how="left")
    basket_item_join = basket.groupby("item_id")["seqnum"].agg({"item_ord_count":"count"}).reset_index()
    basket = basket.join(basket_item_join.set_index("item_id"),on="item_id",how="left")
    basket_A = basket[basket["item_count"]>1][["user_id","seqnum","item_id","item_ord_count"]]
    basket_B = basket_A.copy(deep=True)
    basket_A = basket_A.rename(columns=({"item_id":"A_item_id","item_ord_count":"A_item_ord_count"}))
    basket_B = basket_B.rename(columns=({"item_id": "B_item_id","item_ord_count":"B_item_ord_count"}))
    co_basket = basket_A.join(basket_B.set_index(["user_id","seqnum"]),on=["user_id","seqnum"],how="inner")
    co_basket = co_basket[co_basket["A_item_id"] != co_basket["B_item_id"]]
    ci = co_basket.groupby(["A_item_id","B_item_id","A_item_ord_count","B_item_ord_count"])["user_id"].agg({"AB_count":"count"}).reset_index()
    ci = ci.rename(columns={"A_item_ord_count":"A_count","B_item_ord_count":"B_count"})
    total_count = np.unique(ord["parent_sale_ord_id"]).shape[0]
    ci["lift"] = ci["AB_count"] / ci["A_count"] / ci["B_count"] * total_count
    ci["ci"] = ( 1- ci["lift"] ** 2) / (1 + ci["lift"] ** 2)
    ci["ind1"] = ci["A_item_id"].apply(str) + " " + ci["B_item_id"].apply(str)
    spendswitch["ind1"] = spendswitch["item_id"].apply(str) + " " + spendswitch["from_item_id"].apply(str)
    ci = ci[["ind1","ci"]]
    ci = ci.drop_duplicates("ind1")
    spendswitch = spendswitch.join(ci.set_index("ind1"),on="ind1",how="left")
    spendswitch = spendswitch.drop("ind1",1)
    spendswitch.loc[spendswitch["ci"].isnull(),"ci"] = 0
    spendswitch.loc[spendswitch["ci"] > 0,"ci"] = 0
    spendswitch["spendswitch_ad"] = spendswitch["spendswitch_ca"] * (1 + spendswitch["ci"])

    ##############################
    ### switch probability
    ##############################

    # 筛选item范围
    spendswitch = spendswitch[(spendswitch["item_id"].isin(item_in_scope["item_id"])) & (spendswitch["from_item_id"].isin(item_in_scope["item_id"]))]

    # 把替代金额转换成替代概率 SwitchProb(A,B) = A替代B的金额 / 所有替代B的商品总金额 `D_060_nonPrunedSwitchingMatrix`
    spendswitch_join = spendswitch.groupby("from_item_id")["spendswitch_ad"].agg({"spendswitch_sum":"sum"}).reset_index()
    spendswitch = spendswitch.join(spendswitch_join.set_index("from_item_id"),on="from_item_id",how="left")
    spendswitch["switch_prob"] = spendswitch["spendswitch_ad"] / spendswitch["spendswitch_sum"]

    # save result
    switching_prob = pd.DataFrame(columns=["scope_id","src_item_id","dst_item_id","switching_prob","model"])
    switching_prob["src_item_id"] = spendswitch["from_item_id"]
    switching_prob["dst_item_id"] = spendswitch["item_id"]
    switching_prob["switching_prob"] = spendswitch["switch_prob"]
    switching_prob["scope_id"] = params["scope_id"]
    switching_prob["model"] = "order"
    switching_prob.to_csv(output_path+"/switching_prob.txt",sep="\t",index=False,header=False,encoding="utf8")

    spendswitch["scope_id"] = params["scope_id"]
    spendswitch["model"] = "order"
    spendswitch.to_csv(output_path+"/spendswitch.txt",sep="\t",index=False,header=False,encoding="utf8")

    #columns = ["scope_id", "item_first_cate_cd", "item_second_cate_cd", "item_third_cate_cd","self_pop", "scope_type","scope_desc","item_type"]
    scope = pd.DataFrame(columns=["id", "item_first_cate_cd", "item_second_cate_cd", "item_third_cate_cd","self_pop", "scope_type","scope_desc","item_type"])
    scope["id"] = ["1"]
    scope["id"] = params["scope_id"]
    scope["item_first_cate_cd"] = params["item_first_cate_cd"]
    scope["item_second_cate_cd"] = params["item_second_cate_cd"]
    scope["item_third_cate_cd"] = params['item_third_cate_cd']
    scope["self_pop"] = params["self_pop"]
    scope["scope_type"] = params["scope_type"]
    scope["scope_desc"] = params["scope_desc"]
    scope["item_type"] = params["item_type"]
    scope.to_csv(output_path+"/scope.txt",sep="\t",index=False,header=False,encoding="utf8")








































































































