# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 14:54:16 2017

@author: limen
"""

import yaml
import sys
reload(sys)
import subprocess
sys.setdefaultencoding('utf-8')
import item_fourth_cate as ifc
from attributes_cleaner.attrs_cleaner import *






def run(cid3,lvl='lvl3'):
    param_file_brand = 'params/brand_%s.yaml' % (cid3)
    param_file_sku = 'params/sku_%s.yaml' % (cid3)
    log_file = get_log_file(param_file_sku)
    open(log_file,'w').close()
    
    if lvl == 'lvl4':
        params_sku = yaml.load(file(param_file_sku,'r'))
        code = clean_attributes(params_sku)
        f = open(log_file,'w')
        if code != 'success':
            f.write('attributes cleaner failed! \n')
            sys.exit()
        else:
            f.write('attributes cleaner success! \n')
        f.close()
        pnames_sku = ifc.main(params_sku)
        params_brand = yaml.load(file(param_file_brand,'r'))
        pnames_brand = ifc.main(params_brand)
    else:
        pnames_sku = [param_file_sku]
        pnames_brand = [param_file_brand]
 
    
def concat_switching_file(pnames,log_file):
    p = yaml.load(file(pnames[0],'r'))
    if p['scope_type'] == 'lvl4':
        dst_dir = 'output/%s/%s' % (p['EndDate'],p['super_scope_id'])
    else:
        dst_dir = 'output/%s/%s' % (p['EndDate'],p['scope_id'])
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    dst_file = dst_dir + '/switching_prob.txt'
    open(dst_file,'w').close()
    for pname in pnames:
        p = yaml.load(file(pname,'r'))
        cmd = 'cat output/%s/%s/switching_prob.txt >> %s' % (p['EndDate'],p['scope_id'],dst_file)
        f =open(log_file,'a')
        status = subprocess.call(cmd,stdout=f,stderr=f,shell=True)
        if status == 0:
            f.write('%s done \n' % (cmd))
        else:
            f.write('%s failed \n' % (cmd))
        f.close()
        
def remove_switching_file(pnames,log_file):
    p = yaml.load(file(pnames[0],'r'))
    if p['scope_type'] == 'lvl4':
        dst_dir = 'output/%s/%s' % (p['EndDate'],p['super_scope_id'])
    else:
        dst_dir = 'output/%s/%s' % (p['EndDate'],p['scope_id'])
    dst_file = dst_dir + '/switching_prob.txt'
    if os.path.exist(dst_file):
        os.remove(dst_file)
        f = open(log_file,'a')
        f.write('%s removed \n' % (dst_file))
        f.close()
        









    #run sale summary
    run_command(['python','sale_summary.py',param_file_sku],log_file)
    
    #run switching
    for pname in pnames_sku + pnames_brand:
        run_command(['/software/servers/R-3.3.2-install/bin/Rscript', 'switchR.R', pname], log_file)
        
    #run halo
    run_command(['spark-submit', '--master', 'spark://172.19.142.130:7077',
                 '--deploy-mode', 'client', '--total-executor-cores',
                 '70', '--executor-memory', '10G','halo.py',param_file_sku],log_file)
    
    #run cdt,switching_prediction,sku_selection
    for pname in pnames_sku:
        try:
            run_command(['python','cdt.py',pname],log_file)
            run_command(['python','switching_prediction.py',pname],log_file)
            run_command(['python','utility.py',panme],log_file)
            run_command(['python','brand_transform.py',pname],log_file)
            run_command(['python','sku_selection_new.py',pname],log_file)
        except:
            f =open(log_file,'r')
            f.write(pname+'error')
            f.close()
            
    #run concat switching file
    if lvl == 'lvl4':
        concat_switching_file(pnames_sku,log_file)
        
    #run remove concated switching file
    if lvl == 'lvl4':
        remove_switching_file(pnames_sku,log_file)

def get_log_file(param_file):
    params = yaml.load(file(param_file, 'r'))
    return params['log_file']



def run_command(args,log_file = None):
    if log_file == None:
        log_file = get_log_file(args[-1])
    f = open(log_file, 'r')
    status = subprocess.call(args,stdout=f, stderr=f)
    f.close()
    return status







if __name__ == '__main__':
    n = len(sys.argv) - 1
    if n<1:
        print('Usage: \n python computation_pipe.py cid3 [lvl4] \n')
    else:
        cid3 = sys.agrv[1]
        lvl = sys.argv[2] or 'lvl3'
    run(cid3,lvl)
















