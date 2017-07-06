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
























