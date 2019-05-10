#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author    : Ziming Lee(lyctze1986@gmail.com)
@Date      : 2018-07-05 14:40:06
@Modify    :
'''

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyAnEn.verify import Verify


"""
本程序按照如下[最新AnEn文章]()进行可视化及相关检验
1. 原始观测数据的完整性检验
2. 单站点单起报时次的预报时效检验
3. K值对于RMSE的敏感性检验
4. K值对于R的敏感性检验
5. 运用CPRS检验针对所有站点的最优权重组合的箱线图
6. NWP与AnEn随预报时效的RMSE
7. NWP与AnEn随预报时效的R
7. NWP与AnEn随预报时效的BAS
8. AnEn与实况的空间RMSW
8. AnEn与实况的空间R
"""



class verifyTest(object):
    """docstring for verifyTest"""
    def __init__(self, arg):
        super(verifyTest, self).__init__()
        self.arg = arg

        


        




if __name__ == '__main__':
    homedir = os.path.dirname(os.path.realpath(__file__))
    datadir = os.path.join(
        os.path.dirname(homedir),
        'data'
        )

    print('homedir : ', homedir)
    print('datadir : ', datadir)


