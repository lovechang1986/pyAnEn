#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author    : Ziming Lee(lyctze1986@gmail.com)
@Date      : 2018-07-05 14:40:06
@Modify    :
'''

import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

homedir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(os.path.dirname(homedir)))

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


def read_single_file():
    """
    读取单一文件
    """
    pass


def check_valid_data():
    """
    检查数据的完整性
    """
    pass


def visual_valid_data_rate():
    # 数据完整性地图绘制
    pass


def plot_single_generate_forecast():
    # 绘制单一起报时次的NWP、AnEn和Obs
    pass


def RMSE_sensitive_for_K():
    # 绘制RMSE的K值敏感性
    pass


def R_sensitive_for_K():
    # 绘制R的K值敏感性
    pass


def CPRS_boxes_best_weight():
    # 利用CPRS评分最优权重的组合箱线图
    pass


def RMSE_NWP_ANEN_along_leadtime():
    # NWP和AnEn的RMSE随预报时效的变化
    pass


def R_NWP_ANEN_along_leadtime():
    # NWP和AnEn的R随预报时效的变化
    pass


def BAS_NWP_ANEN_along_leadtime():
    # NWP和AnEn的BAS随预报时效的变化
    pass


def RMSE_space_AnEn_obs():
    # RMSE空间分布
    pass


def R_space_AnEn_obs():
    # R空间分布
    pass


if __name__ == '__main__':
    homedir = os.path.dirname(os.path.realpath(__file__))
    datadir = os.path.join(
        os.path.dirname(homedir), 'data'
    )

    print('homedir : ', homedir)
    print('datadir : ', datadir)
