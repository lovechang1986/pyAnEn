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
from cartopy.io.shapereader import Reader

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


def read_single_file(filein, sel_weight=None):
    """
    读取单一文件:由于产生的文件比较规整，因此不需要考虑太多的错误处理
    """
    if not os.path.exists(filein):
        return False, None

    data = pd.read_csv(filein)
    if sel_weight is None:
        return data
    return data.query('weights == @sel_weight')


def check_valid_data(check_variable_name=None, filedir=None, filein_stainfo=None):
    """
    检查数据的完整性
    """
    if check_valid_data is None:
        check_variable_name = 'O3'
    stations = pd.read_csv(filein_stainfo)['sta']
    # 第一步读取实况观测数据
    total_data = []
    for ista, vsta in enumerate(stations):
        filein_tmp = os.path.join(filedir, f'{vsta}.csv')
        total_data.append(pd.read_csv(filein_tmp))
    total_data = pd.concat(total_data)
    # 第二步将数据转换为浮点(考虑无论何种污染物的情况都是一样的)
    total_data[check_variable_name] = pd.to_numeric(
        total_data[check_variable_name], errors='coerce')

    def check_function(x): return 1 - (x.isnull().sum() / x.count())
    df_total_data = total_data.groupby('sta').apply(check_function) * 100
    # 第三步对数据使用isnull来处理检验完整性(groupby station)
    fileout = os.path.join(
        homedir, 'output', 'check_valid_data_{check_variable_name}.csv')
    if not os.path.exists(os.path.dirname(fileout)):
        os.makedirs(os.path.dirname(fileout), exist_ok=True)
    df_total_data.name['rate']
    df_total_data.to_csv(fileout, float_format='%.2f')
    return


def visual_valid_data_rate(check_variable_name=None, filein_stainfo=None, valid_rate=90):
    # 数据完整性地图绘制
    fileout = os.path.join(
        homedir, 'output', f'check_valid_data_{check_variable_name}.csv')
    data = pd.read_csv(fileout)
    stainfo = pd.read_csv(filein_stainfo)[['sta', 'lon', 'lat']]
    data = data.merge(stainfo, on='sta')
    print(data.head())
    '''
    data should be:
    sta     lon     lat   rate
    54511   116.47    39.8   99.62
    '''
    # 第一步创建地图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    # 第二步读取全国shape
    shp_file = os.path.join(homedir, 'shp', 'bou2_4p.shp')
    ax.add_geometries(Reader(shp_file).geometries(),
                      ccrs.PlateCarree(),
                      facecolor='none')
    # 第三步绘制红色数据
    red_logical = data.rate >= 90
    blue_logical = ~red_logical
    data[['lon', 'lat', 'rate']].iloc[red_logical].plot.scatter(
        ax=ax, x='lon', y='lat', s=20, c='r', alpha=0.75)
    data[['lon', 'lat', 'rate']].iloc[blue_logical].plot.scatter(
        ax=ax, x='lon', y='lat', s=20, c='DarkBlue', alpha=0.75)
    # 调整绘图的其他参数
    # 加入虚线网格
    # 加入经纬度坐标label
    plt.tight_layout()
    # 输出到文件
    fileout_format = 'png'  # tif?
    fileout = os.path.join(
        homedir, 'output', f'check_valid_data_{check_variable_name}.{fileout_format}')
    if not os.path.exists(os.path.dirname(fileout)):
        os.makedirs(os.path.dirname(fileout), exist_ok=True)
    plt.savefig(fileout, dpi=300)
    return


def plot_single_generate_forecast(staid=None, filedir=None, gtime=None, variable_name=None):
    # 绘制单一起报时次的NWP、AnEn和Obs
    # 第一步获取输入的文件
    filein = os.path.join(filedir, f'staid', gtime.strftime('%Y%m%d%H.csv'))
    data = pd.read_csv(filein, parse_dates=['gtime', 'vtime'])
    print(data.head())
    # 第二步根据预报时效获取实况数据
    data_obs = data['O']
    # 第三步计算AnEn的平均值及其groupby leadtime的标准差
    data_anen = data[[f'K_{i}' for i in range(5, 45)]]
    # 第四部获取AnEn的平均值和fill_between,正负2个标准差？
    data_anen_std = data_anen.std(axis=0, ddof=1)
    data_anen_std_up = data_anen_std * 2
    data_anen_std_down = - data_anen_std * 2
    data_anen_mean = data_anen.mean(axis=0)
    # 第五步获取化学模式的结果
    data_nwp = data['P']
    # 第六步绘制
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    ax.plot(data_obs, 'r-', lw=2)
    ax.plot(data_anen_mean, 'g--', lw=2)
    ax.fill_between(x=range(len(data_obs)), data_anen_std_down, data_anen_std_up)
    ax.plot(data_nwp, 'b-', lw=2)
    # 第七步调整绘图的参数
    # add grid
    # label and so on....
    plt.tight_layout()
    # 第八步输出到文件
    fileout_format = 'png'  # tif?
    fileout = os.path.join(
        homedir, 'output', f'single_timeseries_{staid}_{variable_name}_{gtime.strftime("%Y%m%d%H")}.{fileout_format}')
    if not os.path.exists(os.path.dirname(fileout)):
        os.makedirs(os.path.dirname(fileout), exist_ok=True)
    plt.savefig(fileout, dpi=300)
    return


def RMSE_sensitive_for_K(filedir=None, filein_stainfo=None, end_gtime=None, reload_data=True):
    # 绘制RMSE的K值敏感性
    # 选择等权重情况下的K值敏感性进行检验
    # 第一步读取等权重下所有训练期间的数据
    fileout = os.path.join(
        homedir, 'output', f'sensitive_{os.path.basename(filedir)}_RMSE.csv')
    if not os.path.exists(os.path.dirname(fileout)):
        os.makedirs(os.path.dirname(fileout), exist_ok=True)
    if reload_data:
        stainfo = pd.read_csv(filein_stainfo)
        total_data = []
        for ista, vsta in enumerate(stainfo.sta):
            print('Now read this station : ', vsta)
            filedir_in = os.path.join(filedir, f'{vsta}')
            total_filenames = os.listdir(filedir_in)
            total_datetimes = pd.to_datetime(
                total_filenames, format='%Y%m%d%H.csv')
            read_filenames = total_filenames[total_datetimes <= end_gtime]
            for iname, name in enumerate(read_filenames):
                filein = os.path.join(filedir_in, name)
                tmpdata = read_single_file(filein, '2_2_2_2_2')
                if tmpdata is not None:
                    total_data.append(tmpdata)
        total_data = pd.concat(total_data)
        if total_data.empty:
            raise Exception('Error occued in reading data!')

        # 第二步计算不同K值得RMSE
        verify_obj = Verify()
        k_start = 5
        k_end = 50
        K_range = range(k_start, k_end + 1)
        K_rmse = []
        for ik, vk in enumerate(K_range):
            K_rmse.append(
                verify_obj.calc_RMSE(total_data[f'K_{vk}'], total_data['O'])
            )
        data = pd.DataFrame({'K_rmse': K_rmse})
        data.to_csv(fileout, float_format='%.2f')
    else:
        data = pd.read_csv(fileout)
    # 第三步绘制图形
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(data, 'k-', lw=2.5)

    # add grid
    # add label

    plt.tight_layout()
    # 第四步保存
    fileout_format = 'png'  # tif?
    fileout = os.path.join(
        homedir, 'output', f'sensitive_{os.path.basename(filedir)}_RMSE.{fileout_format}')
    if not os.path.exists(os.path.dirname(fileout)):
        os.makedirs(os.path.dirname(fileout), exist_ok=True)
    plt.savefig(fileout, dpi=300)


def R_sensitive_for_K():
    # 绘制R的K值敏感性
     # 选择等权重情况下的K值敏感性进行检验
    fileout = os.path.join(
        homedir, 'output', f'sensitive_{os.path.basename(filedir)}_R.csv')
    if not os.path.exists(os.path.dirname(fileout)):
        os.makedirs(os.path.dirname(fileout), exist_ok=True)
    if reload_data:
        stainfo = pd.read_csv(filein_stainfo)
        total_data = []
        for ista, vsta in enumerate(stainfo.sta):
            print('Now read this station : ', vsta)
            filedir_in = os.path.join(filedir, f'{vsta}')
            total_filenames = os.listdir(filedir_in)
            total_datetimes = pd.to_datetime(
                total_filenames, format='%Y%m%d%H.csv')
            read_filenames = total_filenames[total_datetimes <= end_gtime]
            for iname, name in enumerate(read_filenames):
                filein = os.path.join(filedir_in, name)
                tmpdata = read_single_file(filein, '2_2_2_2_2')
                if tmpdata is not None:
                    total_data.append(tmpdata)
        total_data = pd.concat(total_data)
        if total_data.empty:
            raise Exception('Error occued in reading data!')

        # 第二步计算不同K值得RMSE
        verify_obj = Verify()
        k_start = 5
        k_end = 50
        K_range = range(k_start, k_end + 1)
        K_rmse = []
        for ik, vk in enumerate(K_range):
            K_rmse.append(
                verify_obj.calc_R(total_data[f'K_{vk}'], total_data['O'])
            )
        data = pd.DataFrame({'K_rmse': K_rmse})
        data.to_csv(fileout, float_format='%.2f')
    else:
        data = pd.read_csv(fileout)
    # 第三步绘制图形
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(data, 'k-', lw=2.5)

    # add grid
    # add label

    plt.tight_layout()
    # 第四步保存
    fileout_format = 'png'  # tif?
    fileout = os.path.join(
        homedir, 'output', f'sensitive_{os.path.basename(filedir)}_R.{fileout_format}')
    if not os.path.exists(os.path.dirname(fileout)):
        os.makedirs(os.path.dirname(fileout), exist_ok=True)
    plt.savefig(fileout, dpi=300)


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
