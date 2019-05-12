#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author    : Ziming Lee(lyctze1986@gmail.com)
@Date      : 2018-07-05 14:40:06
@Modify    :
'''

import datetime

import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs



"""
定义数据处理的类：
TODO. 1. 从grads二进制数据到站点数据
TODO. 2. 从netcdf数据到站点数据
3. 针对AnEn数据获取的操作
"""



class grads2Station(object):
    """docstring for grads2Station"""
    def __init__(self, arg):
        super(grads2Station, self).__init__()
        self.arg = arg
        


class NC2Station(object):
    """docstring for NC2Station"""
    def __init__(self, arg):
        super(NC2Station, self).__init__()
        self.arg = arg
        



class dataHandler(object):
    """
    本函数对预报数据做操作：
    1. 获取窗口数据
    2. 区分训练和预测数据
        1. 按照某个时间开始 params : start
        2. 按照预测起报时次当天历史上前后N天内的数据 params : limit_between
        3. 按照预测起报时次当天前的N天数据 params : before_days
        4. 所有的训练数据 params : all
    """
    def __init__(self, window_time=None):
        super(dataHandler, self).__init__()
        if window_time is None:
            window_time = 1
        self.window_time=window_time

    def searchWindow(self, data, ltime):
        # TODO 增加更多选项，不仅限于-1,0,1，这样的零对称等差数列；还应可以自定义，例如[-4, -1, 1, 2]，0对应的时次则为当前的vltime
        idx = pd.IndexSlice
        window_range = np.arange(-self.window_time,
                                 self.window_time + 1, 1) + ltime
        return data.loc[idx[:, window_range], :]

    def getLogicalLimit(self, gtime, gtimes, limit_day):
        days = np.abs(gtimes.dayofyear - pd.to_datetime(gtime).dayofyear)
        return days <= limit_day

    def splitTrainTest(self, data, gtime=None, mark='all', dNum=None, start=None, limit_day=45):
        if gtime is None:
            raise ('Params gtime is None!')
        if mark == 'beforeday':
            if dNum is None or gtime is None:
                raise (
                    'Must set a num to dNum and a datetime to gtime for select data!')
            elif isinstance(dNum, int):
                start = gtime + datetime.timedelta(days=-dNum - 1)
                end = gtime + datetime.timedelta(hours=-1)
                return data.loc[pd.IndexSlice[start:end, :], :], data.loc[pd.IndexSlice[gtime,:],:]
        if mark == 'betweenday':
            logical_mark = self.get_logical_limit(
                gtime, data.index.get_level_values('gtime'), limit_day
                )
            if start is None:
                raise ValueError('Parameter start must not None!')
            else:
                end = gtime + datetime.timedelta(hours=-1)
                return data.iloc[logical_mark].loc[pd.IndexSlice[start:end, :], :], data.loc[pd.IndexSlice[gtime,:],:]
        elif mark == 'total':
            start = datetime.datetime(2016,6,1)
            end = gtime + datetime.timedelta(hours=-1)
            return data.loc[pd.IndexSlice[start:end,:],:], data.loc[pd.IndexSlice[gtime,:],:]
        elif mark == 'startday':
            if start is None:
                raise ValueError('Parameter start must not None!')
            else:
                end = gtime + datetime.timedelta(hours=-1)
                return data.loc[pd.IndexSlice[start:end, :], :], data.loc[pd.IndexSlice[gtime,:],:]
        else:
            raise ValueError(
                'Unable to understand parameters mark = {}, must in ["total", "startday", "beforeday", "betweenday"]'.format(mark))
