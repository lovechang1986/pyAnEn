#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author    : Ziming Lee(lyctze1986@gmail.com)
@Date      : 2018-07-05 14:40:06
@Modify    :
'''

import gc
import collections
import datetime
from collections import Iterable
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from pyAnEn.utils import gWeights, guassianFunc, inverseFunc
from .datahandler import dataHandler

"""
定义AnEn类：

1. 自定义AnEn类，利用pandas实现
2. 
3. 利用sklearn的KNN类实现AnEn
"""


class AnEn(dataHandler):
    """
    该类主要构件AnEn的算法模型：
    1. 接收训练数据
    2. 接收测试数据并根据模型预测
    3. 采用类似sklearn的方式进行设计
    4. 仅仅设计出fit和predict，其他数据的处理则交给其他函数，这里仅仅设计模型
    """

    def __init__(self,
                 max_n_neighbours,
                 weight_strategy,  # 'equal', 'all', 'selfdefine'
                 predict_name,  # 预报变量
                 predictor_names=None,   # 自变量
                 predictor_min_weight=0.1,  # 最小权重
                 predictor_weights=None,  # 自定义权重，列表
                 result_weight='equal',  # equal or guassian or inverse
                 ):
        super(AnEn, self).__init__()

        self.max_n_neighbours = max_n_neighbours
        # 初始化权重
        initWeights(weight_strategy, predictor_names,
                    predictor_min_weight, predictor_weights)

    def initWeights(self, weight_strategy, predictor_names, predictor_min_weight, predictor_weights):
        if weight_strategy == 'equal':
            self.weights = pd.DataFrame(
                np.ones(len(predictor_names)), columns=predictor_names)
        elif weight_strategy == 'weight':
            if not isinstance(predictor_min_weight, float):
                raise ValueError(
                    'predictor_min_weight must float letter than 1!')
            self.weights = gWeights(predictor_names, predictor_min_weight)
        elif weight_strategy == 'selfdefine':
            self.weights = pd.DataFrame(
                predictor_weights, columns=predictor_names)
        else:
            raise ValueError(
                'weight_strategy must be in ["equal", "weight", "selfdefine"]')

    def fit(slef, train_X, train_Y):
        self.train_X = train_X
        self.train_Y = train_Y

    def predict(self, test_X):
        "预报测试"

        # calc delta
        delta_data = self._calcDelta(self.train_X, test_X)
        # calc weight delta
        weight_delta_data = self._calcWeightDelta(delta_data)

        total_distance = []
        test_ltimes = np.unique(test_X.index.get_level_values('ltime'))
        for iltime, vltime in enumerate(test_ltimes):
            tmp_data = self.searchWindow(weight_delta_data, vltime)
            # TODO 增加更多选项，不仅限于-1,0,1，这样的零对称等差数列；还应可以自定义，例如[-4, -1, 1, 2]，0对应的时次则为当前的vltime
            tmp_distance = (
                tmp_data
                .groupby('weight')
                .apply(self._calcDistance, gtime=gtime)
                .to_frame(name='distance')
                .reset_index()
            )
            tmp_distance['gtime'] += pd.to_timedelta(vltime, 'h')
            tmp_distance['ltime'] = vltime
            total_distance.append(tmp_distance)

        total_distance = pd.concat(total_data)

        predict_data = (
            total_distance
            .groupby(['weight', 'ltime'])
            .apply(self._calcNeighbour)
        )

        valid_date = pd.to_datetime(
            gtime) + pd.to_timedelta(predict_data.index.get_level_values('ltime'), 'h')
        predict_data['O'] = train_Y.reindex(observer_date.values).values
        predict_data['P'] = test_X[self.predict_varname].reset_index(
            ['gtime'], drop=True).reindex(predict_data.index.get_level_values('ltime')).values
        predict_data['gtime'] = gtime

        return predict_data


    def _addHour(self, row, ltime):
        return row + np.timedelta64(ltime, 'h')

    def _calcNeighbour(self, row):
        neighbour_data = self.Odata[self.predict_varname].reindex(row.gtime)
        if self.result_weight == 'equal':
            return pd.Series(neighbour_data.expanding(1).mean().values, index=[f'K_{i}' for i in range(self.max_n_neighbours)])
        else:
            if self.result_weight == 'guassian':
                distance_func = guassianFunc
            elif self.result_weight == 'inverse':
                distance_func = inverseFunc
            gas_w = distance_func(row.dis).values
            output = [
                np.nansum(neighbour_data.values[:i]
                          * gas_w[:i] / gas_w[:i].sum())
                for i in range(2, self.max_n_neighbours)]
            return pd.Series(output, index=[f'K_{i}' for i in range(2, self.max_n_neighbours)])

    def _addForecastTime(self, data):
        data['ftime'] = data.index.get_level_values(
            'gtime') + pd.to_timedelta(data.index.get_level_values('ltime'), 'h')
        return data.set_index('ftime', append=True)

    def _calcDelta(self, traindata, testdata):
        print(traindata.head())
        print(testdata.head())
        return (
            traindata
            .sub(testdata.reset_index('gtime', drop=True), axis=0, level='ltime')
            .div(traindata.std(ddof=1), axis=1)
        )

    def _calcWeightDelta(self, d1):
        idx1, idx2, midx = generate_dif_index(d1.index, self.weights.index)
        return pd.DataFrame(
            (d1.values.reshape(-1, 1, len(self.input_varname))
             * self.weights.values.reshape(1, -1, len(self.input_varname))).reshape(
                -1, len(self.input_varname)),
            columns=self.input_varname,
            index=midx) ** 2

    def _calcDistance(self, groupdata, gtime):
        return pd.Series(groupdata
                         .loc[pd.IndexSlice[:, :, :, :gtime], :]
                         .sum(axis=0, level='gtime')
                         .apply(np.sqrt)
                         .sum(axis=1)
                         .sort_values()[:self.max_n_neighbours]
                         )

    def cartesianProduce(self, d1, d2):
        id1 = [i for i in d1 for _ in d2]
        id2 = [i for _ in d1 for i in d2]
        return id1, id2

    def generateMultiIndex(self, index1, index2):
        s_index1 = index1.shape[0]
        s_index2 = index2.shape[0]
        idx1, idx2 = self.cartesianProduce(range(s_index1), range(s_index2))
        new_index1 = np.array(index1)[idx1]
        new_index2 = np.array(index2)[idx2]
        both_index = np.column_stack(
            [new_index1.tolist(), new_index2.tolist()])
        return new_index1, new_index2, pd.MultiIndex.from_arrays(both_index.T, names=list(index1.names) + list(index2.names))


class AnEnSklearn(object):
    """
    # TODO
    使用sklearn实现AnEn，有以下两种方式：
    1. 第一种方式略显复杂，我暂时不想写出来，后面再补充吧。
    2. 对输入数据进行设计，并且传递给sklearn的KNN类一个matrix函数用于自定义计算距离的方式
    """

    def __init__(self, arg):
        super(AnEnSklearn, self).__init__()
        self.arg = arg
