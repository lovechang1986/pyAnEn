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
from sklearn.metrics import confusion_matrix


"""
定义检验类：

1. RMSE
2. R
3. MAE
4. CRPS
"""



class Verify(object):
	"""
	
	Usage:
		verify_obj = Verify()
		real_value = np.random.randint(0, 10, 100)
		predict_value = np.random.randint(0, 10, 100)
		print(verify_obj._calc_RMSE(real_value, predict_value))
		print(verify_obj._calc_MAE(real_value, predict_value))
		print(verify_obj._calc_ME(real_value, predict_value))
		print(verify_obj._calc_R(real_value, predict_value))
		print(verify_obj._calc_TS(real_value, predict_value, [2, 5, 7, 9]))
		print(verify_obj._calc_PO(real_value, predict_value, [3, 5, 7, 9]))
		print(verify_obj._calc_FAR(real_value, predict_value, [2, 5, 7, 9]))
		print(verify_obj._calc_BS(real_value, predict_value, [2, 5, 7, 9]))
	"""
    def __init__(self):
        super(Verify, self).__init__()
        self.confusion_matrix = None
        self.NA = None
        self.NB = None
        self.NC = None

    def _get_level_verify(self, real_value, predict_value, levels):
        self.__init__()
        real_level, predict_level = self._check_levels(
            real_value, predict_value, levels)
        self.confusion_matrix = self._get_confusion_matric(
            real_level, predict_level)
        if self.NA is None or self.NB is None or self.NC is None:
            self.NA = {}
            self.NB = {}
            self.NC = {}
        for ivalue, vvalue in enumerate(self.confusion_matrix.columns):
            self.NA[vvalue] = self.confusion_matrix.ix[vvalue, vvalue]
            self.NB[vvalue] = self.confusion_matrix.ix[slice(
                None), vvalue].sum()
            self.NC[vvalue] = self.confusion_matrix.ix[vvalue,
                                                       slice(None)].sum()

    def _check_and_remove_nan(self, real_value, predict_value):
        logical_nan = ~np.logical_or(
            np.isnan(predict_value), np.isnan(real_value))
        predict_value = predict_value[logical_nan]
        real_value = real_value[logical_nan]
        return real_value, predict_value

    def _get_confusion_matric(self, real_level, predict_level):
        return pd.DataFrame(confusion_matrix(real_level, predict_level),
                            index=np.unique([real_level, predict_level]), columns=np.unique([real_level, predict_level]))

    def _check_levels(self, real_value, predict_value, levels):
        '''
        引用https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.digitize.html
        使用numpy.digitize 函数
        '''
        real_value, predict_value = self._check_and_remove_nan(
            real_value, predict_value)
        predict_level = np.digitize(predict_value, levels)
        real_level = np.digitize(real_value, levels)

        return real_level, predict_level

    def _calc_RMSE(self, real_value, predict_value):
        real_value, predict_value = self._check_and_remove_nan(
            real_value, predict_value)
        return np.sqrt(np.mean((predict_value - real_value) ** 2))

    def _calc_MAE(self, real_value, predict_value):
        real_value, predict_value = self._check_and_remove_nan(
            real_value, predict_value)
        return np.mean(np.abs(predict_value - real_value))

    def _calc_ME(self, real_value, predict_value):
        real_value, predict_value = self._check_and_remove_nan(
            real_value, predict_value)
        return np.mean(predict_value - real_value)

    def _calc_R(self, real_value, predict_value):
        real_value, predict_value = self._check_and_remove_nan(
            real_value, predict_value)
        return np.corrcoef(real_value, predict_value)[1, 0]

    def _calc_TS(self, real_value, predict_value, levels):
        '''
        根据等级计算TS评分,即等级预报的准确率
        TS = True / True + Flase Alarm + True negetive
        '''
        self._get_level_verify(real_value, predict_value, levels)
        print(self.confusion_matrix)
        return (pd.Series(self.NA) / (pd.Series(self.NB) + pd.Series(self.NA) + pd.Series(self.NC))).to_json()

    def _calc_PO(self, real_value, predict_value, levels):
        '''
        PO = True/ True + True negetive
        '''
        print(self.confusion_matrix)
        self._get_level_verify(real_value, predict_value, levels)
        print(self.confusion_matrix)
        return (pd.Series(self.NC) / (pd.Series(self.NA) + pd.Series(self.NC))).to_json()

    def _calc_FAR(self, real_value, predict_value, levels):
        '''
        FAR = False Alarm / True + True negetive
        '''
        self._get_level_verify(real_value, predict_value, levels)
        return (pd.Series(self.NB) / (pd.Series(self.NB) + pd.Series(self.NA))).to_json()

    def _calc_BS(self, real_value, predict_value, levels):
        '''
        BS = True + False Alarm / True + True negetive
        '''
        self._get_level_verify(real_value, predict_value, levels)
        return ((pd.Series(self.NA) + pd.Series(self.NB)) / (pd.Series(self.NA) + pd.Series(self.NC))).to_json()

'''
# 方法调试示例

'''