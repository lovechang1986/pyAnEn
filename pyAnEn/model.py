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
import properscoring as ps
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from pyAnEn.utis import gWeights
from pyAnEn.datahandler import dataHandler

"""
定义AnEn类：

1. 自定义AnEn类，利用pandas实现
2. 
3. 利用sklearn的KNN类实现AnEn
"""


class AnEn(object):
	"""docstring for AnEn"""
	def __init__(self, arg):
		super(AnEn, self).__init__()
		self.arg = arg


class AnEnSklearn(object):
	"""docstring for AnEnSklearn"""
	def __init__(self, arg):
		super(AnEnSklearn, self).__init__()
		self.arg = arg
		