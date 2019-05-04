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
	"""docstring for dataHandler"""
	def __init__(self, arg):
		super(dataHandler, self).__init__()
		self.arg = arg
