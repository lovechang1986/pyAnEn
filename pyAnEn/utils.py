#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author    : Ziming Lee(lyctze1986@gmail.com)
@Date      : 2018-07-05 14:40:06
@Modify    :
'''


import os
import datetime
import fractions
import itertools
import operator

import numpy as np
import pandas as pd


"""
定义AnEn需要的各种方法：

1. 权重生成方法
2. 高斯函数权重|对预测结果
3. 反函数权重|对预测结果
... 
"""


def wd_ws_2_u_v(wd, ws):
    tmp = (270.0 - wd) * np.pi / 180.0
    u = ws * np.cos(tmp)
    v = ws * np.sin(tmp)
    return u, v


def u_v_2_wd_ws(u, v):
    ws = np.sqrt(u * u + v * v)
    tmp = 270.0 - np.arctan2(v, u) * 180 / np.pi
    wd = np.fmod(tmp, 360.)
    return wd, ws


def makedir_exists(filein : str, exists_type='dir'):
    # create file directory
    if exists_type == 'dir':
        try:
            os.makedirs(filein, exist_ok=True)
        except:
            pass
    elif exists_type == 'file':
        try:
            os.makedirs(os.path.dirname(filein), exist_ok=True)
        except:
            pass


def combinations_with_replacement_counts(n: int, r: int) -> list:
    size = n + r - 1
    for indices in itertools.combinations(range(size), n - 1):
        starts = [0] + [index + 1 for index in indices]
        stops = indices + (size,)
        yield tuple(map(operator.sub, stops, starts))


def generate_weights(input_vars: list, weight_bin=0.1):
    """
    Generate all the weight combination! Like there were N box to X basket(X<=N).

    Usage：
        input_vars = ['t2', 'vis', 'msl', 'rh2']
        weight_bin = 0.1
        weight = gWeights(input_vars, weight_bin)
        print(weight)
    Output:
                  t2  vis  msl  rh2
        weight                     
        1_1_1_7  0.1  0.1  0.1  0.7
        1_1_2_6  0.1  0.1  0.2  0.6
        1_1_3_5  0.1  0.1  0.3  0.5
        1_1_4_4  0.1  0.1  0.4  0.4
        1_1_5_3  0.1  0.1  0.5  0.3
    """
    if len(input_vars) * weight_bin > 1:
        raise ValueError('')

    if len(input_vars) * weight_bin == 1:
        return pd.DataFrame(
            [weight_bin] * len(input_vars),
            columns=input_vars,
            index=[''.join(['W'] + ['_1' for _ in input_vars])]
        )

    # 第一步每个变量匹配一个基本权重
    normal_weight = [weight_bin for _ in input_vars]
    # how many balls
    weight_balls = 1 / weight_bin - len(input_vars)
    # how many balls in baskets
    combination_weight = list(combinations_with_replacement_counts(
        len(input_vars), int(weight_balls)))
    # sum normal weight and combination
    end_weight = np.array(normal_weight) + \
        np.array(combination_weight) * weight_bin
    # generate dataframe
    end_weight = pd.DataFrame(end_weight, columns=input_vars)
    weight_index = end_weight.div(weight_bin).round(0).astype(
        int).astype(str).apply('_'.join, axis=1)
    end_weight.index = weight_index
    end_weight.index.name = 'weight'

    return end_weight


def guassian_func(dist, a=1.5, b=0, c=0.3):
    return a * np.e ** (-(dist - b) ** 2 / (2 * c ** 2))


def inverse_func(dist, const=1e-10):
    return 1 / (dist + const)

