#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Date    : 2019-05-12 23:16:32
# Author  : Ziming Lee (lyctze1986@gmail.com)
# Version : $Id$

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyAnEn import AnEn



def main():
    # read all data
    filein_fct = os.path.join(os.path.dirname(homedir), 'data', 'NWP.csv')
    filein_obs = os.path.join(os.path.dirname(homedir), 'data', '1008A.csv')
    data_fct = pd.read_csv(filein_fct, index_col=['gtime', 'ltime'], parse_dates=['gtime'])
    data_obs = pd.read_csv(filein_obs, index_col=['vtime'], parse_dates=['vtime'])


    params = {
        'max_n_neighbours':50, 
        'weight_strategy':'total',
        'predict_name':'O3',
        'predictor_names':['O3', 't2', 'rh2', 'ws10', 'pblh'],
        'result_weight':'equal',
        'window_time':1,
        }

    anen_obj = AnEn(**params)

    gtime = datetime.datetime(2018,7,1,20)
    trainx, testx = anen_obj.splitTrainTest(data_fct, gtime=gtime, mark='beforeday', dNum=450)
    trainy = data_obs.loc[:gtime]

    anen_obj.fit(trainx, trainy)
    predict_data = anen_obj.predict(testx)
    print(predict_data.head())

    fileout_pre = os.path.join(
        os.path.dirname(homedir), 'data', 'predict.csv'
        )
    predict_data.to_csv(fileout_pre)



if __name__ == '__main__':
    homedir = os.path.dirname(os.path.realpath(__file__))
    print(homedir)
    main()