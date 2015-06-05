# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 21:04:57 2014

@author: deokwoo
"""
cur_key=data_dict['sensor_list'][2]
xval=data_dict[cur_key][2][0]
yval=data_dict[cur_key][2][1]
plot(xval,yval)
title(cur_key)
