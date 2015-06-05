#!/usr/bin/python
# To force float point division
from __future__ import division
"""
@author: NGO Quang Minh Khiem
@contact: khiem.ngo@adsc.com.sg

"""
import sys
import urllib
import urllib2

airport_codes = {'SDH' : {'code' : 'KOAK', 'city': 'Berkeley', 'state' : 'CA', 'statename' : 'California'},
                 'VTT' : {'code' : 'EFHF', 'city': 'Espoo', 'state' : '', 'statename' : 'Finland'},
                 'GValley' : {'code' : 'RKSS', 'city' : 'Seoul', 'state': '', 'statename' : 'South Korea'},
                 'SG' : {'code' : 'WSSS', 'city' : 'Singapore', 'state': '', 'statename' : 'Singapore'},
                 }
view_type = {'d' : 'DailyHistory', 'w' : 'WeeklyHistory', 'm': 'MonthlyHistory', 'custom' : "CustomHistory"}
URL_PREFIX = 'http://www.wunderground.com/history/airport/'

############
# construct the request url, based on the locations,
# time period and history view type
############
def construct_url(site_code, sy, sm, sd, view='d', ey=2014, em=12, ed=31):
    airport_code = airport_codes[site_code]['code']
    city = airport_codes[site_code]['city']
    state = city = airport_codes[site_code]['state']
    statename = airport_codes[site_code]['statename']
    
    request_option = 'DailyHistory.html'
    if view_type.has_key(view):
        request_option = view_type[view] + '.html'
        
    
    url = URL_PREFIX + airport_code + '/' + str(sy) + '/' + str(sm) + '/' + str(sd) + '/' + request_option
    # print url
    values = {}
    values['req_city'] = city
    values['req_state'] = state
    values['req_statename'] = statename
    values['format'] = 1  # ## data will be returned in CSV format
    
    if view == 'custom':
        values['yearend'] = ey
        values['monthend'] = em
        values['dayend'] = ed
        
    # print values
    return url, values

############
# Retrieve the weather data, given the site code,
# the history view type, and the time period
# Return the data from server (text), in CSV format

# site_code: SDH, VTT, GValley, SG
# view: history view type: 'd' (day), 'w' (week), 'm' (month), 'custom'
# sy, sm, sd: start year/month/day
# view='d': retrieve hourly weather data during the day sy/sm/sd
# view='w': retrieve daily weather data during the week of sy/sm/sd
# view='m': retrieve daily weather data during the month of sy/sm/sd
#
# view='custom': retrieve daily weather data from sy/sm/sd to ey/em/ed
# if view='custom': the parameters ey,em,ed should be specified
############
def retrieve_data(site_code, sy, sm, sd, view='d', ey=2014, em=12, ed=31):

    # ## construct url based on the parameters    
    url, values = construct_url(site_code, sy, sm, sd, view, ey, em, ed)
    req_data = urllib.urlencode(values)
    
    # ## send request to the server and get response
    req = urllib2.Request(url, req_data)
    response = urllib2.urlopen(req)
    data = response.read()
    
    # ## pre-process the data returned from server    
    data = data.strip()
    data = data.replace('<br />', '')
    
    return data
