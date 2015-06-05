#!/adsc/DDEA_PROTO/bin/python

"""
@author: NGO Quang Minh Khiem
@contact: khiem.ngo@adsc.com.sg

"""

import urllib
import urllib2
from datetime import *

from pathos.multiprocessing import ProcessingPool
import pathos.multiprocessing as pmp
from toolset import dill_save_obj

airport_codes = {
                'SDH' : {'code' : 'KOAK', 'city': 'Berkeley', 'state' : 'CA', 'statename' : 'California'},
                'VTT' : {'code' : 'EFHF', 'city': 'Espoo', 'state' : '', 'statename' : 'Finland'},
                'GValley' : {'code' : 'RKSS', 'city' : 'Seoul', 'state': '', 'statename' : 'South Korea'},
                'SG' : {'code' : 'WSSS', 'city' : 'Singapore', 'state': '', 'statename' : 'Singapore'}
                }

save_path = {
            'SDH':'/adsc/bigdata/input_data/sdh/weather/',
            'VTT':'/adsc/bigdata/input_data/VTT/data_year/weather/',
            'GValley':'/adsc/bigdata/input_data/gvalley/weather/',
            'SG':'/adsc/bigdata/input_data/sg/weather/',
            }

site_prefix = {
            'SDH':'sdh_',
            'VTT':'VTT_',
            'GValley':'gvalley_',
            'SG':'sg_'
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


def construct_filepath(sitecode, sy, sm, sd):
    path = save_path[sitecode] + site_prefix[sitecode] + '{:04d}_{:02d}_{:02d}'.format(sy, sm, sd) + '.bin'
    return path

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

def retrieve_data_package(site_code, sy, sm, sd, view='d', ey=2014, em=12, ed=31):
    # ## construct url based on the parameters
    url, values = construct_url(site_code, sy, sm, sd, view, ey, em, ed)
    path = construct_filepath(site_code, sy, sm, sd)
    return [url, values, path]


def retrieve_data(target):
    url, values, path = target
    req_data = urllib.urlencode(values)

    # ## send request to the server and get response
    req = urllib2.Request(url, req_data)
    response = urllib2.urlopen(req)
    data = response.read()

    if not data:
        return

    # ## pre-process the data returned from server    
    data = data.strip()
    data = data.replace('<br />', '')

    if data:
        dill_save_obj(data, path)

    #return data


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)


"""
    The following section demonstrates how to use the retrieve_data() method,
    with different history view types
        
"""
   
""" 
View type: Day
granularity support: hourly
Ex: Retrieve hourly weather data on the date Jan 8th, 2014
"""
#data = retrieve_data('VTT', 2014, 1, 8, view='d')
#print data

"""
View type: Week
granularity support: daily 
Ex: Retrieve daily weather data during the week of Jan 8th, 2014
"""
# data = retrieve_data('VTT', 2014, 1, 8, view='w')
# print data

"""
View type: Month
granularity support: daily 
Ex: Retrieve daily weather data during the month of Jan 8th, 2014
"""
# data = retrieve_data('VTT', 2014, 1, 8, view='m')
# print data

""" 
View type: Custom
granularity support: daily
Ex: Retrieve daily weather data from Jan 1, 2014 to Jan 7, 2014
"""
# data = retrieve_data('VTT', 2014, 1, 1, view='custom', ey=2014, em=1, ed=7)
# print data

if __name__ == '__main__':

    target = date.today() - timedelta(days=1)
    y = target.year
    m = target.month
    d = target.day
    ey = y + 1

    weather_points = list()
    weather_points.append(retrieve_data_package('SDH',      y, m, d, view='d', ey=ey, em=12, ed=31))
    weather_points.append(retrieve_data_package('VTT',      y, m, d, view='d', ey=ey, em=12, ed=31))
    weather_points.append(retrieve_data_package('GValley',  y, m, d, view='d', ey=ey, em=12, ed=31))
    weather_points.append(retrieve_data_package('SG',       y, m, d, view='d', ey=ey, em=12, ed=31))

#    pool = ProcessingPool(nodes=4)
#    pool.map(retrieve_data, weather_points)

    p = pmp.Pool(4)
    p.map(retrieve_data, weather_points)
    p.close()
    p.join()

    """
    pool = Pool(4)
    pool.map(retrieve_data, weather_points)
    pool.close()
    pool.join()


    for single_date in daterange(date(2014, 5, 11), date(2014, 10, 10)):
        #print date.strftime("%Y-%m-%d", single_date.timetuple())
        #print single_date.year, single_date.month, single_date.day

        y = single_date.year
        m = single_date.month
        d = single_date.day

        weather_points = list()
        weather_points.append(retrieve_data_package('SDH',      y, m, d, view='d',ey=2014, em=12, ed=31))
        weather_points.append(retrieve_data_package('VTT',      y, m, d, view='d',ey=2014, em=12, ed=31))
        weather_points.append(retrieve_data_package('GValley',  y, m, d, view='d',ey=2014, em=12, ed=31))
        weather_points.append(retrieve_data_package('SG',       y, m, d, view='d',ey=2014, em=12, ed=31))

        pool = Pool(4)
        pool.map(retrieve_data, weather_points)
        pool.close()
        pool.join()
    """