__author__ = 'almightykim'

class WeatherData(object):
    def __init__(self):
        self.weather_dict = dict()

class Analysis(object):
    def __init__(self):
        self.avg = list()
        self.diff = list()

class BNAnalData(object):
    def __init__(self):
        self.avg = WeatherData()
        self.diff = WeatherData()
        self.analysis = Analysis()

    def __getitem__(self, key):
        if key == 'avg':
            return self.avg
        elif key == 'diff':
            return self.diff
        elif key == 'analysis':
            return self.analysis
        else:
            return None


class BuildingObject(object):
    """
    building object template

    """
    def __init__(self, bldg_tag):
        self.bldg_tag = bldg_tag
        self.Condictions_dict = None
        self.Event_dict = None
        self.sigtags = dict()
        self.analysis = dict()
        self.anal_out = dict()

class BuildingOptState(object):
    """
    building object analysis optional problem/state set.
    """
    def __init__(self):
        self.optprob_set = list()
        self.optstate_set = list()


class BuildingSigtagProperty(object):
    """
    building sig_tag property.
    """
    def __init__(self, sig_tag, data_state_mat, data_weather_mat, data_time_mat, time_slot, data_exemplar, data_zvar, sensor_names, weather_names, time_names, p_idx, p_names):
        self.sig_tag = sig_tag
        self.data_state_mat = data_state_mat
        self.data_weather_mat = data_weather_mat
        self.data_time_mat = data_time_mat
        self.time_slot = time_slot
        self.data_exemplar = data_exemplar
        self.data_zvar = data_zvar

        self.names = dict()
        self.names['sensor'] = sensor_names
        self.names['weather'] = weather_names
        self.names['time'] = time_names

        self.p_idx = p_idx
        self.p_names = p_names
        #TODO remove clarify var name. This is for conversion
        self.data_weather_mat_ = None


class BuildingAnalysis(object):
    """
    building object analysis.
    """
    def __init__(self, p_name):
        self.sensor_tag = p_name
        self.peak_eff_state = list()
        self.attrs = dict()
        self.attrs['sensor'] = BuildingOptState()
        self.attrs['time'] = BuildingOptState()
        self.attrs['weather'] = BuildingOptState()




