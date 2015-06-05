import numpy as np
import matplotlib.pyplot as plt

import mytool as mt
import bar_chart

# Load from binaries
avgsensor_names = mt.loadObjectBinary("tmp/avgsensor_names.bin")
Conditions_dict = mt.loadObjectBinary("tmp/Conditions_dict.bin")
Events_dict = mt.loadObjectBinary("tmp/Events_dict.bin")
wf_tuple_t = mt.loadObjectBinary("tmp/wf_tuple_t.bin")
wf_tuple_d = mt.loadObjectBinary("tmp/wf_tuple_d.bin")
wf_tuple_h = mt.loadObjectBinary("tmp/wf_tuple_h.bin")
wf_tuple_e = mt.loadObjectBinary("tmp/wf_tuple_e.bin")
wf_tuple_c = mt.loadObjectBinary("tmp/wf_tuple_c.bin")
tf_tuple_mth = mt.loadObjectBinary("tmp/tf_tuple_mth.bin")
tf_tuple_wday = mt.loadObjectBinary("tmp/tf_tuple_wday.bin")
tf_tuple_dhr = mt.loadObjectBinary("tmp/tf_tuple_dhr.bin")

sensor_no = len(avgsensor_names)

# Convert 'inf' to 1
sen_t = [1 if val == float("inf") else val for val in wf_tuple_t[3]]
sen_d = [1 if val == float("inf") else val for val in wf_tuple_d[3]]
sen_h = [1 if val == float("inf") else val for val in wf_tuple_h[3]]
sen_e = [1 if val == float("inf") else val for val in wf_tuple_e[3]]
sen_c = [1 if val == float("inf") else val for val in wf_tuple_c[3]]

SEN = [[sen_t[i], sen_d[i], sen_h[i], sen_e[i], sen_c[i]] for i in range(sensor_no)]
TOTAL_SEN = np.array([sum(SEN[i]) for i in range(sensor_no)])
idx = np.argsort(TOTAL_SEN)[-15:] # Best 15 sensors

data = [[TOTAL_SEN[i] for i in idx]] * 8
labels = [[avgsensor_names[i] for i in idx]] * 8
titles = ["Month", "Day", "Hour", "Temperature", "Dew Point", "Humidity", "Events", "Conditions"]
colors = ["b" if i < 3 else "g" for i in range(8)]

bar_chart.plot(data, labels, titles, colors, grid=True, savefig="bar.png", savereport="bar.csv")

plt.show()
