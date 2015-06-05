import types
import os
import time
# import serial
from datetime import datetime
import socket
import numpy as np
import pickle
import cPickle
import random
from scipy.spatial.distance import pdist,squareform

##################################################
# SERIAL TOOL
##################################################
SERIAL_DIR = "/sys/bus/usb/drivers/usb"

def serialScan(id):
	'''Scan all serial ports to find device that has matching id'''

	try:
		devs = os.listdir(SERIAL_DIR)
	except:
		print "%s not exist. check kernel version."%SERIAL_DIR
		return "none"

	for dev in devs:
		serialdir = "%s/%s/serial"%(SERIAL_DIR, dev)
		try:
			_id = open(serialdir, "r").readline().rstrip("\n")
		except:
			continue
		if _id == id:
			tmp = "%s/%s/%s:1.0"%(SERIAL_DIR, dev, dev)
			try:
				items = os.listdir(tmp)
			except:
				continue
			for item in items:
				if item.startswith("ttyUSB"):
					return "/dev/%s"%item
	return None

def serialConnect(alias, port, baud, timeout=1.0):
	ser = serial.Serial(port, baud, timeout=timeout)
	now = datetime.now().replace(microsecond=0)
	cprint("[%s] %s connected to %s\n"%(now, alias, port), "b")
	return ser

def serialDisconnect(alias, ser, msg):
	ser.close()
	now = datetime.now().replace(microsecond=0)
	cprint("[%s] %s disconnected from %s: %s\n"%(now, alias, ser.port, msg), "r")

def serialTx(port, rate, timeout, message):
	ser = serial.Serial(port, rate, timeout)
	ser.write(message)

def serialRx(port, rate, timeout):
	ser = serial.Serial(port, rate, timeout)
	while True:
		data = ser.readline()
		print data

##################################################
# UDP TOOL
##################################################
def udpRx(ip, port):
	'''Simple UDP transmission template'''
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	sock.bind((ip, port))
	while True:
		data, addr = sock.recvfrom(1000)
		print data

def udpTx(ip, port, message):
	'''Simple UDP receiving template'''
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	sock.sendto(message, (ip, port))

##################################################
# READ FILE
##################################################
def read(filename, index, delim="", ctype=""):
	openfile = open(filename, "r")
	lines = openfile.readlines()
	openfile.close()

	_ctype = type(ctype)
	value = []

	for line in lines:
		tmp = line.rstrip().rsplit(delim)
		for i in index:
			val = tmp[i]
			if _ctype == types.IntType:
				convertedval = int(val)
			elif _ctype == types.FloatType:
				convertedval = float(val)
			elif _ctype == types.StringType:
				convertedval = val
			elif _ctype == types.BooleanType:
				convertedval = bool(val)
			else:
				convertedval = val
			value.append(convertedval)

	return value

##################################################
# COLOR PRINT
##################################################
color = {"p":"\033[95m", "b":"\033[94m", "g":"\033[92m", 
		 "y":"\033[93m", "r":"\033[91m", "end":"\033[0m",
		 "k":"\033[90m", "c":"\033[96m", "w":"\033[97m"}

def cprint(message, colorname):
	'''Print eye candy colored text to terminal'''
	print "%s%s%s"%(color[colorname.lower()], message, color["end"])

##################################################
# BINARY OBJECT
##################################################

def saveObjectBinary(obj, filename):
	with open(filename, "wb") as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def loadObjectBinary(filename):
	with open(filename, "rb") as input:
		obj = pickle.load(input)
	return obj

### Faster version of loading and saving objects into binaries
### using cPickle library (modified by Khiem)
def saveObjectBinaryFast(obj, filename):
	with open(filename, "wb") as output:
		cPickle.dump(obj, output, cPickle.HIGHEST_PROTOCOL)

def loadObjectBinaryFast(filename):
	with open(filename, "rb") as input:
		obj = cPickle.load(input)
	return obj

##################################################
# OTHERS
##################################################

def str2datetime(s):
	parts = s.split('.')
	dt = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
	
	if len(parts) == 1:
		return dt
	elif len(parts) == 2:
		return dt.replace(microsecond=int(parts[1]))

##################################################
# KALMAN FILTER
##################################################

def kalman(y, Q, R):
	# Q: process variance
	# R: estimate of measurement variance

	length = len(y)
	yhat = np.zeros(length)      	# a posteri estimate of x
	P = np.zeros(length)         	# a posteri error estimate
	yhatminus = np.zeros(length) 	# a priori estimate of x
	Pminus = np.zeros(length)    	# a priori error estimate
	K = np.zeros(length)         	# gain or blending factor

	yhat[0] = y[0]
	P[0] = 1.0
	for k in range(1, length):
	    # time update
	    yhatminus[k] = yhat[k-1]
	    Pminus[k] = P[k-1]+Q

	    # measurement update
	    K[k] = Pminus[k]/( Pminus[k]+R )
	    yhat[k] = yhatminus[k]+K[k]*(y[k]-yhatminus[k])
	    P[k] = (1-K[k])*Pminus[k]

	return yhat

##################################################
# INTERPOLATE
##################################################

def interpolate(data, ts, newts):
	i, j = 0, 0
	newdata = []

	for j in range(len(newts)):
		if newts[j] < ts[1]:
			a = 0
		elif newts[j] > ts[-2]:
			a = -2
		else:
			while ts[i+1] < newts[j]:
				i += 1
			a = i
		tmp = 1.0 * (data[a+1] - data[a]) * (newts[j] - ts[a+1]) / (ts[a+1] - ts[a]) + data[a+1]
		newdata.append(tmp)
	return newdata

##################################################
# MOVING AVERAGE
##################################################

def average(data, wsize):
	newdata = []
	for i in range(len(data)):
		end = max(i, wsize)
		s = sum(data[end - wsize:end])
		newdata.append(s * 1.0 / wsize)
	return newdata

##################################################
# DIFFERENTIATION
##################################################

def df(data):
	return [data[i+1] - data[i] for i in range(len(data)-1)]

##################################################
# MISC.
##################################################
def rand(fromval, toval):
	return fromval + random.random() * (toval - fromval)

def drange(start, stop, step):
	tmp = []
	currval = start
	while currval < stop:
		tmp.append(currval)
		currval += step

	return tmp

def diff(a, b):
	assert len(a) == len(b)
	return [a[i] - b[i] for i in range(len(a))]

##################################################
# GENERATE RANDOM EUCLIDEAN DISTANCE MATRIX
# (added by Khiem)
##################################################
def generate_random_distance_matrix(n):
	### generate n random (x,y) points
	xs = np.random.rand(n)
	ys = np.random.rand(n)    
	points_vector = np.array(zip(xs,ys))    
	   
	dist_mat = squareform(pdist(points_vector, 'euclidean'))
	return dist_mat
